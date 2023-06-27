'''Training Dqn that can do collision Avoidance

Training methodology: Vanilla Training (No scheduling/curriculum learning)
'''
'''Complete Information DQN (CIdqn)
'''

import glob
import imp
import math
import gc
import os
from sre_constants import SUCCESS
import time
import datetime
import pybullet as p
import cv2
import numpy as np
from graphviz import Digraph
import argparse
import random
import torch
import matplotlib.pyplot as plt
from time import sleep
import copy

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from Config.constants import (
    GRIPPER_PUSH_RADIUS,
    PIXEL_SIZE,
    PUSH_DISTANCE,
    WORKSPACE_LIMITS,
    TARGET_LOWER,
    TARGET_UPPER,
    orange_lower,
    orange_upper,
    BG_THRESHOLD,
    MIN_GRASP_THRESHOLDS
)

from Environments.environment_sim import Environment
import Environments.utils as env_utils
from V1_destination_prediction.Test_cases.tc4_collisions import TestCase4

from create_env import get_push_start2, get_max_extent_of_target_from_bottom


from collections import namedtuple, deque

from V2_next_best_action.models.caware_dqn import CAwarePushDQN
from Environments.utils import sample_goal, get_pose_distance
import wandb

# wandb setup
number = 1
NAME = "model" + str(number)
ID = 'Vision-Dqn' + str(number)
run = wandb.init(project='CAware_Dqns', name = NAME, id = ID)


torch.cuda.empty_cache()

Transition = namedtuple('Transition', ('state_height', 'state_linear', 'action', 'next_state_height', 'next_state_linear', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        
        '''Save a transition'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 10 #128
GAMMA = 0.9 # 0.999 # Discount factor
EPS_START = 0.9 # Random action choosing probability starts with this value and decays until EPS_END
EPS_END = 0.05 # Random action choosing probability starts at EPS_START and decays until EPS_END
EPS_DECAY = 200 # Decay rate of random action choosing probability, with the passage of episodes and time
TARGET_UPDATE = 10
TARGET_SAVE_CHECKPOINTS = [200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
SAVE_FREQ = 1000
REPLAY_MEMORY_SIZE = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_observations = 6 # 3 for initial state, 3 for goal state
n_actions = 16 # 16 push + 1 grasp

policy_net = CAwarePushDQN(use_cuda=True).to(device)
target_net = CAwarePushDQN(use_cuda=True).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(REPLAY_MEMORY_SIZE) # 10000

steps_done = 0


def select_action(state_height, state_linear):
    '''Select the next best action 
    state: tensor(shape=(6))
    '''
    global steps_done
    sample = random.uniform(0.0, 1.0) # random.randint(a=0, b=16) 
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0*steps_done / EPS_DECAY)
    steps_done += 1

    if sample>eps_threshold:
        with torch.no_grad():
            return policy_net(state_height, state_linear).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def get_reward5(prev_state, current_state, prev_states, cur_states):
    '''
    prev_state: (x1, y1, theta1, x2, y2, theta2)
    current_state: (x3, y3, theta3, _, _, _)
    '''

    # print("Rewarding ----------------------------")
    pos_diff = np.linalg.norm(prev_state[0:2] - prev_state[3:5]) - np.linalg.norm(current_state[0:2] - prev_state[3:5])
    # print(prev_states)
    # print(prev_states.shape, cur_states.shape)
    col_reward = 0
    if len(prev_states) > 0:
        col_reward = -1 * np.sum(np.sum(np.linalg.norm(prev_states - cur_states, axis=1)))
    reward = pos_diff + col_reward # 0.1*orn_diff  #np.linalg.norm(prev_state[0:3] - prev_state[3:6]) - np.linalg.norm(current_state[0:3] - prev_state[3:6]) # prev distance - current distance
    # print(f"Position Diff: {pos_diff}\tOrn Diff: {0.1*orn_diff}\nReward Aggregate: {reward}")
    return reward

def optimize_model(timestep=0, batch_num=0, reward=0):
    if len(memory) < BATCH_SIZE:
        return
    # print("Optimization!")
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state_linear)), device=device, dtype=torch.bool)
    non_final_next_states_linear = None
    non_final_next_states_height = None
    if torch.sum(non_final_mask)>0:
        non_final_next_states_linear = torch.cat([s for s in batch.next_state_linear
                                                    if s is not None])
        non_final_next_states_height = torch.cat([s for s in batch.next_state_height
                                                    if s is not None])

    state_linear_batch = torch.cat(batch.state_linear)
    state_height_batch = torch.cat(batch.state_height)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_height_batch, state_linear_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states_height, non_final_next_states_linear).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    wandb.log({'loss': loss, 'reward': reward_np, 'timestep': timestep}) 

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    # for param in policy_net.parameters():
    #     if param.grad == None:
    #         continue
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

from itertools import count

from Config.constants import MIN_GRASP_THRESHOLDS

is_viz = False

# env = Environment()
env = Environment(gui=False)
num_of_envs = 1000
max_num_of_actions = 30
is_viz = False
max_extent_threshold = 1 # Max extent threshold of the target object in pixel units
push_directions = [0, np.pi/8, np.pi/4, 3*np.pi/8, 
                    np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, 
                    np.pi, 9*np.pi/8, 5*np.pi/4, 11*np.pi/8,  
                    3*np.pi/2, 13*np.pi/8, 7*np.pi/4, 15*np.pi/8] # 16 standard directions
# num_episodes = 50 # 10
max_timesteps = num_of_envs*max_num_of_actions # 50050
timestep = 0
        
episodal_rewards = []
episodal_time = []

wandb.config.update({
    'max_timesteps': max_timesteps,
    'max_num_envs': num_of_envs,
    'max_num_actions': max_num_of_actions,
    'batch_size': BATCH_SIZE,
    'optimizer': 'Adam',
    'learning_rate': 'default',
    'replay_memory': REPLAY_MEMORY_SIZE, # 10000
    'n_actions': 16,
    'n_observations': "Height map + 6 linear (cur_state, goal_state)",
    'action_types': 'Only push in 16 different directions, with one push type',
    'push_types': 'POSMAX only'
})

import progressbar

widgets = ['Training: ', progressbar.Bar('-')]# progressbar.AnimatedMarker()]
bar = progressbar.ProgressBar(max_value=max_timesteps+1, widgets=widgets) #, widget_kwargs={}).start()

for env_num in range(num_of_envs):
    # Initialize the environment and state
    env.reset()
    testcase1 = TestCase4(env)
    body_ids, success = testcase1.sample_test_case(bottom_obj='default') #'random') # testcase1.create_standard()

    target_pos, target_orn = p.getBasePositionAndOrientation(body_ids[0])
    target_euler = p.getEulerFromQuaternion(target_orn)

    marker_pos, marker_orn = None, None
    goal_suc = False
    
    marker_pos, marker_orn = sample_goal(target_pos, target_orn, testcase1.current_target_size)
    if is_viz:
        marker_obj, goal_suc = testcase1.add_marker_obj(marker_pos, marker_orn, half_extents=testcase1.current_target_size/2)

    marker_euler = p.getEulerFromQuaternion(marker_orn)
    cur_target_st = np.array([target_pos[0], target_pos[1], target_euler[2]], dtype=np.float64)
    cur_target_goal = np.array([marker_pos[0], marker_pos[1], marker_euler[2]], dtype=np.float64)
    # cur_obj_size = testcase1.current_target_size[0:2]
    cur_state = np.hstack((cur_target_st, cur_target_goal)) #, cur_obj_size))
    color_image, depth_image, _ = env_utils.get_true_heightmap(env)
    depth_image = np.stack((depth_image, )*3, axis=-1)
    state = {
        'cur_state_height': torch.tensor(np.array([np.transpose(depth_image, (2, 0, 1))]), dtype=torch.float, device=device),
        'cur_state_linear': torch.tensor(cur_state, dtype=torch.float, device=device).unsqueeze(0),
    }
    done = False

    ep_reward = 0
    ep_start_time = timestep

    prev_states = [] # collision states
    for col_st in range(len(body_ids)-1):
        tpos, _ = p.getBasePositionAndOrientation(body_ids[col_st+1])
        prev_states.append(np.array(tpos))
    prev_states = np.array(prev_states)

    for t in count(): # Per episode
        # Select and perform an action
        timestep += 1
        bar.update(timestep)
        
        action = select_action(state['cur_state_height'], state['cur_state_linear']) # select_action(state['rgb'], state['height_map'])
        if action.item() in range(0, 16): # push action
            targetPos, targetOrn = p.getBasePositionAndOrientation(body_ids[0])
            current_target_obj_size = testcase1.current_target_size

            push_dir = push_directions[(action.item())%16] # Sample push directions
            push_type = 0 # 0 indicates position maximization 
            if action.item() >= 16:
                push_type = 1 # 1 indicates orientation maximization
            push_start, push_end = get_push_start2(push_type, push_dir, targetPos, targetOrn, current_target_obj_size, is_viz=False)
            env.push(push_start, push_end) # Action performed 
            
            target_pos, target_orn = p.getBasePositionAndOrientation(body_ids[0])
            euler_orn = p.getEulerFromQuaternion(target_orn)

            new_target_st = np.array([target_pos[0], target_pos[1], euler_orn[2]], dtype=np.float)
            new_state = np.hstack((new_target_st, cur_target_goal))
            
            cur_states = [] # Cur states
            for col_st in range(len(body_ids)-1):
                tpos, _ = p.getBasePositionAndOrientation(body_ids[col_st+1])
                cur_states.append(np.array(tpos))
            cur_states = np.array(cur_states)
            reward = get_reward5(current_state=new_state, prev_state=state['cur_state_linear'].squeeze().cpu().numpy(), prev_states=prev_states, cur_states=cur_states)

            if np.linalg.norm(new_target_st - cur_target_goal) < 0.05:
                done = True
                reward += 10

            prev_states = cur_states
        elif action.item()>=16:
            print("Invalid Action!!!!!")
            exit()
            
        
        targetPos, _ = p.getBasePositionAndOrientation(body_ids[0])
        # bottomPos, _ = p.getBasePositionAndOrientation(body_ids[0])
        if testcase1.check_target_within_workspace_bounds(targetPos) == False:
            reward -= 1
            # print("WAIT WHAAAAAAAAAAAAAAT")
            # print(targetPos, WORKSPACE_LIMITS)
            done = True

        reward_np = reward
        ep_reward += reward_np
        reward = torch.tensor([reward], dtype=torch.float, device=device)

        color_image, depth_image, _ = env_utils.get_true_heightmap(env)
        depth_image = np.stack((depth_image, )*3, axis=-1)
        
        if not done:
            target_pos, target_orn = p.getBasePositionAndOrientation(body_ids[0])
            euler_orn = p.getEulerFromQuaternion(target_orn)
            
            new_target_st = np.array([target_pos[0], target_pos[1], euler_orn[2]], dtype=float)
            new_state = np.hstack((new_target_st, cur_target_goal))
            next_state = {
                'cur_state_height': torch.tensor(np.array([np.transpose(depth_image, (2, 0, 1))]), dtype=torch.float, device=device),
                'cur_state_linear': torch.tensor(new_state, dtype=torch.float, device=device).unsqueeze(0), 
            }
        else:
            next_state = {
                'cur_state_height': None,
                'cur_state_linear': None
            }

        # Store the transition in memory
        memory.push(state['cur_state_height'], state['cur_state_linear'], action, next_state['cur_state_height'], next_state['cur_state_linear'], reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        
        optimize_model(timestep=timestep, batch_num=t, reward=reward_np)

    # Update the target network, copying all weights and biases in DQN
        if timestep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # print("Target updated")
            

        # if i_episode % TARGET_SAVE == 0 or i_episode==10:
        if timestep in TARGET_SAVE_CHECKPOINTS or timestep%SAVE_FREQ==0:
            print("Saved")
            SAVE_PATH = './V2_next_best_action/models/model_checkpoints/dqnv4/{}.pt'.format(timestep)
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), SAVE_PATH)
            SAVE_PATH = './V2_next_best_action/models/model_checkpoints/dqnv4/episodal_rewards_{}.npy'.format(timestep)
            with open(SAVE_PATH, 'wb') as f:
                np.save(f, np.array(episodal_rewards))

            SAVE_PATH = './V2_next_best_action/models/model_checkpoints/dqnv4/episodal_times_{}.npy'.format(timestep)
            with open(SAVE_PATH, 'wb') as f:
                np.save(f, np.array(episodal_time))

        torch.cuda.empty_cache()



        if t>=max_num_of_actions: # 30
            done = True
        
        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            break

    ep_end_time = timestep
    ep_time = ep_end_time - ep_start_time
    episodal_time.append(ep_time)
    episodal_rewards.append(ep_reward)

print('Complete')
# env.render()
# env.close()
# plt.ioff()
# plt.savefig('durations_count.png')
# plt.show()
run.finish()