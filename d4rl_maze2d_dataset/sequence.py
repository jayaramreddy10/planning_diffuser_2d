from collections import namedtuple
import numpy as np
import torch
import pdb
import os
from torch.utils.data import Dataset
import math
import random
import sys
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pathlib

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

from .upsample import *
from .RRTStar import RRTStar

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='maze2d-large-v1', horizon=384,
        normalizer='LimitsNormalizer', preprocess_fns=['maze2d_set_terminals'], max_path_length=40000,
        max_n_episodes=10000, termination_penalty=0, use_padding=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class RRT_star_traj():
    def __init__(self):
        # ====Search Path with RRT====
        self.obstacle_list = [   #in 2d exmaple of [(-1,-1) to (1, 1)]
            (-1/2, 1/2, 0.3),
            (2/3,0,0.3),
            (-1/6,-1/2,0.3)

        ]  # [x,y,size(radius)]
        self.min_traj_len = 0.1
        self.num_waypoints = 384
        self.pathres = 0.02
        self.iet_per_sample = 500   #consider path which has least cost (best of 500) for the same (strt, goal) points

    def generate_traj(self):   #generate just single path
        start = self.gengoal(obstacle_list=self.obstacle_list,min_dist=0.05)  #(x,y ) are generated randomlt in 2d space ouside obstacles with min dist from the cetre 0.05
        goal = self.gengoal(obstacle_list=self.obstacle_list,min_dist=0.05)

        #for binary image


        while( (start[0]-goal[0])**2 + (start[1]-goal[1])**2 < self.min_traj_len):   #check for min traj len b/w (strt, end) points
            goal = self.gengoal(obstacle_list=self.obstacle_list,min_dist=0.05)

        print("start : ",start, "\n" + "goal : ", goal)
        # start = (-0.75,-0.75)
        # goal = (0.75,-0.75)

        # Set Initial parameters
        rrtobj = RRTStar(
            start=start,
            goal=goal,
            rand_area=[-1, 1],
            obstacle_list=self.obstacle_list,
            expand_dis=0.1,
            robot_radius=0.05,
            path_resolution=0.02,
            max_iter=300)

        min_len = math.inf
        path_found = False
        for _ in range(self.iet_per_sample):
            path = rrtobj.planning(animation=False)

            # print(path)


            if path is None:
                continue
            else:
                # print("found path!!")

                (samples,l_traj) = generate_samples(path,self.num_waypoints,self.pathres)


                if(len(samples)!=self.num_waypoints):
                    # print("invalid traj length (", len(samples),"), skipping")
                    continue


                if(l_traj < min_len):
                    traj = samples
                    min_len = l_traj

                path_found = True

                # plt.plot([x for (x, y) in samples], [y for (x, y) in samples])
                # plt.grid(True)
                # plt.xlim([-1,1])
                # plt.ylim([-1,1])
        
        if(path_found == False):
            print('cannot find path b/w given start and goal points')

        cond = {0 : np.array(start), self.num_waypoints - 1: np.array(goal)}

        # print('len of traj :{}'min_len)

        # for (ox, oy, size) in self.obstacle_list:
        #     deg = list(range(0, 360, 5))
        #     deg.append(0)
        #     xl = [ox + size * math.cos(np.deg2rad(d)) for d in deg]
        #     yl = [oy + size * math.sin(np.deg2rad(d)) for d in deg]
        #     plt.plot(xl, yl)

        #     plt.plot(start[0],start[1],marker="s")
        #     plt.plot(goal[0],goal[1],marker="o")

        # plt.plot([x for (x, y) in traj], [y for (x, y) in traj])
        # plt.grid(True)
        # plt.xlim([-1,1])
        # plt.ylim([-1,1])
        
        # plt.show()  
        return np.array(traj), cond, -1.0 * min_len

    def gengoal(self, obstacle_list,min_dist=0):

        p = (round(random.uniform(-1,1),3), round(random.uniform(-1,1),3))

        collision = False
        for obst in obstacle_list:  # obst = [x,y,size(radius)]
            if( (obst[0] - p[0])**2 + (obst[1] - p[1])**2 <= (obst[2] + min_dist)**2 ):
                collision = True

        if(collision):
            p = self.gengoal(obstacle_list=obstacle_list,min_dist=min_dist)

        return p

class RRT_star_traj_binary(object):

    def __init__(self, map, q_init, q_final):
        # Map: 0 for barrier, 1 for free space
        self.map = map[:,:].astype(int).T
        self.SIZE = 1024 # Size of the world (1024*1024)

        # self.q_init = [40.0, 60.0] # Start point
        # self.q_final = [60.0, 40.0] # End point

        self.q_init = q_init # Start point
        self.q_final = q_final # End point

        self.delta_q = 0.2 # Percentage of length to keep between q_rand and q_near when generating new point q_new
        self.dis_num = 100 # Discretization number of line collision checking

        # Store the final route from the start point to end point
        self.qs = [] # Store the [x,y] coordinate of each point
        self.qs_parent = [] # Store the index of each point's parent
        self.qs.append(self.q_init)
        self.qs_parent.append(0)

        self.fig = plt.figure()


    def random_vertex_generate(self):
        # Generate random vertex
        q_rand = [self.SIZE*np.random.rand(), self.SIZE*np.random.rand()]
        return q_rand


    def collision_check_point(self, point):
        # Check if a point is in collision with the geometry
        if self.map[int(math.floor(point[0])), int(math.floor(point[1]))] == 0:
            # Collision detected
            return True
        else:
            # Non-collision
            return False


    def collision_check_line(self, q_start, q_end):
        # Check if a line is in collision with the geometry

        # Step length
        dx = (q_end[0] - q_start[0]) / self.dis_num
        dy = (q_end[1] - q_start[1]) / self.dis_num
        q = [int(q_start[0]), int(q_start[1])]
        for i in range(self.dis_num):
            x = q_start[0] + i * dx
            y = q_start[1] + i * dy
            # Skip repeated points
            if int(x)==q[0] and int(y)==q[1]:
                continue
            q = [int(x), int(y)]
            if self.collision_check_point(q) == True:
                return True

        # Collision check of q_end
        if self.collision_check_point(q_end) == True:
            return True

        return False


    def nearest_vertex_check(self, q_rand):
        # Search for the nearest existing point

        # Maximum length in the map
        min_length = 2 * self.SIZE ** 2

        # Find the nearest point in Euclidean distance
        for index, q in enumerate(self.qs):
            length = (q_rand[0]-q[0]) ** 2 + (q_rand[1]-q[1]) ** 2
            if length < min_length:
                # print('jai')
                min_length = length
                index_near = index
                q_near = q
        return index_near, q_near


    def new_point_generate(self, q_near, q_rand, index_near):
        # Generate q_new according to delta_q
        x = q_near[0] + (q_rand[0]-q_near[0]) * self.delta_q
        y = q_near[1] + (q_rand[1]-q_near[1]) * self.delta_q
        q_new = [x,y]

        # Write into qs & q_parent
        self.qs.append(q_new)
        self.qs_parent.append(index_near)

        # Plot the new generated line
        self.new_point_plot(q_near, q_new)

        return q_new


    def connection_check(self, q_new):
        # Check if the new vertex can connect with the final point
        if not self.collision_check_line(q_new, self.q_final):
            self.qs.append(self.q_final)
            self.qs_parent.append(len(self.qs)-2)
            # Plot the line connect to the end point
            self.new_point_plot(q_new, self.q_final)
            return True
        else:
            return False

    def new_point_plot(self, q_near, q_new):
        # Plot the new generated line
        verts=[q_near, q_new]
        codes=[Path.MOVETO,Path.LINETO]
        path = Path(verts, codes)
        patch = patches.PathPatch(path)
        ax = self.fig.add_subplot(111)
        ax.add_patch(patch)


    def figure_generate(self):
        # Create figure
        ax = self.fig.add_subplot(111)

        # Plot the N_map
        plt.imshow(self.map.T)

        # Plot the initial and end points
        ax.plot(self.q_init[0], self.q_init[1], '*')
        ax.plot(self.q_final[0], self.q_final[1], '*')

        #plot the route from start point to end point
        i = len(self.qs)-1
        while (True):
            x = [self.qs[i][0], self.qs[self.qs_parent[i]][0]]
            y = [self.qs[i][1], self.qs[self.qs_parent[i]][1]]
            ax.plot(x, y, "r")
            i = self.qs_parent[i]
            if i == 0:
                break

        plt.show()

    def return_traj(self):
        traj = []
        #plot the route from start point to end point
        i = len(self.qs)-1
        while (True):
            x = self.qs[i][0]
            y = self.qs[i][1]
            traj.append(np.array((x, y)))
            i = self.qs_parent[i]
            if i == 0:
                traj.append(np.array((self.qs[i][0], self.qs[i][1])))
                break
        
        return np.array(traj)

class Maze2dDataset_Value(Dataset):   #this class handles a single maze2d env
    def __init__(self, trajs, is_train = True, n_trajs = 30000, transform = None):
        # self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        self.n_trajs = n_trajs
        self.trajs = trajs

    def __len__(self):
        return self.n_trajs

    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.to_list()
        traj = self.trajs[idx][0]  #shape: ([1, 384, 2])

        #get cond for the traj    //dict
        cond = self.trajs[idx][1]

        #get value for the traj
        val = self.trajs[idx][2]
        return traj, cond, val
    
class Maze2dDataset(Dataset):   #this class handles a single maze2d env
    def __init__(self, trajs, is_train = True, n_trajs = 30000, transform = None):
        # self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        self.n_trajs = n_trajs
        self.trajs = trajs

    def __len__(self):
        return self.n_trajs

    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.to_list()
        traj = self.trajs[idx][0]  #shape: ([1, 384, 2])

        #get cond for the traj    //dict
        cond = self.trajs[idx][1]

        return traj, cond


