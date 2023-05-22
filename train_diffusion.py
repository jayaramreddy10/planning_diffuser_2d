import argparse
from collections import OrderedDict as odict
import pickle
import os
import random
import socket
import time

import numpy as np
from models.diffusion_jayaram import GaussianDiffusion_jayaram
from models.temporal_model_jayaram import TemporalUnet_jayaram
from utils.training import Trainer_jayaram
from utils.arrays import batchify
from d4rl_maze2d_dataset.sequence import SequenceDataset, Maze2dDataset, RRT_star_traj
from utils.rendering import Maze2dRenderer

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


def train_network_single_sample(args, path, cond):
    training_start_time = time.time()

    action_dim = 2
    state_dim = 4
    transition_dim = state_dim + action_dim
    cond_dim = 4
    #load model architecture 
    model = TemporalUnet_jayaram(args.horizon, transition_dim, cond_dim)
    model = model.to(device)

    diffusion = GaussianDiffusion_jayaram(model, args.horizon, state_dim, action_dim, device)
    diffusion = diffusion.to(device)

    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#
    dataset = []
    dataset.append((path, cond))
    trainer = Trainer_jayaram(diffusion, dataset)

    for i in range(args.epochs):
        print(f'Epoch {i} / {args.epochs} | {args.save_ckpt}')
        trainer.train(device, i, n_train_steps=args.number_of_steps_per_epoch)

    # Save results
    # if save_results:
    #     # Rename the final training log instead of re-writing it
    #     training_log_path = os.path.join(args.output_dir, "training_log.pkl")
    #     os.rename(epoch_training_log_path, training_log_path)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
    print("Done.")
    print("")
    print("Total training time: {} seconds.".format(time.time() - training_start_time))
    print("")

def train_network(args, dataset):
    training_start_time = time.time()

    action_dim = 0
    state_dim = 2
    transition_dim = state_dim + action_dim
    cond_dim = 2
    #load model architecture 
    model = TemporalUnet_jayaram(args.horizon, transition_dim, cond_dim)
    model = model.to(device)

    diffusion = GaussianDiffusion_jayaram(model, args.horizon, state_dim, action_dim, device)
    diffusion = diffusion.to(device)

    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#

    # renderer = Maze2dRenderer('maze2d-large-v1')
    trainer = Trainer_jayaram(diffusion, dataset)

    for i in range(args.epochs):
        print(f'Epoch {i} / {args.epochs} | {args.save_ckpt}')
        trainer.train(device, i, n_train_steps=args.number_of_steps_per_epoch)

    # Save results
    # if save_results:
    #     # Rename the final training log instead of re-writing it
    #     training_log_path = os.path.join(args.output_dir, "training_log.pkl")
    #     os.rename(epoch_training_log_path, training_log_path)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
    print("Done.")
    print("")
    print("Total training time: {} seconds.".format(time.time() - training_start_time))
    print("")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False


def get_toy_dataset(num_samples, traj_len, bounds = np.array([[-1, -1], [1, 1]])):
    
    start = np.random.uniform(low = bounds[0], high = bounds[1], size = (num_samples, 2))
    goal = np.random.uniform(low = bounds[0], high = bounds[1], size = (num_samples, 2))

    n = traj_len

    line = goal - start

    dist = np.linalg.norm(line, axis = 1)

    point_dists = np.linspace(0., dist, n).T

    theta = np.arctan2(line[:, 1], line[:, 0])
    theta = np.expand_dims(theta, axis = 1)

    delta_x = np.multiply(point_dists, np.cos(theta))
    delta_y = np.multiply(point_dists, np.sin(theta))

    delta_x = np.expand_dims(delta_x, axis = 2)
    delta_y = np.expand_dims(delta_y, axis = 2)

    delta = np.concatenate([delta_x, delta_y], axis = 2)

    trajectories = np.expand_dims(start, axis = 1) + delta

    return trajectories.copy()


if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument(
    #     "-i", "--input-data-path", help="Path to training data."
    # )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Path to output directory for training results. Nothing specified means training results will NOT be saved.",
    )

    parser.add_argument(
        "-ar",
        "--architecture-config",
        type=str,
        default="logs/maze2d-large-v1/diffusion/H384_T256/diffusion_config.pkl",
        help="Path to a configuration file that describes the neural network architecture configuration for diffusion",
    )

    parser.add_argument(
        "-save_ckpt",
        "--save_ckpt",
        type=str,
        default="/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/logs/maze2d-large-v1/diffusion",
        help="save checkpoints of diffusion model while training",
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=2, help="Number of epochs to train."
    )

    parser.add_argument(
        "-n_steps_per_epoch",
        "--number_of_steps_per_epoch",
        type=int,
        default=200,
        help="Number of steps per epoch",
    )

    parser.add_argument(
        "-hr",
        "--horizon",
        type=int,
        default=384,
        help="Horizon or no of waypoints in path",
    )

    # parser.add_argument(
    #     "-b",
    #     "--batch-size",
    #     type=int,
    #     required=True,
    #     help="The number of samples per batch used for training.",
    # )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0002,
        help="The learning rate used for the optimizer.",
    )

    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=8,
        help='The number of subprocesses ("workers") used for loading the training data. 0 means that no subprocesses are used.',
    )

    # parser.add_argument(
    #     "-rt",
    #     "--retrain-network",
    #     type=str2bool,
    #     default=False,
    #     help="Retraines network after unfreezing last few modules.",
    # )

    args = parser.parse_args()

    #generate dataset

    # n_samples = 100
    # dataset = get_toy_dataset(n_samples, args.horizon)

    # #fetch first sample path in batch format, condition on terminal states as a dict
    # path = torch.tensor(dataset[0][None, :, :]).float()
    # path = path.to(device)

    # cond = {}
    # cond[0] = torch.tensor(dataset[0][0][None, :]).float().to(device)
    # cond[args.horizon - 1] = torch.tensor(dataset[0][args.horizon - 1][None, :]).float().to(device)

    # dataset = SequenceDataset()
    # batch = batchify(dataset[0])

    #generate collision free trajs
    n_trajs = 10
    trajs = []
    for i in range(n_trajs):
        RRT_Star = RRT_star_traj()
        traj, cond, val = RRT_Star.generate_traj()    #higher the val, higher the reward (so if val is high, path len is less which is optimal)
        trajs.append((traj, cond))

    #generate remaining trajs

    dataset = Maze2dDataset(trajs, n_trajs = len(trajs))
    # batch = batchify(dataset[0])
	# batch[0].shape: (1, 384, 6)
	# batch[0] len : 384 (horizon)

	# batch[1]: 2 elements in dict: (0, 383) as initial, final states are fixed in n_horizon
	# {0: array([ 0.04206157, ...e=float32), 383: array([ 0.31135416, ...e=float32)}   , each with dim = (4, )

    #-----------------------------------------------------------------------------#
    #------------------------ test forward & backward pass -----------------------#
    #-----------------------------------------------------------------------------#
    # change action dim to 2 later
    action_dim = 0
    state_dim = 2
    transition_dim = state_dim + action_dim
    cond_dim = 2

	# batch[0].shape: (1, 384, 6)
	# batch[0] len : 384 (horizon)

	# batch[1]: 2 elements in dict: (0, 383) as initial, final states are fixed in n_horizon
	# {0: array([ 0.04206157, ...e=float32), 383: array([ 0.31135416, ...e=float32)}   , each with dim = (4, )

    #-----------------------------------------------------------------------------#
    #------------------------ test forward & backward pass -----------------------#
    #-----------------------------------------------------------------------------#
    # #load model architecture 
    # model = TemporalUnet_jayaram(args.horizon, transition_dim, cond_dim)
    # model = model.to(device)

    # diffusion = GaussianDiffusion_jayaram(model, args.horizon, state_dim, action_dim, device)
    # diffusion = diffusion.to(device)

    # print('Testing forward...', end=' ', flush=True)

    # # loss = diffusion.loss(*batch, device)     #forward pass
    # loss, _ = diffusion.loss(*batch, device)     #forward pass
    # loss.backward()                      #backward pass
    # print('✓')

    # # Train the network on just first sample
    # train_network_single_sample(args, *batch)

    # Train the network on many samples
    train_network(args, dataset)