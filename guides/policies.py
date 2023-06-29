from collections import namedtuple
# import numpy as np
import torch
import torch.nn as nn
import einops
import pdb

import diffuser.utils as utils
# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')
# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')


class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad

class Policy:

    def __init__(self, diffusion_model, action_dim, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.action_dim = action_dim
        # self.normalizer = normalizer
        # self.action_dim = normalizer.action_dim
        self.sample_kwargs = sample_kwargs

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        # conditions = utils.apply_dict(
        #     self.normalizer.normalize,
        #     conditions,
        #     'observations',
        # )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=10):


        conditions = self._format_conditions(conditions, batch_size)   #normalizing the conditional points

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        denoised_trajs_stack = list(self.diffusion_model(conditions, **self.sample_kwargs))
        # sample, denoised_diffusion_trajs_stack = self.diffusion_model(conditions, **self.sample_kwargs)
        denoised_trajs_stack[0] = utils.to_np(denoised_trajs_stack[0])

        ## extract action [ batch_size x horizon x transition_dim ]
        # actions = sample[:, :, :self.action_dim]
        # actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        # action = actions[0, 0]

        # if debug:
        # denoised_trajs_stack[0] = denoised_trajs_stack[0][:, :, self.action_dim:]
        # normed_observations = sample[:, :, self.action_dim:]
        # observations = self.normalizer.unnormalize(normed_observations, 'observations')

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        # trajectories = Trajectories(actions, observations)
        return denoised_trajs_stack
        # return denoised_traj, denoised_diffusion_trajs_stack
        # else:
        #     return action

class GuidedPolicy:

    def __init__(self, guide, diffusion_model, action_dim, sample_fn, **sample_kwargs):
        self.guide = guide   #value fn model
        self.diffusion_model = diffusion_model
        self.action_dim = action_dim    #0
        self.sample_fn = sample_fn
        self.sample_kwargs = sample_kwargs

    def __call__(self, cond, batch_size=1, verbose=True):   #batch_size:64
        ## run reverse diffusion process
        cond = self._format_conditions(cond, batch_size)   #normalizing the conditional points

        # sample, denoised_diffusion_trajs_stack = self.diffusion_model(cond, guide=self.guide, verbose=verbose, sample_fn = self.sample_fn, **self.sample_kwargs)
        denoised_trajs_stack = list(self.diffusion_model(cond, guide=self.guide, verbose=verbose, sample_fn = self.sample_fn, **self.sample_kwargs))
        denoised_trajs_stack[0] = utils.to_np(denoised_trajs_stack[0])
        return denoised_trajs_stack
        # return trajectory, denoised_diffusion_trajs_stack

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        # conditions = utils.apply_dict(
        #     # self.normalizer.normalize,
        #     conditions,
        #     'observations',
        # )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
