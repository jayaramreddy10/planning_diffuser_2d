import numpy as np
import torch
import os
from torch import nn
import pdb
from utils.progress import Progress, Silent
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)
from utils.arrays import to_torch, to_device
import sys
sys.path.append("/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning")
from compute_value_guide_grad import n_step_guided_p_sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values

class GaussianDiffusion_jayaram(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, device, n_timesteps=256,
        loss_type='l2', clip_denoised=True, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.model = model   #this is temporalUNET, get this from model.pkl file
        self.observation_dim = observation_dim
        self.action_dim = 0   #lets not consider action for now
        self.transition_dim = observation_dim + action_dim

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))


        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights).to(device)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=True, sample_fn=default_sample_fn, guidance = False, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        #do activation maximization
        traj_path = '/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/test_guided_diffusion'
        x = np.load(os.path.join(traj_path, 'trajectory_without_guidance_trajectory_0.npy'))
        x = to_torch(x)
        x = to_device(x, device)
        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        # denoised_trajs = []
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if guidance: 
                x = sample_fn(self, x, cond, timesteps, **sample_kwargs)    #kwargs has (guide, sample_fn)
            else: 
                x = self.p_sample(x, cond, timesteps)

            x = apply_conditioning(x, cond, self.action_dim)
            # denoised_trajs.append(x)
            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod.to(device), t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod.to(device), t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):   #x_start: [32, 384, 6] , cond: dict of states (0, 383) of len 4, t is random timestep in entire horizon (say 155)
        #here x_start means start timestep of forward diffusion (not the start state of agent in the current path)
        noise = torch.randn_like(x_start)    #[32, 384, 6]   #gaussian dist

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)      #[32, 384, 6]  -- forward pass of diffusion
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)    #fix x_noisy[0][0], x_noisy[0][383] with start and goal points i.e cond[0], cond[383]

        x_recon = self.model(x_noisy, cond, t)   #[32, 384, 6] using Temporal UNET   (error: expected double but got float)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise)
            # loss, info = self.loss_fn(x_recon, noise)
        else:
            loss = self.loss_fn(x_recon, x_start)
            # loss, info = self.loss_fn(x_recon, x_start)

        return loss
        # return loss, info

    def loss(self, x, *args):   #x.shape: (b, 384, 6) , cond : batch[1], cond[0].shape: (b, 4) and cond[1].shape: (b, 4)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()    #choose a random timestep uniformly in reverse diffusion process
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
    
class ValueDiffusion_jayaram(GaussianDiffusion_jayaram):

    def p_losses(self, x_start, cond, target, t):   #target is scalar
        noise = torch.randn_like(x_start)    #torch.Size([32, 4, 23])   
        #target shape: (32, 1)
        #cond shape: (32, 17)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  #forward diffusion process, compute x_t from x_0 for each sample in batch based on t (for that sample)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t).view(-1)   #scalars, so this value fn predicts rewards/values from noisy trajectories. Each traj shape:  (4, 23), it has info of both (a, s)

        loss = nn.MSELoss()(pred, target)
        return loss

    def forward(self, x, cond, t):
        return self.model(x, cond, t)

