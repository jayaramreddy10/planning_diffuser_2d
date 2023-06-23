import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import einops
import pickle
import glob
from collections import namedtuple

DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def load_diffusion(*loadpath, epoch='latest', device='cuda:0'):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    # render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config()
    # renderer = render_config()
    renderer = None
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)

def cosine_func(t, T):
    
    s = 1e-10
    
    return np.cos((t/T + s) * (np.pi/2) / (1 + s)) ** 0.15

def schedule_variance(T, thresh = 0.02):
    
    return np.linspace(0, thresh, T+1)[1:]

def plot_trajectories(x_in):

    x = np.array(x_in)
    
    for i in range(x.shape[0]):
        plt.scatter(x[i, 0, 1:-1], x[i, 1, 1:-1])
        plt.scatter(x[i, 0, 0], x[i, 1, 0], color = 'black')
        plt.scatter(x[i, 0, -1], x[i, 1, -1], color = 'red')

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

def pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * (((x - mu) / sigma) ** 2))

def KL_divergence(distr1, distr2):

    return np.mean(distr1 * np.log(distr1 / distr2))

def KL_divergence_against_gaussian(sample):

    mu = np.mean(sample)
    sigma = np.var(sample)

    x = np.linspace(-10, 10, int(1e+6))

    distr1 = pdf(x, 0, 1) 
    distr2 = pdf(x, mu, sigma)

    return KL_divergence(distr1, distr2)

def forward_diffuse(trajectory, T, beta):

    # Remember that the trajectory that is being input here is a singular starting trajectory of shape (n+2, 2)
    
    # Gather information from trajectory:
    n = trajectory.shape[0] - 2

    alpha = 1 - beta
    
    # Initialize the diffusion trajectories across time steps:
    diffusion_trajs = np.zeros((T+1, n+2, 2))
    diffusion_trajs[0] = trajectory.copy()

    # Calculate epsilons to forward diffuse current trajectory:
    mean = np.zeros(n*2)
    cov = np.eye(n*2)
    eps = np.random.multivariate_normal(mean, cov, size = (T,)).reshape(T, n, 2)
    kl_divs = np.zeros((T, ))

    for i in range(1, T+1):

        diffusion_trajs[i, 1:-1] = np.sqrt(alpha[i-1]) * diffusion_trajs[i-1, 1:-1] + np.sqrt(1 - alpha[i-1]) * eps[i-1]
        diffusion_trajs[i, 0] = trajectory[0].copy()
        diffusion_trajs[i, -1] = trajectory[-1].copy()
        kl_divs[i-1] = KL_divergence_against_gaussian(diffusion_trajs[i].flatten())

    return diffusion_trajs.copy(), eps.copy(), kl_divs.copy()

def reverse_diffuse(xT, eps, T):

    #xT = np.random.multivariate_normal(mean = np.zeros((traj_len*2,)), cov = np.eye(traj_len*2)).reshape(traj_len, 2)

    diffusion_trajs = np.zeros((T+1, xT.shape[0], 2))
    diffusion_trajs[T] = xT.copy()

    beta = schedule_variance(T)
    alpha = 1 - beta

    for t in range(T, 0, -1):

        diffusion_trajs[t-1] = (diffusion_trajs[t] - np.sqrt(1 - alpha[t-1]) * eps[t-1]) / np.sqrt(alpha[t-1])

    return diffusion_trajs.copy()

def train_denoiser(model, num_samples, traj_len, T, epochs_per_set = 10, num_sets = 10):

    losses = np.zeros((num_sets, epochs_per_set))

    for s in range(num_sets):

        X = np.zeros((num_samples*model.T, 2*(traj_len+2)))
        Y_true = np.zeros((num_samples*model.T, 2*(traj_len)))

        trajectories = generate_trajectories(num_samples, traj_len)

        for i in range(num_samples):

            print("Generating Sample ", i+1)

            # Diffuse just one sample trajectory:
            diff_traj, epsilon = forward_diffuse(trajectories[i], model.T, model.beta)

            clear_output(wait=True)

            X[i*T : (i+1)*T] = diff_traj[1:].reshape(T, 2*(traj_len+2))
            Y_true[i*T : (i+1)*T] = epsilon.reshape(T, 2*(traj_len))

        X = torch.tensor(X, dtype = torch.float32)
        Y_true = torch.tensor(Y_true, dtype = torch.float32)

        for e in range(epochs_per_set):
                
            model.train(True)

            Y_pred = model(X)

            model.optimizer.zero_grad()

            loss = model.loss_fn(Y_pred, Y_true)

            loss.backward()

            model.optimizer.step()

            loss_value = torch.norm(Y_pred - Y_true).item()
            losses[s, e] = loss_value/epochs_per_set

            print("Current epoch loss = ", loss_value)

            clear_output(wait=True)

        model.save()
        np.save("Models/" + model.model_name + "/losses.npy", losses)

    return losses

def generate_training_sample(num_samples, generate_trajectories, traj_len = 1000, T = 500, bounds = np.array([[-1, -1], [1, 1]]),
                             return_type = "tensor"):

    # Refer to the Training Algorithm in Ho et al 2020 for the psuedo code of this function
        
    x0 = generate_trajectories(num_samples = num_samples, traj_len = traj_len, bounds = bounds).numpy()
    time_step = np.random.randint(1, T+1, size = (num_samples, ))

    # Remember, for each channel, we get a multivariate normal distribution.
    mean = np.zeros(traj_len)
    cov = np.eye(traj_len)
    eps = np.random.multivariate_normal(mean, cov, (num_samples, 2))

    beta = schedule_variance(T)
    alpha = 1 - beta
    alpha_bar = np.reshape(np.array(list(map(lambda t:np.prod(alpha[:t]), time_step))), (-1, 1, 1))  # Tested: This works
    
    # Size chart:
    # x0         => (num_samples, 2, traj_len)
    # xt         => (num_samples, 2, traj_len)
    # alpha_bar  => (num_samples, 1, 1)
    # eps        => (num_samples, 2, traj_len)
    
    xt = (np.sqrt(alpha_bar) * x0) + (np.sqrt(1 - alpha_bar) * eps)

    # CONDITIONING:
    xt[:, :, 0] = x0[:, :, 0].copy()
    xt[:, :, -1] = x0[:, :, -1].copy()

    if return_type == "tensor":
        X = torch.tensor(xt, dtype = torch.float32)
        Y = torch.tensor(eps, dtype = torch.float32)
        time_step = torch.tensor(time_step)
    elif return_type == "numpy":
        X = xt.copy()
        Y = eps.copy()

    return X, Y, time_step

def q_sample_means(x0, t, beta):

    alpha = 1 - beta
    alpha_bar = np.prod(alpha[:t])

    eps = np.random.multivariate_normal(mean = np.zeros((x0.shape[-1], )), 
                                        cov = np.eye(x0.shape[-1]),
                                        size = (x0.shape[0], ))
    
    xt = (np.sqrt(alpha_bar) * x0) + (np.sqrt(1 - alpha_bar) * eps)
    mean = (np.sqrt(alpha_bar) * x0)
    var = np.sqrt(1 - alpha_bar)

    return xt, mean, var

def get_limits(mu, sigma):

    p_min = 0.01 * pdf(mu, mu, sigma)
    x_1 = -1 * sigma * np.sqrt(2 * np.log(1 / (p_min * sigma * np.sqrt(2*np.pi)))) + mu
    x_2 = sigma * np.sqrt(2 * np.log(1 / (p_min * sigma * np.sqrt(2*np.pi)))) + mu

    return x_1, x_2

def get_multivariate_probability(mean, var, size = 1024, limits = (-1, 1)):
    """
    mean => 1D array of size k, in a 2D trajectory it is k=2 and for 3D k=3
    covariance => 2D array of (k x k)
    """

    mean = np.array(mean)
    
    k = mean.size

    axis_div = np.repeat([np.linspace(limits[0], limits[1], size)], repeats = k, axis = 0)
    x = np.array(np.meshgrid(*axis_div))

    mu = np.reshape(mean, (k, *np.ones(k, dtype = np.int32)))

    pdf = (1 / (((2 * np.pi) ** (k/2)) * np.sqrt(var**k))) * np.exp(-0.5 * np.sum((x - mu)**2, axis = 0) / var)

    return pdf / np.sum(pdf)

def point_to_pixel(x, limit = 1024, bound = 1):
    """
    Image should always be square with equal and opposite bounds
    """

    return (limit / (2 * bound)) * (x + bound)

def pixel_to_point(p, limit = 1024, bound = 1):
    """
    Image should always be square with equal and opposite bounds
    """

    return (2 * bound / limit) * p - bound