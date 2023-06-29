import numpy as np
import cv2
import torch
import torch.nn.functional as F
import einops

from .animation import Animation
from .maze import Maze
from .diffusion import Diffusion

class ClassicGuide(Animation):

    def __init__(self, env, border_value = 0.8):

        self.env = env.copy()

        self.bordered_env = self.add_borders(env)
        self.env_image = self.bordered_env * 255

        self.contours, _ = cv2.findContours(image = self.env_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        self.gradient_map, self.cost_map = self.get_cost_map(border_value)

        Animation.__init__()

    def add_borders(self):
        
        return np.pad(self.env, pad_width = ((1, 1), (1, 1)), mode = 'constant', constant_values = ((1, 1), (1, 1))).copy()

    def get_cost_map(self, border_value = 0.8):

        cost_map = np.zeros(self.env.shape, dtype = np.float32)

        for i in range(self.env.shape[0]):
            for j in range(self.env.shape[1]):

                min_obs_dist = self.env.size

                for c in range(len(self.contours)):

                    obs_dist = np.abs(cv2.pointPolygonTest(self.contours[c],(j,i),True))
                    if obs_dist < min_obs_dist:
                        min_obs_dist = obs_dist
                        
                if self.env[i, j]:
                    cost = (1 - np.exp(-min_obs_dist/100)) * (1 - border_value) + border_value
                else:
                    cost = (np.exp(-min_obs_dist/100)) * border_value

                cost_map[i, j] = cost

        return cost_map.copy()

    def get_gradient_map(self, env, border_value = 0.8):

        cost_map = self.get_cost_map(self.contours, env, border_value)
        
        scharr_1 = np.array([[3, 0, -3],
                        [10, 0, -10],
                        [3, 0, -3]])

        scharr_0 = scharr_1.T

        scharr_0 = torch.tensor([[scharr_0]], dtype = torch.float32)
        scharr_1 = torch.tensor([[scharr_1]], dtype = torch.float32)

        gradient = np.zeros((env.shape[0], env.shape[1], 2))

        cost_map_tensor = torch.tensor([[np.pad(cost_map, ((1, 1), (1, 1)), mode = 'edge')]], dtype = torch.float32)
        gradient[:, :, 1]  = F.conv2d(cost_map_tensor, scharr_1).numpy()[0][0]
        gradient[:, :, 0]  = F.conv2d(cost_map_tensor, scharr_0).numpy()[0][0]

        return gradient.copy(), cost_map.copy()

    def env_gradient(self, point):

        pixel = self.point_to_pixel(point, self.gradient_map.shape[:-1])
        pixel = np.clip(pixel, 0, self.gradient_map.shape[0] - 1)
        # pixel  => (1, 2, N)
        # Check axis correspondences carefully here:
        gradient = self.gradient_map[pixel[:, 0, :], pixel[:, 1, :]]
        gradient = einops.rearrange(gradient, 'b n c -> b c n')
        return gradient.copy()

    def length_gradient(self, X):

        grad = np.zeros(X.shape)

        for i in range(1, X.shape[2] - 1):

            grad[:, :, i] = 2 * (2 * X[:, :, i] - (X[:, :, i-1] + X[:, :, i+1]))

        return grad

class OverlapGuide(Diffusion, Maze, Animation):
    
    def __init__(self, T, variance_thresh = 0.02):

        Diffusion.__init__(T, variance_thresh)  # This class already depends on Gaussian
        Maze.__init__()
        Animation.__init__()
    
    def get_overlap_score(self, env, mean_point, var):
        """
        env => 2D array of shape (env_size x env_size)
        mean_point => 2D array of shape (channels x traj_len)
        """

        pdf = self.multivariate_pdf([0, 0], var)    # Inherited from Gaussian
        pdf_size = pdf.shape[0]

        env_size = env.shape[0]

        mean_array = self.point_to_pixel(mean_point, 1024, 1, return_int = False)        # Inherited from Animation

        score = 0

        for i in range(mean_array.shape[-1]):

            mean = mean_array[:, i]
            
            overlap_start_x = np.max([mean[0] - pdf_size//2, 0]) - mean[0] + pdf_size//2
            overlap_start_y = np.max([mean[1] - pdf_size//2, 0]) - mean[1] + pdf_size//2
            overlap_end_x = np.min([mean[0] + pdf_size//2, env_size]) - mean[0] + pdf_size//2
            overlap_end_y = np.min([mean[1] + pdf_size//2, env_size]) - mean[1] + pdf_size//2

            if overlap_start_x > 1023 or overlap_start_y > 1023 or overlap_end_x < 0 or overlap_end_y < 0:
                overlap_start_x, overlap_start_y, overlap_end_x, overlap_end_y = 0, 0, 0, 0

            env_start_x = np.max([overlap_start_x - pdf_size//2 + mean[0], 0])
            env_start_y = np.max([overlap_start_y - pdf_size//2 + mean[1], 0])
            env_end_x = np.min([overlap_end_x - pdf_size//2 + mean[0], env_size])
            env_end_y = np.min([overlap_end_y - pdf_size//2 + mean[1], env_size])

            if env_start_x > 1023 or env_start_y > 1023 or env_end_x < 0 or env_end_y < 0:
                env_start_x, env_start_y, env_end_x, env_end_y = 0, 0, 0, 0

            overlap = np.ones((pdf.shape[0], pdf.shape[1]))
            overlap[overlap_start_x : overlap_end_x, overlap_start_y : overlap_end_y] = env[env_start_x : env_end_x, env_start_y : env_end_y]

            score += np.sum(np.multiply(overlap, pdf)) / np.sum(pdf)

        return score / mean_array.shape[-1]
    
    def generate_training_sample(self, num_samples, env_path, traj_path, env_size = 1024, return_type = "tensor"):

        env_indices, traj_indices = self.load_random_mazes_and_trajectories(num_samples, env_path, traj_path)
        
        dummy_traj = np.load(traj_path + "trajectory" + str(traj_indices[0]) + ".npy")

        c = dummy_traj.shape[-2]
        n = dummy_traj.shape[-1]

        # Inputs:
        environments = np.zeros((num_samples, 1, env_size, env_size))
        X = np.zeros((num_samples, c, n))       # trajectories
        time_steps = np.random.randint(1, self.T + 1, size = (num_samples, ))

        # Outputs:
        costs = np.zeros((num_samples, 1))

        for i in range(num_samples):

            maze = np.load(env_path + "maze" + str(env_indices[i]) + ".npy")
            if maze.shape[0] != env_size:
                env = self.convert_maze_to_image(maze, env_size, env_size)
            else:
                env = maze.copy()

            trajectory = np.load(traj_path + "trajectory" + str(traj_indices[i]) + ".npy")

            xt, mean, var = self.q_sample_from_x0(np.array([trajectory[0]]), time_steps[i])

            costs[i, 0] = self.get_overlap_score(env, mean[0], var)  # Batch size is 1 so thats why we have [0] next to mean

            # Set inputs:
            environments[i, 0] = env
            X[i] = xt[0]  # Batch size is 1 so thats why we have [0] next to trajectory

        if return_type == "tensor":
            X = torch.tensor(X, dtype = torch.float32)
            environments = torch.tensor(environments, dtype = torch.float32)
            time_steps = torch.tensor(time_steps, dtype = torch.float32)
            Y = torch.tensor(costs, dtype = torch.float32)

        return X, environments, time_steps, Y

class DiscountedGuide(Diffusion, Maze, Animation):
    
    def __init__(self, T, variance_thresh = 0.02):

        Diffusion.__init__(T, variance_thresh)  # This class already depends on Gaussian
        Maze.__init__()
        Animation.__init__()
    
    def get_true_cost(self, x0, env, t, gamma = 0.9):

        pixel_traj = self.point_to_pixel(x0, env.shape[-1])
        obs_overlap = env[pixel_traj[0, :], pixel_traj[1, :]]
        overlap_cost = (np.sum(obs_overlap, axis = -1) / pixel_traj.shape[-1]).reshape((-1, 1))

        cost = (gamma ** t) * overlap_cost
        #cost = overlap_cost

        return cost.squeeze()

    def generate_training_sample(self, num_samples, env_path, traj_path, env_size = 1024, return_type = "tensor"):
        
        env_indices, traj_indices = self.load_random_mazes_and_trajectories(num_samples, env_path, traj_path)
        
        dummy_traj = np.load(traj_path + "trajectory" + str(traj_indices[0]) + ".npy")

        c = dummy_traj.shape[-2]
        n = dummy_traj.shape[-1]

        # Inputs:
        environments = np.zeros((num_samples, 1, env_size, env_size))
        x0 = np.zeros((num_samples, c, n))       # trajectories

        # Outputs:
        costs = np.zeros((num_samples, 1))

        for i in range(num_samples):

            maze = np.load(env_path + "maze" + str(env_indices[i]) + ".npy")
            env = self.convert_maze_to_image(maze, env_size, env_size)

            trajectory = np.load(traj_path + "trajectory" + str(traj_indices[i]) + ".npy")

            # Get Cost Value and set outputs:
            costs[i, 0] = self.get_true_cost(trajectory[0], maze, time_step[i])

            # Set inputs:
            environments[i, 0] = env
            x0[i] = trajectory[0]

        X, time_step, _, _, _ = self.generate_q_sample(x0, return_type = return_type)

        if return_type == "tensor":
            Y = torch.tensor(costs, dtype = torch.float32)
            environments = torch.tensor(environments, dtype = torch.float32)
        elif return_type == "numpy":
            Y = costs.copy()

        return X, environments, time_step, Y

    



    