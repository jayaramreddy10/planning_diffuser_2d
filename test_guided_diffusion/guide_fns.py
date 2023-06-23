import numpy as np
import cv2
import torch
import torch.nn.functional as F
import einops
import matplotlib.pyplot as plt
from diffusion_fns import *

def add_borders(env):
    
    return np.pad(env, pad_width = ((1, 1), (1, 1)), mode = 'constant', constant_values = ((1, 1), (1, 1))).copy()

def get_cost_map(contours, env, border_value = 0.8):

    cost_map = np.zeros(env.shape, dtype = np.float32)

    for i in range(env.shape[0]):
        for j in range(env.shape[1]):

            min_obs_dist = env.size

            for c in range(len(contours)):

                obs_dist = np.abs(cv2.pointPolygonTest(contours[c],(j,i),True))
                if obs_dist < min_obs_dist:
                    min_obs_dist = obs_dist
                    
            if env[i, j]:
                cost = (1 - np.exp(-min_obs_dist/100)) * (1 - border_value) + border_value
            else:
                cost = (np.exp(-min_obs_dist/100)) * border_value

            cost_map[i, j] = cost #* 255

    return cost_map.copy()

def get_gradient_map(env, border_value = 0.8):

    bordered_env = add_borders(env)
    env_image = bordered_env * 255

    contours, _ = cv2.findContours(image=env_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    cost_map = get_cost_map(contours, env, border_value)
    
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

def guide_point_to_pixel(point, shape):

    limits = np.array(shape).reshape([1, 2, 1])

    # point  => (1, 2, N)
    # limits => (1, 2, 1)
    # pixel  => (1, 2, N)
    
    # For a -1 to 1 space
    pixel_coord = (point + 1) * (limits / 2)

    pixel = np.ceil(pixel_coord).astype(int) - 1

    return pixel.copy()

def guide_pixel_to_point(pixel, shape):

    limits = np.array(shape).reshape([1, 2, 1])

    # point  => (1, 2, N)
    # limits => (1, 2, 1)
    # pixel  => (1, 2, N)
    
    # For a -1 to 1 space
    point = 2 * (pixel) / limits - 1

    return point.copy()

def get_gradient(gradient_map, point):

    pixel = guide_point_to_pixel(point, gradient_map.shape[:-1])
    # pixel = guide_point_to_pixel(point)
    pixel = np.clip(pixel, 0, gradient_map.shape[0] - 1)
    # pixel  => (1, 2, N)
    # Check axis correspondences carefully here:
    gradient = gradient_map[pixel[:, 0, :], pixel[:, 1, :]]
    gradient = einops.rearrange(gradient, 'b n c -> b c n')
    return gradient.copy()

def length_gradient(X):

    grad = np.zeros(X.shape)

    for i in range(1, X.shape[2] - 1):

        grad[:, :, i] = 2 * (2 * X[:, :, i] - (X[:, :, i-1] + X[:, :, i+1]))

    return grad

def plot_environment(env):

    env_image = np.array(env * 255, dtype = np.uint8)
    env_image = np.rot90(env_image)

    plt.imshow(env_image, cmap = 'gray')
    



    