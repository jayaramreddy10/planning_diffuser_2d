import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
# from helper_functions.diffusion_functions import *
# from helper_functions.guide_functions import *

# -------------- Edit these parameters -------------- #

env_name = "maze10"

# --------------------------------------------------- #

# Create a figure and axis to plot on
fig, ax = plt.subplots()

diffusion_trajs = np.load("reverse_guided_trajectory_" + env_name + ".npy")
# gradient_points = -10 * np.load("gradients_" + env_name + ".npy") + diffusion_trajs

env_path = "environments/2D_maze/15x15/" + env_name + ".npy"
env = np.load(env_path)

pixel_traj = point_to_pixel(diffusion_trajs, env.shape[0], return_int = False)
pixel_traj[:, 1, :] = env.shape[1] - 1 - pixel_traj[:, 1, :]
# pixel_grad = point_to_pixel(gradient_points, env.shape[0], return_int = False)
# pixel_grad[:, 1, :] = env.shape[1] - 1 - pixel_grad[:, 1, :]

T = diffusion_trajs.shape[0]

# beta = schedule_variance(T)
# alpha = 1 - beta
# alpha_bar = np.array(list(map(lambda t:np.prod(alpha[:t]), np.arange(T+1)[1:])))

# Define the animation function
def animate(i):
    # Update the y data with a new sine wave shifted in the x direction
    ax.clear()

    # if i == 0:
    #     mean = np.zeros((diffusion_trajs.shape[-2], diffusion_trajs.shape[-1]))
    #     var = 1
    # else:
    #     mean = diffusion_trajs[T-i, :, :] * (1/np.sqrt(alpha[T-1-i]))
    #     var = np.sqrt(1 - alpha_bar[T-1-i])

    # total_pdf = np.zeros((1024, 1024))

    # pdf = get_multivariate_probability([0, 0], var, size = 1024)

    # for j in range(mean.shape[-1]):

    #     mean_pixel = np.floor(point_to_pixel(mean[:, j], limit = 1024, bound = 1)).astype('int')

    #     #print(j, mean_pixel)

    #     start_x = np.max([mean_pixel[0] - 512, 0])
    #     start_y = np.max([mean_pixel[1] - 512, 0])
    #     end_x = np.min([mean_pixel[0] + 512, 1023])
    #     end_y = np.min([mean_pixel[1] + 512, 1023])

    #     if start_x > 1023 or start_y > 1023 or end_x < 0 or end_y < 0:
    #         continue

    #     pdf_start_x = 512 - (mean_pixel[0] - start_x)
    #     pdf_end_x = 512 + (end_x - mean_pixel[0])
    #     pdf_start_y = 512 - (mean_pixel[1] - start_y)
    #     pdf_end_y = 512 + (end_y - mean_pixel[1])

    #     total_pdf[start_x : end_x, start_y : end_y] += pdf[pdf_start_x : pdf_end_x, pdf_start_y : pdf_end_y]

    # total_pdf = total_pdf / np.max(total_pdf)

    # display_env = env + total_pdf
    
    ax.imshow(np.rot90(255 - env), cmap = 'gray')

    ax.scatter(pixel_traj[T-1-i, 0, 1:-1], pixel_traj[T-1-i, 1, 1:-1])
    ax.scatter(pixel_traj[T-1-i, 0, 0], pixel_traj[T-1-i, 1, 0], color = 'green')
    ax.scatter(pixel_traj[T-1-i, 0, -1], pixel_traj[T-1-i, 1, -1], color = 'red')

    # for j in range(1, pixel_grad.shape[-1] - 1):

    #     x = [pixel_traj[T-1-i, 0, j], pixel_grad[T-1-i, 0, j]]
    #     y = [pixel_traj[T-1-i, 1, j], pixel_grad[T-1-i, 1, j]]
    #     ax.plot(x, y, linewidth = 2, color = 'red')

    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])

    plt.title("Time step = " + str(T-1-i))

    os.system("cls")
    print("Animated ", i, " time steps")

# Create the animation object, with a 50ms delay between frames
ani = FuncAnimation(fig, animate, frames = T, interval=1000, repeat=False)

# Set up the writer to save the animation as an MP4 video
writer = FFMpegWriter(fps=60)

# Save the animation as an MP4 video file
ani.save("reverse_guided_diffusion_" + env_name + ".mp4", writer=writer)

# Display the animation
#plt.show()

