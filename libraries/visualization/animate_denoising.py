import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
# from diffusion_functions import *
import matplotlib.patches as patches
import os

# from helper_functions.diffusion_functions import *

# Create a figure and axis to plot on
fig, ax = plt.subplots()

diffusion_trajs = np.load("reverse_trajectory.npy")
T = diffusion_trajs.shape[0]

# beta = schedule_variance(T)
# alpha = 1 - beta
# alpha_bar = np.array(list(map(lambda t:np.prod(alpha[:t]), np.arange(T+1)[1:])))

# Define the animation function
def animate(i):

    ax.clear()

    # for j in range(diffusion_trajs.shape[-2]):
    #     rectangle = patches.Rectangle(x_min[j, :], x_max[j, 0] - x_min[j, 0], x_max[j, 1] - x_min[j, 1], facecolor = 'gray', alpha = 0.1)
    #     ax.add_patch(rectangle)
    

    # if i == 0:
    #     mean = np.zeros((diffusion_trajs.shape[-2], diffusion_trajs.shape[-1]))
    #     var = 1
    # else:
    #     mean = diffusion_trajs[T-i, :, :] * (1/np.sqrt(alpha[T-1-i]))
    #     var = np.sqrt(1 - alpha_bar[T-1-i])

    # total_pdf = np.zeros((1024, 1024))
    # for j in range(mean.shape[0]):

    #     pdf = get_multivariate_probability([0, 0], var, size = 512)
    #     mean_pixel = np.floor(point_to_pixel(mean[j], limit = 1024, bound = 2)).astype('int')

    #     #print(mean, mean_pixel)

    #     start_x = np.max([mean_pixel[0] - 256, 0])
    #     start_y = np.max([mean_pixel[1] - 256, 0])
    #     end_x = np.min([mean_pixel[0] + 256, 1023])
    #     end_y = np.min([mean_pixel[1] + 256, 1023])

    #     pdf_start_x = 256 - (mean_pixel[0] - start_x)
    #     pdf_end_x = 256 + (end_x - mean_pixel[0])
    #     pdf_start_y = 256 - (mean_pixel[1] - start_y)
    #     pdf_end_y = 256 + (end_y - mean_pixel[1])

    #     total_pdf[start_x : end_x, start_y : end_y] += pdf[pdf_start_x : pdf_end_x, pdf_start_y : pdf_end_y]

    # total_pdf = total_pdf / np.sum(total_pdf)

    # ax.imshow(total_pdf, cmap = 'gray')

    ax.scatter(diffusion_trajs[T-1-i, 1:-1, 0], diffusion_trajs[T-1-i, 1:-1, 1])
    ax.scatter(diffusion_trajs[T-1-i, 0, 0], diffusion_trajs[T-1-i, 0, 1], color = 'black')
    ax.scatter(diffusion_trajs[T-1  -i, -1, 0], diffusion_trajs[T-1-i, -1, 1], color = 'red')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    plt.title("Time step = " + str(T-1-i))

    os.system("cls")
    print("Animated ", i, " time steps")


# def render_video(denoised_trajs_stack):
    # Create the animation object, with a 50ms delay between frames
ani = FuncAnimation(fig, animate, frames = T, interval=1000, repeat=False)

# Set up the writer to save the animation as an MP4 video
writer = FFMpegWriter(fps=60)

# Save the animation as an MP4 video file
ani.save("reverse_diffusion2.mp4", writer=writer)

# Display the animation
# plt.show()

