import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

from helper_functions.diffusion_functions import *

traj_len = 100
T = 1000

trajectory = generate_trajectories(1, traj_len)[0]

# plt.plot(trajectory[0, :, 0], trajectory[0, :, 1])
# plt.scatter(trajectory[0, :, 0], trajectory[0, :, 1])
# plt.xlim([-1.1, 1.1])
# plt.ylim([-1.1, 1.1])
# plt.show()

beta = schedule_variance(T)
diffusion_trajs, _ = forward_diffuse(trajectory, T, beta)

# Create a figure and axis to plot on
fig, ax = plt.subplots()

# Define the animation function
def animate(i):

    ax.clear()
    ax.scatter(diffusion_trajs[i, :, 0], diffusion_trajs[i, :, 1])

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    plt.title("Time step = " + str(i))

    os.system("cls")
    print("Animated ", i, " time steps")

# Create the animation object, with a 50ms delay between frames
ani = FuncAnimation(fig, animate, frames = T, interval=1000, repeat=False)

# Set up the writer to save the animation as an MP4 video
writer = FFMpegWriter(fps=60)

# Save the animation as an MP4 video file
ani.save("forward_diffusion_trajectory_straight.mp4", writer=writer)