import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from diffusion_fns import *
from guide_fns import *

# -------------- Edit these parameters -------------- #

# env_name = "two_pillars"

# --------------------------------------------------- #

def write_video(frames, output_path, fps=30.0, codec='mp4v'):#, bitrate=4000000 
    # Get the dimensions of the first frame
    height, width, _ = frames[0].shape

    # Create a VideoWriter object with codec and bitrate options
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    # Write each frame to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()

def convert_maze_to_image(generated_maze, image_width, image_height):

    maze_width = generated_maze.shape[1]
    maze_height = generated_maze.shape[0]
    
    cell_width = image_width // maze_width
    cell_height = image_height // maze_height

    pad_width_left = (image_width % maze_width) // 2
    pad_width_right = pad_width_left + ((image_width % maze_width) % 2)
    pad_length_up = (image_height % maze_height) // 2
    pad_length_down = pad_length_up + ((image_height % maze_height) % 2)

    maze_image = np.zeros((image_height - (pad_length_up + pad_length_down), image_width - (pad_width_left + pad_width_right)), dtype = np.uint8)

    for row in range(maze_height):
        for col in range(maze_width):

            if generated_maze[row, col] == 1:
                maze_image[row * cell_height : (row+1) * cell_height, col * cell_width : (col+1) * cell_width] = 255
            elif generated_maze[row, col] == 0:
                maze_image[row * cell_height : (row+1) * cell_height, col * cell_width : (col+1) * cell_width] = 0

    maze_image = np.pad(maze_image, ((pad_length_up, pad_length_down), (pad_width_left, pad_width_right)), 'edge')

    return maze_image.copy()

# Create a figure and axis to plot on
fig, ax = plt.subplots()

diffusion_trajs = np.load("reverse_guided_trajectory" + ".npy")
print(diffusion_trajs.shape)

#load env
file_num = 500
envfile = "/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/generate_traj_maze2d/5x5/maze" + str(file_num) + ".npy"
data = np.load(envfile)   #(5, 5)
w , h = 1024, 1024
num_waypoints = 256
env = convert_maze_to_image(data, w, h)   #(1024, 1024)
print('env shape: {}'.format(env.shape))
# x = input('sd')

pixel_traj = guide_point_to_pixel(diffusion_trajs, env.shape)
pixel_traj[:, 1, :] = env.shape[1] - 1 - pixel_traj[:, 1, :]

T = diffusion_trajs.shape[0]
# T = 5
# Define the animation function
    # Update the y data with a new sine wave shifted in the x direction
imgs = [None for i in range(T)]
for i in range(T - 1, -1 ,-1):
            # Update the y data with a new sine wave shifted in the x direction
    ax.clear()

    ax.imshow(np.rot90(env), cmap = 'gray')

    ax.scatter(pixel_traj[i, 0, 1:-1], pixel_traj[i, 1, 1:-1])
    ax.scatter(pixel_traj[i, 0, 0], pixel_traj[i, 1, 0], color = 'green')
    ax.scatter(pixel_traj[i, 0, -1], pixel_traj[i, 1, -1], color = 'red')

    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])

    plt.title("Time step = " + str(T-1-i))

    os.system("clear")
    print("Animated ", i, " time steps")

    temp_image_path = "tmp.png"
    plt.savefig(temp_image_path)

    imgs[i] = cv2.imread(temp_image_path)

print(type(imgs[i]))
output_path = './diffusion_animate.mp4'
write_video(imgs, output_path)
# # Create the animation object, with a 50ms delay between frames
# ani = FuncAnimation(fig, animate, frames = T, interval=1000, repeat=False)

# # Set up the writer to save the animation as an MP4 video
# writer = FFMpegWriter(fps=60.0, bitrate=4000000)

# # Save the animation as an MP4 video file
# ani.save("reverse_guided_diffusion_"  + ".mp4", writer=writer)

# Display the animation
#plt.show()

