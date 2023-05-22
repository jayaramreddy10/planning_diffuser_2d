import mazelib
from mazelib import Maze 
from mazelib.generate.Prims import Prims

import numpy as np

def generate_maze(maze_width, maze_height):

    if maze_height % 2 == 0 or maze_width % 2 == 0:
        raise Exception("Only Odd-numbered widths are allowed")
    
    maze = Maze()

    maze.generator = Prims((maze_height+1)//2 , (maze_width+1)//2)
    maze.generate()

    return maze.grid[1:-1, 1:-1]

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