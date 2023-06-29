from mazelib import Maze 
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import heapq as heap
from collections import defaultdict
import numpy as np
import time
import os

# from helper_functions.diffusion_functions import *

class Maze:
    
    def generate_maze(self, maze_width, maze_height):

        if maze_height % 2 == 0 or maze_width % 2 == 0:
            raise Exception("Only Odd-numbered widths are allowed")
        
        maze = Maze()

        maze.generator = Prims((maze_height+1)//2 , (maze_width+1)//2)
        maze.generate()

        return maze.grid[1:-1, 1:-1]

    def convert_maze_to_image(self, generated_maze, image_width, image_height):

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

    def generate_random_start_goal(self, maze):

        free_indices = np.array(np.where(1 - maze)).T

        start_index, goal_index = np.random.choice(np.arange(free_indices.shape[0]), 2, replace = False)
        start_pixel, goal_pixel = free_indices[start_index], free_indices[goal_index]

        start = np.array(start_pixel, dtype = np.int32)
        goal = np.array(goal_pixel, dtype = np.int32)

        return start, goal
    
    def load_random_mazes_and_trajectories(self, num_samples, env_path, traj_path):

        _, _, env_files = next(os.walk(env_path))
        num_envs = len(env_files)
        env_indices = np.random.choice(range(1, num_envs + 1), size = num_samples) #, replace = False)

        _, _, traj_files = next(os.walk(traj_path))
        num_trajs = len(traj_files)
        traj_indices = np.random.choice(range(1, num_trajs + 1), size = num_samples) #, replace = False)

        return env_indices, traj_indices

class A_star:

    def __init__(self, maze, max_iter = 10000):
        
        self.max_iter = max_iter

        self.maze = maze.copy()
        self.q_min = np.array([0, 0])
        self.q_max = np.array(maze.shape) - 1
    
    def plan_path(self, start, goal):

        output = self.run_a_star(start, goal)

        start = tuple(start)

        if type(output) == type(None):
            print("A-star did not find a path")
            return None
        
        else:
            
            end_node, parentDic, time_taken, num_iterations = output
            end_node = tuple(end_node)
            
            path = [end_node]

            while(end_node != start):
                
                end_node = tuple(parentDic[end_node])
                path.append(end_node)
            

            path.reverse()

            return np.array(path), time_taken, num_iterations

    def run_a_star(self, start, goal):

        start_time = time.time()
        
        # Get data on current node (now the start node):
        start, goal = tuple(start), tuple(goal)
        start_node = (self.distance_between(start, goal), start)

        # Initialize:
        iter = 0
        nodes = []
        heap.heapify(nodes)

        # Dictionary indicating actual costs of each node:
        actualCostMap = defaultdict(lambda: float('inf'))
        actualCostMap[start] = 0

        # Push the current node (start node) into heap
        heap.heappush(nodes, start_node)

        # Nodes that have to be explored:
        open_nodes = set()
        open_nodes.add(start)

        parentDict = {}
        closed_nodes = set()

        while iter < self.max_iter and len(nodes) > 0:

            iter += 1

            # Choose the node at the top of the heap to explore and remove it from open nodes:
            heurCost, curr_node = heap.heappop(nodes)
            open_nodes.remove(curr_node)

            # Check if the current node is the goal, if it is return it with its dictionary of parents:
            if self.is_goal(curr_node, goal):
                time_taken = time.time() - start_time
                #print("A-star took " + str(np.round(time_taken, 2)) + " seconds to converge")
                return curr_node, parentDict, time_taken, iter    
            
            # Add the current node to the list of closed nodes:
            closed_nodes.add(curr_node)

            # Get neighbouring nodes:
            adj_nodes = self.get_adjacent_nodes(curr_node)

            # print("Current Node: ", curr_node)
            # print("Adjacent Nodes: ", adj_nodes)

            # print(nodes)

            # _ = input("Start Exploring?")

            # Loop through all neighbours:
            for i in range(adj_nodes.shape[0]):

                adj_node = tuple(adj_nodes[i])

                # Get the edge cost of the adjacent node using euclidean distance and add it to cost of current node
                newEdgeCost = actualCostMap[curr_node] + self.distance_between(curr_node, adj_node)
                # Get the heuristic cost, i.e. distance to goal
                heurCost = self.distance_between(adj_node, goal)
                # Get the previously stored cost of next point for comparison
                currCostofAdjPoint = actualCostMap[adj_node]

                # If we found a lower cost for an adjacent point, update the cost of that point:
                if currCostofAdjPoint > newEdgeCost:
                    
                    # Update parents dictionary, the cost map and get the total cost of this node
                    parentDict[adj_node] = curr_node
                    actualCostMap[adj_node] = newEdgeCost
                    totalCost = heurCost + newEdgeCost

                    # print("Pushed the node with total cost = ", totalCost)
                    # print("--------------------------------------------------------")

                    # If the node is not open for exploration, add it to open nodes and push it into the heap
                    if adj_node not in open_nodes:
                        open_nodes.add(adj_node)
                        heap.heappush(nodes, (totalCost, adj_node))

        time_taken = time.time() - start_time
        
        return curr_node, parentDict, time_taken, iter 
    
    def distance_between(self, node1, node2):
        
        return np.linalg.norm(np.array(node1) - np.array(node2))
    
    def is_goal(self, node, goal):

        if self.distance_between(node, goal) <= 0.1:
            return True
        else:
            return False
    
    def get_adjacent_nodes(self, node):

        adj_nodes = np.zeros((0, 2))
        
        theta = np.linspace(start = 0, stop = 2*np.pi, num = 4 + 1)[:-1]
        x_diff = np.cos(theta).astype('int')
        y_diff = np.sin(theta).astype('int')
        temp_adj_nodes = np.zeros((theta.shape[0], 2))
        temp_adj_nodes[:, 0] = (node[0] + x_diff)
        temp_adj_nodes[:, 1] = (node[1] + y_diff)
        temp_adj_nodes = temp_adj_nodes.astype('int')
        
        for i in range(temp_adj_nodes.shape[0]):

            if np.all(temp_adj_nodes[i] >= self.q_min) and np.all(temp_adj_nodes[i] <= self.q_max):
                if not self.maze[temp_adj_nodes[i, 0], temp_adj_nodes[i, 1]]:
                    adj_nodes = np.concatenate([adj_nodes, [temp_adj_nodes[i]]], axis = 0)

        return adj_nodes.copy()

        

