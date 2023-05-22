import numpy as np
import matplotlib.pyplot as plt
from d4rl_maze2d_dataset.d4rl import load_environment
from d4rl_maze2d_dataset.maze_functions import convert_maze_to_image
from d4rl_maze2d_dataset.sequence import RRT_star_traj_binary
from d4rl_maze2d_dataset.upsample import *
from scipy.interpolate import interp1d
from d4rl_maze2d_dataset.interparc import *

def generate_equidistant_points_scipy(points, num_points):
    num_given_points = len(points)
    t = np.linspace(0, 1, num_given_points)
    t_new = np.linspace(0, 1, num_points)
    interpolator = interp1d(t, points, kind='linear', axis=0)
    equidistant_points = interpolator(t_new)
    return equidistant_points

# def generate_equidistant_points_scipy2(points, num_points,x, y):
#     x = list(points[:, 0])
#     y = list(points[:, 1])

#     # Linear length on the line
#     distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
#     distance = distance/distance[-1]

#     fx, fy = interp1d( distance, x ), interp1d( distance, y )

#     alpha = np.linspace(0, 1, 15)
#     x_regular, y_regular = fx(alpha), fy(alpha)

#     samples = 
#     # plt.plot(x, y, 'o-')
#     # plt.plot(x_regular, y_regular, 'or')
#     # plt.axis('equal')

def get_random_pixel_coordinates(image):
    # Get the coordinates of all pixels with a value of 1
    coordinates = np.argwhere(image == 255)
    # print(coordinates[0])
    
    if len(coordinates) > 0:
        # Randomly select one of the coordinates
        random_index = np.random.randint(0, len(coordinates))
        random_coordinates = coordinates[random_index]

        # Extract x and y coordinates
        x, y = random_coordinates

        return x, y

    else:
        return None
    
data = np.load('maze1.npy')
# h ,w = data.shape[0], data.shape[1]
w , h = 1024, 1024
num_waypoints = 384
img = convert_maze_to_image(data, w, h)
# print(img)
n_trajs = 1
trajs = []
for i in range(n_trajs):
    # get random initial and final points
    q_init = get_random_pixel_coordinates(img)
    q_final = get_random_pixel_coordinates(img)
    # print(q_init)
    # print(q_final)

    RRT_Star = RRT_star_traj_binary(img, q_init, q_final)

    # Main loop of RRT
    q_new = RRT_Star.q_init
    while True:
        # Check if the new vertex can connect with the final point
        # If yes, finish the loop
        if RRT_Star.connection_check(q_new) == True:
            break

        # Generate random vertex
        q_rand = RRT_Star.random_vertex_generate()

        # Check if the random vertex is in collision with the geometry
        if RRT_Star.collision_check_point(q_rand) == True:
            continue

        # Search for the nearest point in map
        index_near, q_near = RRT_Star.nearest_vertex_check(q_rand)

        # Check if the line between q_near and q_rand collides with the geometry
        if RRT_Star.collision_check_line(q_near, q_rand) == True:
            continue

        # Generate new vertex according to delta_q
        q_new = RRT_Star.new_point_generate(q_near, q_rand, index_near)

    RRT_Star.figure_generate()
    traj = RRT_Star.return_traj()
    samples = interparc_fn(traj[:, 0], traj[:, 1], num_waypoints)

    print('traj: {}'.format(traj))

    for (x,y) in samples:
        plt.plot(x,y,'r',marker='.')

    print(samples.shape)
    cond = {0 : np.array(q_init), num_waypoints - 1: np.array(q_final)}
    
    # traj, cond, val = RRT_Star.generate_traj()    #higher the val, higher the reward (so if val is high, path len is less which is optimal)
    trajs.append((samples, cond))

plt.imshow(img)
plt.show()
print(data)
