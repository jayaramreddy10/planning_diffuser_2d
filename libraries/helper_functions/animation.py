import numpy as np
import matplotlib.pyplot as plt

class Animation:

    def __init__(self):
        pass
    
    def point_to_pixel(self, x, limit = 1024, bound = 1, return_int = True):
        """
        Image should always be square with equal and opposite bounds
        """

        if return_int == True:
            return ((limit / (2 * bound)) * (x + bound)).astype('int')
        else:
            return ((limit / (2 * bound)) * (x + bound))

    def pixel_to_point(self, p, limit = 1024, bound = 1):
        """
        Image should always be square with equal and opposite bounds
        """

        return (2 * bound / limit) * p - bound
    
    def plot_trajectories(self, x, ax = None, condition = True):
        """
        Plots 2D trajectories of any length in the range [-1, 1]
        Inputs:
        x -> Numpy array of shape (num_trajs, 2, traj_len)
        Outputs:
        None
        """
        
        if type(ax) == type(None):
            ax = plt

        for i in range(x.shape[0]):
            if condition:
                ax.scatter(x[i, 0, 1:-1], x[i, 1, 1:-1])
                ax.scatter(x[i, 0, 0], x[i, 1, 0], color = 'black')
                ax.scatter(x[i, 0, -1], x[i, 1, -1], color = 'red')
            else:
                ax.scatter(x[i, 0, :], x[i, 1, :])

        # ax.xlim([-1, 1])
        # ax.ylim([-1, 1])

    def plot_environment(self, env, ax = None):
        """
        env -> A 2D binary array with 1s representing obstacles and 0s representing free space.
        """
        
        if type(ax) == type(None):
            ax = plt
        
        ax.imshow(np.rot90(1 - env), cmap = 'gray')
        ax.xlabel("X-axis")
        ax.ylabel("Y-axis (axis labels are reversed)")
        # ax.show(block=True)  # Display the plot and keep it open

    def plot_points_on_maze(self, points, env, ax = None, color = 'blue'):
        """
        points => (b x c x n) array
        """

        if type(ax) == type(None):
            ax = plt
        
        points = np.array(points)
        
        pixel_coords = self.point_to_pixel(points, env.shape[0], return_int = False)
        self.plot_environment(env, ax)
        ax.scatter(pixel_coords[:, 0, :], env.shape[1] - 1 - pixel_coords[:, 1, :], color = color)
        ax.show(block=True)  # Display the plot and keep it open
