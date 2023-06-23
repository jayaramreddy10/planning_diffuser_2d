import torch
import time
import os
import argparse
import matplotlib.pyplot as plt
import einops

from diffusion_fns import *
from guide_fns import *
from models import TemporalUnet_jayaram
from diffusion import GaussianDiffusion_jayaram
from diffusion_fns import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------- Edit these parameters --------------------------------- #

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False

if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument(
    #     "-i", "--input-data-path", help="Path to training data."
    # )

    parser.add_argument(
        "-l",
        "--logbase",
        type=str,
        default="/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/logs",
        help="Path to logs folder.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="maze2d-test",
        help="dataset",
    )

    parser.add_argument(
        "-diffusion_loadpath",
        "--diffusion_loadpath",
        type=str,
        default="/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/logs/maze2d-test/diffusion",
        help="save checkpoints of diffusion model while training",
    )

    parser.add_argument(
        "-diffusion_epoch", "--diffusion_epoch", type=str, default="latest", help="Diffusion model latest epoch"
    )

    parser.add_argument(
        "-savepath",
        "--savepath",
        type=str,
        default="/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/logs/maze2d-large-v1/plans",
        help="save inference results",
    )

    parser.add_argument(
        "-config",
        "--config",
        type=str,
        default="config.maze2d",
        help="config",
    )

    parser.add_argument(
        "-cond",
        "--conditional",
        type=str2bool,
        default=True,
        help="conditioned on.",
    )

    args = parser.parse_args()

    # env = load_environment(args.dataset)

    #---------------------------------- loading ----------------------------------#

    # diffusion_experiment = load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
    # diffusion = diffusion_experiment.ema

    # dataset = diffusion_experiment.dataset
    # renderer = diffusion_experiment.renderer

    # policy = Policy(diffusion, dataset.normalizer)

    checkpoint_path = "/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/logs/maze2d-test/diffusion/state_107.pt"

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Access the desired variables from the checkpoint
    model_state_dict = checkpoint['model']
    # optimizer_state_dict = checkpoint['optimizer']
    # epoch = checkpoint['epoch']

    # Create the model and optimizer objects
    horizon = 256
    action_dim = 0
    state_dim = 2
    transition_dim = state_dim + action_dim
    cond_dim = 2
    #load model architecture 
    model = TemporalUnet_jayaram(horizon, transition_dim, cond_dim)
    model = model.to(device)

    diffusion = GaussianDiffusion_jayaram(model, horizon, state_dim, action_dim, device)
    diffusion = diffusion.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Load the model and optimizer states from the checkpoint
    diffusion.load_state_dict(model_state_dict)

    traj_len = 256
    T = 255
    # batch_size = 2048

    # env_name = "two_pillars"

    point_list = np.array([[-0.85 ,  -0.83], 
                        #    [-0.4 ,  0.5],
                        #    [0.4 , -0.5],
                        [0.85 ,  -0.83]])

    guide_lr = 1

    # model_name = "Model_weights/bezier2d/" + "TemporalUNetModel_10_Points_Bezier_T" + str(T) + "_N" + str(traj_len)

    # ----------------------------------------------------------------------------------------- #



    #load env
    file_num = 500
    envfile = "/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/generate_traj_maze2d/5x5/maze" + str(file_num) + ".npy"

    data = np.load(envfile)   #(5, 5)
                # h ,w = data.shape[0], data.shape[1]
    w , h = 1024, 1024
    num_waypoints = 256
    img = convert_maze_to_image(data, w, h)   #(1024, 1024)
    cost_map = np.load("cost_map.npy")
    gradient_map = np.load("gradient_map.npy")

    start_pixel= guide_point_to_pixel(np.array(point_list[0]).reshape((1, 2, 1)), img.shape)
    goal_pixel = guide_point_to_pixel(np.array(point_list[-1]).reshape((1, 2, 1)), img.shape)
    print('start pixel: {}'.format(start_pixel))
    print('goal pixel: {}'.format(goal_pixel))


    if img[start_pixel[0][0], start_pixel[0][1]] == 1 or img[goal_pixel[0][0], goal_pixel[0][1]] == 1:

        raise Exception("Either the Start or Goal point lies on an obstacle.")

    # if not os.path.exists(model_name):
        
    #     raise Exception("Model does not exist for these parameters. Train a model first.")


    # denoiser = TemporalUnet_jayaram(model_name = model_name, input_dim = 2, time_dim = 32)
    # _ = denoiser.to(device)

    beta = schedule_variance(T)
    alpha = 1 - beta
    alpha_bar = np.array(list(map(lambda t:np.prod(alpha[:t]), np.arange(T+1)[1:])))

    num_trajs = point_list.shape[0] - 1
    traj_means = np.array([(point_list[i] + point_list[i+1])/2 for i in range(num_trajs)]).reshape((num_trajs, 2, 1))

    mean = np.zeros(traj_len)
    cov = np.eye(traj_len)

    xT = np.random.multivariate_normal(mean, cov, (num_trajs, 2)) + traj_means
    X_t = xT.copy()

    # CONDITIONING:
    X_t[:, :, 1] = point_list[:-1, :]
    X_t[:, :, -2] = point_list[1:, :]
    X_t[:, :, 0] = point_list[:-1, :]
    X_t[:, :, -1] = point_list[1:, :]
    cond = {0 : point_list[:-1, :], num_waypoints - 1: point_list[1:, :]}

    reverse_diff_traj = np.zeros((T+1, 2, traj_len*(num_trajs)))
    # reverse_diff_traj[T] = einops.rearrange(xT, 'b c n -> c (b n)').copy()

    diffusion.train(False)

    start_time = time.time()
    
    for t in range(T, 0, -1):

        if t > 1:
            z = np.random.multivariate_normal(mean, cov, (num_trajs, 2))
        else:
            z = np.zeros((num_trajs, 2, traj_len))
        
        X_input = torch.tensor(X_t, dtype = torch.float32).to(device)
        time_in = torch.tensor([t], dtype = torch.float32).to(device)

        epsilon = diffusion.model(X_input, cond, time_in).numpy(force=True)
        print(X_t[0, 1, 200])
        # posterier_mean = (1/np.sqrt(alpha[t-1])) * (X_t - ((1 - alpha[t-1])/(np.sqrt(1 - alpha_bar[t-1]))) * epsilon)
        # if t > 1:
        #     posterier_var = ((1 - alpha_bar[t-2]) / (1 - alpha_bar[t-1])) * beta[t-1]

        # for ch in range(X_t.shape[1]):
        #     X_t[0, ch, :] = np.random.multivariate_normal(mean = posterier_mean[0, ch, :], cov = beta[t-1] * cov)
        
        gradient = 0.05*get_gradient(gradient_map, X_t) #- 0.005*length_gradient(X_t) # Add your new gradient func here
        X_t = (1/np.sqrt(alpha[t-1])) * (X_t - ((1 - alpha[t-1])/(np.sqrt(1 - alpha_bar[t-1]))) * epsilon) + beta[t-1]*z #+ gradient
        
        # CONDITIONING:
        X_t[:, :, -2] = point_list[:-1, :]
        X_t[:, :, 1] = point_list[1:, :]
        X_t[:, :, -1] = point_list[:-1, :]
        X_t[:, :, 0] = point_list[1:, :]
        
        reverse_diff_traj[t-1] = einops.rearrange(X_t, 'b c n -> c (b n)').copy()

        os.system("cls")
        print("Denoised to ", t-1, " steps")

    time_taken = time.time() - start_time

    np.save("reverse_guided_trajectory" + ".npy", reverse_diff_traj)  #(T, 2, len_traj)

    # #inference
    # z = np.random.multivariate_normal(mean, cov, (num_trajs, 2))

    # X_input = torch.tensor(X_t, dtype = torch.float32).to(device)
    # cond = {0 : X_input[0], num_waypoints - 1: X_input[-1]}
    # time_in = torch.tensor([T], dtype = torch.float32).to(device)

    # for i in range(T-1, -1, -1):
    #     print(f"\rTime step: {i}", end="")
    #     time_in = torch.tensor([i], dtype = torch.float32).to(device)
    #     xhat_0 = diffusion.model(X_input, cond, time_in)#.numpy(force=True)
    #     if i==0:
    #         break
    #     X_input[:, :, 1:255] = xhat_0[:, :, 1:255]
    #     new_X_input = diffusion.q_sample(X_input, torch.tensor([i - 1], dtype = torch.int64).to(device))
    #     new_X_input[:, :, 0] = torch.tensor(point_list[:-1, :], dtype = torch.float32).to(device)
    #     new_X_input[:, :, -1] = torch.tensor(point_list[1:, :], dtype = torch.float32).to(device)

    #     X_input = new_X_input

    # xhat_0 = xhat_0.cpu().detach().numpy()

    # np.save("reverse_guided_trajectory"_+ ".npy", reverse_diff_traj)
    # plt.imshow(np.rot90(img), cmap = 'gray')
    # pixel_traj = guide_point_to_pixel(xhat_0, img.shape)
    # pixel_traj[:, 1, :] = img.shape[1] - 1 - pixel_traj[:, 1, :]
    # plt.scatter(pixel_traj[0, 0, 1:-1], pixel_traj[0, 1, 1:-1])
    # plt.scatter(pixel_traj[0, 0, 0], pixel_traj[0, 1, 0], color = 'green')
    # plt.scatter(pixel_traj[0, 0, -1], pixel_traj[0, 1, -1], color = 'red')

    # plt.set_xlim([-1, 1])
    # plt.set_ylim([-1, 1])
    # print("jai")

    # print("Denoising took ", np.round(time_taken, 2), " seconds")



