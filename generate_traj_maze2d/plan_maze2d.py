import json
import argparse
import numpy as np
from os.path import join
import pdb

from guides.policies import Policy
from d4rl_maze2d_dataset.d4rl import load_environment
from utils.serialization import load_diffusion

def rollout_policy(env, diffusion, dataset, renderer, policy):
    #---------------------------------- main loop ----------------------------------#

    observation = env.reset()   #[6.7004, 0.8052]  #x is down, y is right (axes in plot) (not working for this point)

    #start state: tensor([[ 0.0421,  0.1417,  0.0141, -0.0005]], device='cuda:0')
    #end state: tensor([[ 0.3114,  0.6512,  0.0737, -0.8078]], device='cuda:0')
    
    # observation = np.array([0.0421,  0.1417,  0.0141, -0.0005])
    if args.conditional:
        print('Resetting target')
        env.set_target(np.array([6.7004,10.3052]))

    ## set conditioning xy position to be the goal
    target = env._target
    target = np.array([6.7004,10.3052])
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    for t in range(env.max_episode_steps):

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:
            cond[0] = observation

            action, samples = policy(cond, batch_size=1)    #samples[0].shape: (1, 384, 6)
            actions = samples.actions[0]   #(384, 2)
            sequence = samples.observations[0]  #(384, 4) - qpose, qvel
        # pdb.set_trace()

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            # pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])   #not clear why we subtract vel at time t in the second term
        # pdb.set_trace()
        ####

        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()



        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'{action}'
        )

        if 'maze2d' in args.dataset:
            xy = next_observation[:2]
            goal = env.unwrapped._target
            print(
                f'maze | pos: {xy} | goal: {goal}'
            )

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        if t % 10 == 0 or terminal:
            fullpath = join(args.savepath, f'{t}.png')

            if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


            #renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

            ## save rollout thus far
            renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

    # logger.finish(t, env.max_episode_steps, score=score, value=0)

    ## save result as a json file
    json_path = join(args.savepath, 'rollout.json')
    json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
        'epoch_diffusion': diffusion_experiment.epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

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
        default="maze2d-large-v1",
        help="dataset",
    )

    parser.add_argument(
        "-diffusion_loadpath",
        "--diffusion_loadpath",
        type=str,
        default="/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/logs/maze2d-large-v1/diffusion",
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

    env = load_environment(args.dataset)

    #---------------------------------- loading ----------------------------------#

    diffusion_experiment = load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer

    policy = Policy(diffusion, dataset.normalizer)

    rollout_policy(env, diffusion, dataset, renderer, policy)

#start state: tensor([[ 0.0421,  0.1417,  0.0141, -0.0005]], device='cuda:0')
#end state: tensor([[ 0.3114,  0.6512,  0.0737, -0.8078]], device='cuda:0')