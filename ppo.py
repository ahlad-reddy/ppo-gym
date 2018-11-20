import tensorflow as tf 
import numpy as np
import gym
import argparse
from operator import itemgetter 

from lib.utils import make_logdir
from lib.models import PPO
from lib.runner import Runner
from lib.wrappers import wrap_env
from lib.logger import Logger


def parse_args():
    desc = "Implementation of Proximal Policy Optimization for OpenAI Gym environments."  
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, help='Gym Environment', default="PongNoFrameskip-v4")

    parser.add_argument('--iterations', type=int, help='Number of training iterations', default=1000)

    parser.add_argument('--num_envs', type=int, help='Number of episodes to run per iteration', default=8)

    parser.add_argument('--timesteps', type=int, help='Number of timesteps per episode', default=256)

    parser.add_argument('--epochs', type=int, help='Number of training epochs per iteration', default=10)

    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)

    parser.add_argument('--lr', type=float, help='Learning Rate', default=2.5e-4)

    parser.add_argument('--gamma', type=float, help='Discount Factor', default=0.99)

    parser.add_argument('--lam', type=float, help='GAE Parameter', default=0.95)

    parser.add_argument('--clip_param', type=float, help='Gradient Clip Parameter', default=0.2)

    parser.add_argument('--entropy_coef', type=float, help='Entropy Loss Coefficient', default=0.01)

    parser.add_argument('--vf_coef', type=float, help='Value Function Loss Coefficient', default=1.0)

    parser.add_argument('--render', type=bool, default=False, help='Render the game environment.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logdir = make_logdir(args.env)

    print('Creating game environment for {}...'.format(args.env))
    env = gym.make(args.env)
    if env.observation_space.shape == (210, 160, 3):
        env = wrap_env(env)

    runner = Runner(env, args.timesteps, args.gamma, args.lam)

    print("Building agent...")
    agent = PPO(input_shape = (None, *env.observation_space.shape), 
                num_actions = env.action_space.n, 
                lr          = args.lr, 
                clip_param  = args.clip_param, 
                entropy_coef= args.entropy_coef, 
                vf_coef     = args.vf_coef, 
                logdir      = logdir)

    logger = Logger(agent.g, logdir)

    rewards = []
    for i in range(args.iterations):
        print('Iteration {}/{}'.format(i, args.iterations))
        exp_buffer = []
        print("Running environments...")
        for e in range(args.num_envs):
            trajectory, total_reward = runner.run_episode(agent)
            exp_buffer += trajectory
            rewards.append(total_reward)
            logger.log_reward(total_reward, np.mean(rewards[-100:]), runner.episode_count)

        print("Updating policy...")
        for e in range(args.epochs):
            pg_losses, vf_losses, entropy_losses, total_losses = [], [], [], []
            for b in range(len(exp_buffer)//args.batch_size):
                indices = np.random.choice(len(exp_buffer), args.batch_size, replace=False)
                batch = itemgetter(*indices)(exp_buffer)
                pg_loss, vf_loss, entropy_loss, loss = agent.update_policy(*zip(*batch))

                pg_losses.append(pg_loss)
                vf_losses.append(vf_loss)
                entropy_losses.append(entropy_loss)
                total_losses.append(loss)

            logger.log_losses(np.mean(pg_losses), np.mean(vf_losses), np.mean(entropy_losses), np.mean(total_losses), i*args.epochs+e)



if __name__ == '__main__':
    main()