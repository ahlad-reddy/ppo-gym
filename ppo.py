import numpy as np
import gym
import argparse
from operator import itemgetter
from math import ceil

from lib.utils import make_logdir, dist_type
from lib.models import PPO
from lib.runner import Runner
from lib.wrappers import make_atari, make_gym, ParallelEnvWrapper
from lib.logger import Logger


def parse_args():
    desc = "Implementation of Proximal Policy Optimization for OpenAI Gym environments."  
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, help='Gym Environment', default="PongNoFrameskip-v4")

    parser.add_argument('--train', help='Train model', action="store_true")

    parser.add_argument('--timesteps', type=float, help='Number of timesteps', default=1e6)

    parser.add_argument('--num_envs', type=int, help='Number of episodes to run per iteration', default=4)

    parser.add_argument('--horizon', type=int, help='Number of timesteps to run before update', default=128)

    parser.add_argument('--epochs', type=int, help='Number of training epochs per update', default=3)

    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)

    parser.add_argument('--save_freq', type=int, help='Number of updates before saving checkpoint', default=5000)

    parser.add_argument('--lr', type=float, help='Learning Rate', default=2.5e-4)

    parser.add_argument('--lr_anneal', type=bool, help='Anneal learning rate from initial value to zero over course of training', default=True)

    parser.add_argument('--gamma', type=float, help='Discount Factor', default=0.99)

    parser.add_argument('--lam', type=float, help='GAE Parameter', default=0.95)

    parser.add_argument('--clip_param', type=float, help='Gradient Clip Parameter', default=0.1)

    parser.add_argument('--clip_anneal', type=bool, help='Anneal clipping parameter from initial value to zero over course of training', default=True)

    parser.add_argument('--entropy_coef', type=float, help='Entropy Loss Coefficient', default=0.01)

    parser.add_argument('--vf_coef', type=float, help='Value Function Loss Coefficient', default=0.5)

    parser.add_argument('--evaluate', help='Evaluate model', action="store_true")

    parser.add_argument('--model_path', type=str, help='Path to saved model')

    parser.add_argument('--logdir', type=str, help='Path to log directory')

    parser.add_argument('--render', help='Render game environment during evaluation', action="store_true")

    args = parser.parse_args()

    return args


def train(env_fn, logdir, args):
    env = ParallelEnvWrapper(env_fn, args.env, args.num_envs)

    runner = Runner(env, args.horizon, args.gamma, args.lam)

    print("Building agent...")
    agent = PPO(input_shape = (None, *env.observation_space.shape), 
                num_actions = env.action_space.n, 
                dist_type   = dist_type(env),
                entropy_coef= args.entropy_coef, 
                vf_coef     = args.vf_coef, 
                logdir      = logdir)

    logger = Logger(agent.g, logdir)

    iterations = ceil(args.timesteps / (args.num_envs*args.horizon))
    for i in range(iterations):

        transitions, total_rewards = runner.run(agent)

        lr = args.lr * (1 - args.lr_anneal*i/iterations)
        clip_param = args.clip_param * (1 - args.clip_anneal*i/iterations)

        pg_losses, vf_losses, entropy_losses, total_losses = [], [], [], []
        indices = np.arange(len(transitions))
        for e in range(args.epochs):
            np.random.shuffle(indices)
            for b in range(len(transitions)//args.batch_size):
                batch_idx = indices[b*args.batch_size:(b+1)*args.batch_size]
                batch = itemgetter(*batch_idx)(transitions)
                pg_loss, vf_loss, entropy_loss, loss, gs = agent.update_policy(*zip(*batch), lr, clip_param)

                pg_losses.append(pg_loss)
                vf_losses.append(vf_loss)
                entropy_losses.append(entropy_loss)
                total_losses.append(loss)

                if gs % args.save_freq == 0: agent.save_model()

        if total_rewards:
            logger.log_reward(total_rewards, runner.frames)
        logger.log_losses(np.mean(pg_losses), np.mean(vf_losses), np.mean(entropy_losses), np.mean(total_losses), lr, clip_param, runner.frames)
        logger.log_console(runner.frames)

        if runner.frames >= args.timesteps:
            model_path = agent.save_model()
            break

    return model_path


def evaluate(env_fn, logdir, model_path, args):
    env = env_fn(args.env, training=False)
    env = gym.wrappers.Monitor(env, logdir)

    agent = PPO(input_shape = (None, *env.observation_space.shape), 
                num_actions = env.action_space.n,
                dist_type   = dist_type(env))
    agent.load_model(model_path)

    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        if args.render: env.render()
        action, _, __ = agent.sample_action([obs])
        obs, r, done, _ = env.step(action)
        total_reward += r
    print('Total Reward: {}'.format(total_reward))

    env.env.close()


def main():
    args = parse_args()

    logdir = args.logdir if args.logdir else make_logdir(args.env)
    model_path = args.model_path

    print('Creating game environments for {}...'.format(args.env))
    env = gym.make(args.env)
    if env.observation_space.shape == (210, 160, 3):
        env_fn = make_atari
    else:
        env_fn = make_gym

    if args.train:
        model_path = train(env_fn, logdir, args)

    evaluate(env_fn, logdir, model_path, args)


if __name__ == '__main__':
    main()