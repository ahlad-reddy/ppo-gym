import os
import glob
import gym

def make_logdir(env):
    if not os.path.exists('logdir'): os.mkdir('logdir')
    logdir = 'logdir/{}_{:03d}'.format(env, len(glob.glob('logdir/*')))
    os.mkdir(logdir)
    print('Saving to results to {}'.format(logdir))
    return logdir

def dist_type(env):
    space = type(env.action_space)
    if space == gym.spaces.Discrete:
        return 'categorical'
    elif space == gym.spaces.Box:
        return 'normal'