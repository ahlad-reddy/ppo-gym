import os
import glob

def make_logdir(env):
    if not os.path.exists('logdir'): os.mkdir('logdir')
    logdir = 'logdir/{}_{:03d}'.format(env, len(glob.glob('logdir/*')))
    os.mkdir(logdir)
    print('Saving to results to {}'.format(logdir))
    return logdir