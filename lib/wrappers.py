'''
From https://github.com/colinskow/move37/blob/master/dqn/lib/wrappers.py
'''


import cv2
import gym
import gym.spaces
import numpy as np
import collections
import asyncio


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user needs to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        img = np.reshape(frame, frame.shape).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*old_space.shape[:-1], old_space.shape[-1]*n_steps), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:, :, :-1] = self.buffer[:, :, 1:]
        self.buffer[:, :, -1] = observation[:, :, 0]
        return self.buffer


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class ParallelEnvWrapper(object):
    def __init__(self, env_fn, env_id, num_envs):
        self.num_envs = num_envs
        self.envs = [env_fn(env_id) for i in range(self.num_envs)]
        self.obs = [env.reset() for env in self.envs]
        self.reward = [0 for i in range(self.num_envs)]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.loop = asyncio.get_event_loop()

    def step(self, actions):
        results = [self.loop.run_until_complete(self._step(i, actions[i])) for i in range(self.num_envs)]
        self.obs, r, d, info = zip(*results)
        return self.obs, r, d, info

    async def _step(self, i, action):
        o, r, d, info = self.envs[i].step(action)
        self.reward[i] += r
        if d == True:
            o = self.envs[i].reset()
            info['episode'] = { 'total_reward': self.reward[i] }
            self.reward[i] = 0
        return o, r, d, info


def make_atari(env_id):
    env = gym.make(env_id)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
