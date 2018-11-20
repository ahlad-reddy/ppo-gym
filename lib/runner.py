import numpy as np

class Runner(object):
    def __init__(self, env, timesteps, gamma, lam):
        self.env = env
        self.timesteps = timesteps
        self.gamma = gamma
        self.lam = lam
        self.episode_count = 0
        self.frame_count = 0

    def run_episode(self, agent):
        obs, actions, probs, values, rewards, masks = [], [], [], [], [], []
        state = self.env.reset()
        for t in range(self.timesteps):
            action, prob, value = agent.sample_action([state])
            next_state, reward, done, _ = self.env.step(action)
            
            obs.append(state)
            actions.append(action)
            probs.append(prob)
            values.append(value)
            rewards.append(reward)
            masks.append(not done)

            state = next_state
            self.frame_count += 1

        self.episode_count += 1
        total_reward = sum(rewards)

        _, __, value = agent.sample_action([state])
        values.append(value)

        advantages, returns = self._calculate_gae(np.array(values), np.array(rewards), np.array(masks))
        trajectory = [(obs[t], actions[t], probs[t], advantages[t], returns[t]) for t in range(self.timesteps)]
        return trajectory, total_reward

    def _calculate_gae(self, values, rewards, masks):
        returns, advantages = [], []
        deltas = rewards + self.gamma * values[1:] * masks - values[:-1]
        gae = 0
        for t in reversed(range(self.timesteps)):
            gae = deltas[t] + self.gamma * self.lam * masks[t] * gae
            advantages.insert(0, gae)
        returns = advantages + values[:-1]
        advantages = (advantages - np.mean(advantages))/(np.std(advantages) + 1e-8)
        return advantages, returns
