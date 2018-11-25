import numpy as np

class Runner(object):
    def __init__(self, env, horizon, gamma, lam):
        self.env = env
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.frames = 0

    def run(self, agent):
        obs, actions, probs, values, rewards, masks = [], [], [], [], [], []
        state = self.env.obs
        total_rewards = []
        for t in range(self.horizon):
            action, prob, value = agent.sample_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            obs.append(state)
            actions.append(action)
            probs.append(prob)
            values.append(value)
            rewards.append(reward)
            masks.append(np.invert(done))

            state = next_state
            for i in info:
                ep_info = i.get('episode')
                if ep_info: total_rewards.append(ep_info['total_reward'])
            self.frames += self.env.num_envs

        _, __, value = agent.sample_action(state)
        values.append(value)

        obs, actions, probs, values, rewards, masks = map(np.array, [obs, actions, probs, values, rewards, masks])

        advantages, returns = self._calculate_gae(values, rewards, masks)
        transitions = self._return_transitions(obs, actions, probs, values[:-1], advantages, returns)
        return transitions, total_rewards

    def _calculate_gae(self, values, rewards, masks):
        returns, advantages = np.zeros_like(rewards), np.zeros_like(rewards)
        deltas = rewards + self.gamma * values[1:] * masks - values[:-1]
        gae = 0
        for t in reversed(range(self.horizon)):
            advantages[t] = gae = deltas[t] + self.gamma * self.lam * masks[t] * gae
        returns = advantages + values[:-1]
        advantages = (advantages - np.mean(advantages))/(np.std(advantages) + 1e-8)
        return advantages, returns

    def _return_transitions(self, *args):
        transitions = []
        for n in range(self.env.num_envs):
            transitions += list(zip(*[a[:, n] for a in args]))
        return transitions


