import tensorflow as tf
import numpy as np

class Logger(object):
    def __init__(self, graph, logdir):
        self.pg_losses = []
        self.vf_losses = []
        self.ent_losses = []
        self.total_losses = []
        self.rewards = []

        self.g = graph
        self.logdir = logdir
        with self.g.as_default():
            self.writer = tf.summary.FileWriter(self.logdir, self.g)
            self._summaries()
            init_op = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(init_op)

    def _summaries(self):
        self.pg_loss = tf.placeholder(tf.float32)
        pg_loss_sum = tf.summary.scalar('Policy Loss', self.pg_loss)

        self.vf_loss = tf.placeholder(tf.float32)
        vf_loss_sum = tf.summary.scalar('Value Function Loss', self.vf_loss)

        self.entropy_loss = tf.placeholder(tf.float32)
        entropy_loss_sum = tf.summary.scalar('Entropy Loss', self.entropy_loss)

        self.total_loss = tf.placeholder(tf.float32)
        total_loss_sum = tf.summary.scalar('Total Loss', self.total_loss)

        self.lr = tf.placeholder(tf.float32)
        lr_sum = tf.summary.scalar('Learning Rate', self.lr)

        self.clip_param = tf.placeholder(tf.float32)
        cp_sum = tf.summary.scalar('Clipping Parameter', self.clip_param)

        self.merged_loss_sum = tf.summary.merge([pg_loss_sum, vf_loss_sum, entropy_loss_sum, total_loss_sum, lr_sum, cp_sum])

        self.reward = tf.placeholder(tf.float32)
        reward_sum = tf.summary.scalar('Reward', self.reward)

        self.mean_reward = tf.placeholder(tf.float32)
        mean_reward_sum = tf.summary.scalar('Mean Reward 100', self.mean_reward)

        self.merged_reward_sum = tf.summary.merge([reward_sum, mean_reward_sum])

    def log_losses(self, pg_loss, vf_loss, entropy_loss, total_loss, lr, clip_param, frame):
        self.pg_losses.append(pg_loss)
        self.vf_losses.append(vf_loss)
        self.ent_losses.append(entropy_loss)
        self.total_losses.append(total_loss)

        merged_loss_sum = self.sess.run(self.merged_loss_sum, feed_dict={ self.pg_loss: pg_loss, self.vf_loss: vf_loss, self.entropy_loss: entropy_loss, self.total_loss: total_loss, self.lr: lr, self.clip_param: clip_param})
        self.writer.add_summary(merged_loss_sum, frame)

    def log_reward(self, rewards, frame):
        for r in rewards:
            self.rewards.append(r)
            merged_reward_sum = self.sess.run(self.merged_reward_sum, feed_dict={ self.reward: r, self.mean_reward: np.mean(self.rewards[-100:]) })
            self.writer.add_summary(merged_reward_sum, frame)

    def log_console(self, frame):
        print('--------')
        print('Frame:           {}'.format(frame))
        print('Mean Reward 100: {:.1f}'.format(np.mean(self.rewards[-100:])))
        print('Policy Loss:     {:.3f}'.format(self.pg_losses[-1]))
        print('Value Loss:      {:.3f}'.format(self.vf_losses[-1]))
        print('Entropy Loss:    {:.3f}'.format(self.ent_losses[-1]))
        print('Total Loss:      {:.3f}'.format(self.total_losses[-1]))
        print('--------')

