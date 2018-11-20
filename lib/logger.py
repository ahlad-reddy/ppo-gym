import tensorflow as tf

class Logger(object):
    def __init__(self, graph, logdir):
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

        self.merged_loss_sum = tf.summary.merge([pg_loss_sum, vf_loss_sum, entropy_loss_sum, total_loss_sum])


        self.reward = tf.placeholder(tf.float32)
        reward_sum = tf.summary.scalar('Total Reward', self.reward)

        self.mean_reward = tf.placeholder(tf.float32)
        mean_reward_sum = tf.summary.scalar('Mean Reward 100', self.mean_reward)

        self.merged_reward_sum = tf.summary.merge([reward_sum, mean_reward_sum])

    def log_losses(self, pg_loss, vf_loss, entropy_loss, total_loss, epoch):
        merged_loss_sum = self.sess.run(self.merged_loss_sum, feed_dict={ self.pg_loss: pg_loss, self.vf_loss: vf_loss, self.entropy_loss: entropy_loss, self.total_loss: total_loss})
        self.writer.add_summary(merged_loss_sum, epoch)

    def log_reward(self, reward, mean_reward, episode):
        merged_reward_sum = self.sess.run(self.merged_reward_sum, feed_dict={ self.reward: reward, self.mean_reward: mean_reward })
        self.writer.add_summary(merged_reward_sum, episode)