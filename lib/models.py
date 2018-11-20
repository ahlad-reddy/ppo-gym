import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os


class PPO(object):
    def __init__(self, input_shape, num_actions, lr, clip_param, entropy_coef, vf_coef, logdir):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.lr = lr
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.logdir = logdir
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.global_step = tf.Variable(0, trainable=False)
            self._placeholders()
            self._network()
            self._compute_loss()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=3)

    def _placeholders(self):
        self.observation = tf.placeholder(tf.float32, self.input_shape)
        self.action = tf.placeholder(tf.int32, (None, ))
        self.policy_old = tf.placeholder(tf.float32, (None, ))
        self.advantage = tf.placeholder(tf.float32, (None, ))
        self.total_return = tf.placeholder(tf.float32, (None, ))

    def _network(self):
        if len(self.input_shape) == 4:
            latent = self._cnn()
            out = slim.fully_connected(latent, 512)
        else:
            latent = self._mlp()
            out = latent

        logits = slim.fully_connected(out, self.num_actions, None)
        self.policy = slim.softmax(logits)

        self.value_fn = slim.fully_connected(latent, 1, None)

    def _cnn(self):
        conv_1 = slim.conv2d(self.observation, 32, 8, 4)
        conv_2 = slim.conv2d(conv_1, 64, 4, 2)
        conv_3 = slim.conv2d(conv_2, 64, 3, 1) 
        flatten = slim.flatten(conv_3)
        return flatten

    def _mlp(self):
        fc_1 = slim.fully_connected(self.observation, 64, tf.nn.tanh)
        fc_2 = slim.fully_connected(fc_1, 64, tf.nn.tanh)
        return fc_2

    def _compute_loss(self):
        mask = tf.one_hot(self.action, self.num_actions, on_value=True, off_value=False, dtype=tf.bool)
        policy_new = tf.boolean_mask(self.policy, mask)
        ratio = policy_new / self.policy_old
        pg_loss_1 = self.advantage * ratio
        pg_loss_2 = self.advantage * tf.clip_by_value(ratio, 1-self.clip_param, 1+self.clip_param)
        self.pg_loss = -tf.reduce_mean(tf.minimum(pg_loss_1, pg_loss_2))

        self.vf_loss = tf.losses.mean_squared_error(self.total_return, tf.squeeze(self.value_fn))

        entropy = -tf.reduce_sum(self.policy * tf.log(self.policy), axis=1)
        self.entropy_loss = tf.reduce_mean(entropy)

        self.loss = self.pg_loss + self.vf_coef * self.vf_loss - self.entropy_coef * self.entropy_loss
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def save_model(self):
        save_path = self.saver.save(self.sess, os.path.join(self.logdir, 'model.ckpt'), global_step=self.global_step)
        print("Model saved in path: %s" % save_path)
        return save_path

    def load_model(self, model_path):
        print('Loading model from %s' % model_path)
        self.saver.restore(self.sess, model_path)

    def sample_action(self, observation):
        p, value = self.sess.run([self.policy, self.value_fn], feed_dict={ self.observation: observation })
        p = p.squeeze()
        value = value.squeeze()
        action = np.random.choice(self.num_actions, p=p)
        return action, p[action], value

    def update_policy(self, observation, action, policy_old, advantage, total_return):
        pg_loss, vf_loss, entropy_loss, loss, _ = self.sess.run([self.pg_loss, self.vf_loss, self.entropy_loss, self.loss, self.train_op], feed_dict={ self.observation: observation, self.action: action, self.policy_old: policy_old, self.advantage: advantage, self.total_return: total_return })
        return pg_loss, vf_loss, entropy_loss, loss
