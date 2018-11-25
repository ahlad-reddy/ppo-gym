import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os


class PPO(object):
    def __init__(self, input_shape, num_actions, entropy_coef=0.01, vf_coef=0.5, logdir=None):
        self.input_shape = input_shape
        self.num_actions = num_actions
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
        self.value_old = tf.placeholder(tf.float32, (None, ))
        self.advantage = tf.placeholder(tf.float32, (None, ))
        self.total_return = tf.placeholder(tf.float32, (None, ))
        self.lr = tf.placeholder(tf.float32)
        self.clip_param = tf.placeholder(tf.float32)

    def _network(self):
        if len(self.input_shape) == 4:
            latent = self._cnn()
        else:
            latent = self._mlp()

        logits = slim.fully_connected(latent, self.num_actions, None)
        self.policy = slim.softmax(logits)

        value_fn = slim.fully_connected(latent, 1, None)
        self.value_fn = tf.squeeze(value_fn)

    def _cnn(self):
        conv_1 = slim.conv2d(self.observation, 32, 8, 4)
        conv_2 = slim.conv2d(conv_1, 64, 4, 2)
        conv_3 = slim.conv2d(conv_2, 64, 3, 1) 
        flatten = slim.flatten(conv_3)
        fc = slim.fully_connected(flatten, 512)
        return fc

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

        vf_loss_1 = tf.losses.mean_squared_error(self.total_return, self.value_fn)
        vf_clipped = self.value_old + tf.clip_by_value(self.value_fn - self.value_old, -self.clip_param, self.clip_param)
        vf_loss_2 = tf.losses.mean_squared_error(self.total_return, vf_clipped)
        self.vf_loss = tf.reduce_mean(tf.maximum(vf_loss_1, vf_loss_2))

        entropy = -tf.reduce_sum(self.policy * tf.log(self.policy), axis=1)
        self.entropy_loss = tf.reduce_mean(entropy)

        self.loss = self.pg_loss + self.vf_coef * self.vf_loss - self.entropy_coef * self.entropy_loss
        params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-5)
        grads_and_var = optimizer.compute_gradients(self.loss, params)
        grads, var = zip(*grads_and_var)
        grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.train_op = optimizer.apply_gradients(list(zip(grads, var)), global_step=self.global_step)
        # self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

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
        prob, value = self.sess.run([self.policy, self.value_fn], feed_dict={ self.observation: observation })
        action = [np.random.choice(self.num_actions, p=p) for p in prob]
        return action, prob[range(len(prob)), action], value

    def update_policy(self, observation, action, policy_old, value_old, advantage, total_return, lr, clip_param):
        pg_loss, vf_loss, entropy_loss, loss, _, gs = self.sess.run([
            self.pg_loss, 
            self.vf_loss, 
            self.entropy_loss, 
            self.loss, 
            self.train_op, 
            self.global_step], 
            feed_dict={ 
                self.observation: observation, 
                self.action: action, 
                self.policy_old: policy_old, 
                self.value_old: value_old, 
                self.advantage: advantage, 
                self.total_return: total_return, 
                self.lr: lr, 
                self.clip_param: clip_param })
        return pg_loss, vf_loss, entropy_loss, loss, gs

