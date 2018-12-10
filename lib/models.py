import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os
import gym


class PPO(object):
    def __init__(self, input_shape, output_shape, num_actions, dist_type, entropy_coef=0.01, vf_coef=0.5, logdir=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_actions = num_actions
        self.dist_type = dist_type
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
        action_dtype = tf.int32 if self.dist_type == 'categorical' else tf.float32
        self.action_old = tf.placeholder(action_dtype, self.output_shape)
        self.logprob_old = tf.placeholder(tf.float32, (None, ))
        self.value_old = tf.placeholder(tf.float32, (None, ))
        self.advantage = tf.placeholder(tf.float32, (None, ))
        self.total_return = tf.placeholder(tf.float32, (None, ))
        self.lr = tf.placeholder(tf.float32)
        self.clip_param = tf.placeholder(tf.float32)

    def _network(self):
        if len(self.input_shape) == 4:
            pg_latent = self._cnn()
            vf_latent = pg_latent
        else:
            pg_latent = self._mlp()
            vf_latent = self._mlp()

        self.dist = self._distribution(pg_latent)

        self.action = self.dist.sample()
        self.logprob = self._logprob(self.action)

        value_fn = slim.fully_connected(vf_latent, 1, None)
        self.value_fn = tf.squeeze(value_fn, axis=-1)

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

    def _distribution(self, latent):
        if self.dist_type == "categorical":
            logits = slim.fully_connected(latent, self.num_actions, None)
            return tf.distributions.Categorical(logits=logits)
        elif self.dist_type == "normal":
            mean = slim.fully_connected(latent, self.num_actions, None)
            logstd = tf.get_variable('logst', shape=(1, self.num_actions), initializer=tf.zeros_initializer())
            return tf.distributions.Normal(loc=mean, scale=tf.exp(logstd))

    def _logprob(self, action):
        return {
            'categorical': self.dist.log_prob(action),
            'normal'     : tf.reduce_sum(self.dist.log_prob(action), axis=-1)
        }[self.dist_type]

    def _compute_loss(self):
        logprob = self._logprob(self.action_old)
        ratio = tf.exp(logprob - self.logprob_old)
        pg_loss_1 = self.advantage * ratio
        pg_loss_2 = self.advantage * tf.clip_by_value(ratio, 1-self.clip_param, 1+self.clip_param)
        self.pg_loss = -tf.reduce_mean(tf.minimum(pg_loss_1, pg_loss_2))

        vf_loss_1 = tf.losses.mean_squared_error(self.total_return, self.value_fn)
        vf_clipped = self.value_old + tf.clip_by_value(self.value_fn - self.value_old, -self.clip_param, self.clip_param)
        vf_loss_2 = tf.losses.mean_squared_error(self.total_return, vf_clipped)
        self.vf_loss = tf.reduce_mean(tf.maximum(vf_loss_1, vf_loss_2))

        self.entropy_loss = tf.reduce_mean(self.dist.entropy())

        self.loss = self.pg_loss + self.vf_coef * self.vf_loss - self.entropy_coef * self.entropy_loss
        params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-5)
        grads_and_var = optimizer.compute_gradients(self.loss, params)
        grads, var = zip(*grads_and_var)
        grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.train_op = optimizer.apply_gradients(list(zip(grads, var)), global_step=self.global_step)

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
        return self.sess.run([self.action, self.logprob, self.value_fn], feed_dict={ self.observation: observation })

    def update_policy(self, observation, action_old, logprob_old, value_old, advantage, total_return, lr, clip_param):
        pg_loss, vf_loss, entropy_loss, loss, _, gs = self.sess.run([
            self.pg_loss, 
            self.vf_loss, 
            self.entropy_loss, 
            self.loss, 
            self.train_op, 
            self.global_step], 
            feed_dict={ 
                self.observation: observation, 
                self.action_old: action_old, 
                self.logprob_old: logprob_old, 
                self.value_old: value_old, 
                self.advantage: advantage, 
                self.total_return: total_return, 
                self.lr: lr, 
                self.clip_param: clip_param })
        return pg_loss, vf_loss, entropy_loss, loss, gs



def build_agent(env, logdir, args):
    input_shape = (None, *env.observation_space.shape)
    output_shape = (None, *env.action_space.shape)
    num_actions = output_shape[-1] or env.action_space.n
    dist_type = {
        gym.spaces.Discrete : 'categorical',
        gym.spaces.Box      : 'normal'        
    }[type(env.action_space)]
    
    return PPO( input_shape = input_shape, 
                output_shape= output_shape, 
                num_actions = num_actions,
                dist_type   = dist_type,
                entropy_coef= args.entropy_coef, 
                vf_coef     = args.vf_coef, 
                logdir      = logdir)


