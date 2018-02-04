from __future__ import print_function
import numpy as np
import tensorflow as tf
from actor import Actor

class Context_Gaussian_Actor(Actor):

	def __init__(self, net, sess, pms):
		super(Context_Gaussian_Actor, self).__init__(net, sess, pms)
		self.context_ph = self.net.context
		with tf.name_scope(self.pms.name_scope):
			self.action_logstd = tf.Variable( tf.truncated_normal([self.net.output_dim], stddev = 0.01), name = 'logstd_t'%i) 
			# self.action_logstd = tf.tile(self.action_logstd_param, )
			self.action_std = tf.exp(logstd)
			self.action_std = tf.maximum(a_std, self.pms.min_std)
			self.action_std = tf.minimum(a_std, self.pms.max_std)

		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.pms.name_scope)]
		self.shared_var_list = [v for v in self.var_list if 'KB' in v.name]
		self.task_var_list = [v for v in self.var_list if 's_vector' in v.name]
		self.var_list = self.shared_var_list + self.task_var_list

	def get_action(self, inputs, contexts):

		if len(inputs.shape) < 2:
			inputs = inputs[np.newaxis, :]
		if len(contexts.shape)<2:
			contexts = contexts[np.newaxis, :]

		feed_dict = {self.input_ph: inputs, self.context_ph: contexts}

		a_mean, a_std, a_logstd = self.sess.run( [self.output_net, self.action_std,	self.action_logstd], feed_dict = feed_dict )
		a_mean, a_std, a_logstd = map(np.squeeze, [a_mean, a_std, a_logstd] )

		if self.pms.train_flag:
			action = np.random.normal(a_mean, a_std)
		else:
			action = a_mean
		return action, dict(mean = a_mean, std = a_std, logstd = a_logstd)
