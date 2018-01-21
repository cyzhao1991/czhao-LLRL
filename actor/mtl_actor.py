from __future__ import print_function
import numpy as np
import tensorflow as tf
from actor import Actor

class Mtl_Gaussian_Actor(Actor):

	def __init__(self, net, sess, pms, num_of_tasks):
		super(Mtl_Gaussian_Actor, self).__init__(net, sess, pms)
		self.num_of_tasks = num_of_tasks

		with tf.name_scope(self.pms.name_scope):
			self.action_logstd = [tf.Variable( tf.truncated_normal([self.net.output_dim], stddev = 0.01), \
				name = 'logstd_t%i_'%i) for i in range(self.num_of_tasks)]
			# self.action_logstd = tf.tile(self.action_logstd_param, )
			self.action_std = [tf.exp(logstd) for logstd in self.action_logstd]
			self.action_std = [tf.maximum(a_std, self.pms.min_std) for a_std in self.action_std]
			self.action_std = [tf.minimum(a_std, self.pms.max_std) for a_std in self.action_std]

		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.pms.name_scope)]
		self.shared_var_list = [v for v in self.var_list if 'KB' in v.name]
		self.task_var_list = [  [v for v in self.var_list if 't%i_'%i in v.name] for i in range(self.num_of_tasks)]
			

	def get_action(self, inputs, task_indexes):

		if len(inputs.shape) < 2:
			inputs = inputs[np.newaxis, :]

		feed_dict = {self.input_ph: inputs}

		if isinstance(task_indexes, int):
			a_mean, a_std, a_logstd = self.sess.run( [self.output_net[task_indexes], self.action_std[task_indexes], \
				self.action_logstd[task_indexes]], feed_dict = feed_dict )

			a_mean, a_std, a_logstd = map(np.squeeze, [a_mean, a_std, a_logstd] )
			if self.pms.train_flag:
				action = np.random.normal( a_mean, a_std )
			else:
				action = a_mean
			return action, dict(mean = a_mean, std = a_std, logstd = a_logstd)

		# else if isinstance(task_indexes, list):
