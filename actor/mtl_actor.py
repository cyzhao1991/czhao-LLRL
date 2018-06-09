from __future__ import print_function
import numpy as np
import tensorflow as tf
from actor.actor import Actor

class MtlGaussianActor(Actor):

	def __init__(self, mtl_net, sess, pms):
		super(MtlGaussianActor, self).__init__(net, sess, pms)
		self.num_of_tasks = mtl_net.num_of_tasks

		with tf.name_scope(self.name):
			self.action_logstd = [tf.Variable( np.zeros([self.net.output_dim]).astype(np.float32), \
				name = 'logstd_t%i'%i) for i in range(self.num_of_tasks)]
			# self.action_logstd = tf.tile(self.action_logstd_param, )
			self.action_std = [tf.exp(logstd) for logstd in self.action_logstd]

			if isinstance(self.pms.min_std, float):
				min_std = np.ones([1, self.net.output_dim]) * self.pms.min_std
			else:
				min_std = self.pms.min_std
			if isinstance(self.pms.max_std, float):
				max_std = np.ones([1, self.net.output_dim]) * self.pms.max_std
			else:
				max_std = self.pms.max_std
			
			self.action_std = [tf.clip_by_value(ac, min_std, max_std) for ac in self.action_std]

		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.name)]
		self.shared_var_list = [v for v in self.var_list if 'shared' in v.name]
		self.task_var_list = [  [v for v in self.var_list if 't%i'%i in v.name] for i in range(self.num_of_tasks) ]
			

	def get_action(self, inputs, task_indexes):

		if len(inputs.shape) < 2:
			inputs = inputs[np.newaxis, :]

		feed_dict = {self.input_ph: inputs}

		if isinstance(task_indexes, int):
			a_mean, a_std, a_logstd = self.sess.run( [self.output_net[task_indexes], self.action_std[task_indexes], \
				self.action_logstd[task_indexes]], feed_dict = feed_dict )

			a_mean, a_std, a_logstd = map(np.squeeze, [a_mean, a_std, a_logstd] )
			a_logstd = np.log(a_std)
			if self.pms.train_flag:
				action = np.random.normal( a_mean, a_std )
			else:
				action = a_mean
			return action, dict(mean = a_mean, std = a_std, logstd = a_logstd)

		# else if isinstance(task_indexes, list):
