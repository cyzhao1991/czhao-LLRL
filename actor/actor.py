import numpy as np
import tensorflow as tf

class Actor(object):

	def __init__(self, net, sess, pms):

		self.net = net
		self.sess = sess
		self.pms = pms

		self.input_ph = self.net.input
		self.output_net = self.net.output

		self.var_list = []

	def get_action(self, inputs):
		raise NotImplementedError

class Gaussian_Actor(Actor):

	def __init__(self, net, pms):
		super(Gaussian_Actor, self).__init__(self, net, sess, pms)
		with tf.name_scope(self.net.name):
			self.action_logstd_param = tf.Variable( (0.01*np.random.randn(1, self.net.output_dim)).astype(np.float32) ,name = 'weights_logstd')
			self.action_logstd = tf.tile(self.action_logstd_param, tf.stack( [tf.shape(self.output_net)[0].,1] ) )
			self.action_std = tf.exp(self.action_logstd)
			self.action_std = tf.maximum(self.action_std, self.pms.min_std)
			self.action_std = tf.minimum(self.action_std, self.pms.max_std)
		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith('self.pms.name')]

	def get_action(self, inputs):

		if len(inputs.shape) < 2:
			inputs = inputs[np.newaxis,:]

		feed_dict = {self.input_ph: inputs}
		a_mean, a_std, a_logstd = self.sess.run([self.output_net, self.action_std, self.action_logstd], feed_dict = feed_dict)
		a_mean, a_std, a_logstd = map(np.squeeze, [a_mean, a_std, a_logstd])
		if self.pms.train_flag:
			action = np.random.normal( a_mean, a_std )
		else:
			action = a_mean
		return action, dict(mean = a_mean, std = a_std,log_std = a_logstd)


