from __future__ import print_function
import numpy as np
import tensorflow as tf

class Actor(object):

	def __init__(self, net, sess, pms):

		self.net = net
		self.sess = sess
		self.pms = pms

		self.input_ph = self.net.input
		self.output_net = self.net.output
		if self.pms.with_context:
			self.context_ph = self.net.context
		self.var_list = []

	def get_action(self, inputs):
		raise NotImplementedError

class GaussianActor(Actor):

	def __init__(self, net, sess, pms):
		super(GaussianActor, self).__init__(net, sess, pms)
		with tf.name_scope(self.pms.name_scope):
			self.action_logstd_param = tf.Variable( (0.01*np.random.randn(1, self.net.output_dim)).astype(np.float32) ,name = 'weights_logstd')
			self.action_logstd = tf.tile(self.action_logstd_param, tf.stack( [tf.shape(self.output_net)[0] ,1] ) )
			self.action_std = tf.exp(self.action_logstd)
			self.action_std = tf.maximum(self.action_std, self.pms.min_std)
			self.action_std = tf.minimum(self.action_std, self.pms.max_std)
		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.pms.name_scope)]
	def get_action(self, inputs, contexts = None):

		if len(inputs.shape) < 2:
			inputs = inputs[np.newaxis,:]
		if contexts is not None and len(contexts.shape) < 2:
			contexts = contexts[np.newaxis, :]

		if self.pms.with_context:
			feed_dict = {self.input_ph: inputs, self.context_ph: contexts}
		else:
			feed_dict = {self.input_ph: inputs}
		a_mean, a_std, a_logstd = self.sess.run([self.output_net, self.action_std, self.action_logstd], feed_dict = feed_dict)
		a_mean, a_std, a_logstd = map(np.squeeze, [a_mean, a_std, a_logstd])
		# a_mean = np.tanh(a_mean) * self.pms.max_action
		if self.pms.train_flag:
			action = np.random.normal( a_mean, a_std ) 
		else:
			action = a_mean
		return action, dict(mean = a_mean, std = a_std,logstd = a_logstd)

class DeterministicActor(Actor):

	def __init__(self, net, sess, pms):
		super(DeterministicActor, self).__init__(net, sess, pms)
		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.net.name + '/')]

	def get_action(self, inputs):
		inputs = inputs[np.newaxis,:] if len(inputs.shape)<2 else inputs
		if self.pms.with_context:
			feed_dict = {self.input_ph: inputs, self.context_ph: contexts}
		else:
			feed_dict = {self.input_ph: inputs}
		action = self.sess.run([self.output_net], feed_dict = feed_dict)
		return np.squeeze(action)