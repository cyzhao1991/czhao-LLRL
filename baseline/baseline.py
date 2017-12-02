import tensorflow as tf
import numpy as np

class Baseline(object):
	def __init__(self, sess, pms):
		self.sess = sess
		self.pms = pms

	def fit(self, path):
		raise NotImplementedError

	def predict(self, path):
		raise NotImplementedError

class Baseline_Zeros(Baseline):
	def __init__(self, sess, pms):
		super(Baseline_Zeros, self).__init__(self, sess, pms)

	def fit(self, path):
		return None

	def predict(self, path):
		return np.zeros(len(path['rewards']))

class Baseline_FCNN(Baseline):
	def __init__(self, net, sess, pms, name = 'baseline_v_net'):
		super(Baseline_FCNN, self).__init__(self, sess, pms)
		self.net = net

		self.input_ph = self.net.input
		self.output_net = self.net.output
		self.optimizer = tf.train.AdamOptimizer()
		self.name = name
		self.build_net()

	def build_net(self):
		with tf.name_scope(self.name):
			self.value = tf.placeholder(tf.float32, [None, ])
			loss = tf.losses.mean_squared_error()


