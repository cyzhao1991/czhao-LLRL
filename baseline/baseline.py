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

'''
TO_DO

# class Baseline_FCNN(Baseline):
# 	def __init__(self, net, sess, pms):
# 		super(Baseline_FCNN, self).__init__(self, sess, pms)
# 		self.net = net

# 		self.input_ph = self.net.input
# 		self.output_net = self.net.output
# 		self.build_net()

# 	def build_net(self):
# 		with tf.name_scope(self.net.name):
# 			self.optimizer = tf.train.AdamOptimizier(name = 'adam')
# 			self.value = tf.placeholder(tf.float32, [None], name = 'y')
# 			self.loss = tf.losses.mean_squared_error(self.value, self.output_net)
# 			self.train = self.optimizer.minimize(self.loss)

# 	def fit(self, path):
# 		feed_dict = {self.input_ph: path['observations'], self.value: path['returns']}
# 		loss, _ = self.sess.run([self.loss, self.train], feed_dict = feed_dict)
# 		return loss

# 	def predict(self, path):
# 		feed_dict 

