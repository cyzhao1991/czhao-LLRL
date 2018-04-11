from __future__ import print_function
import tensorflow as tf
import numpy as np
from baseline import Baseline
import sys
class Context_Baseline(Baseline):

	def __init__(self, net, sess, pms):
		super(Context_Baseline, self).__init__(sess, pms)
		self.net = net

		self.input = self.net.input
		self.output = self.net.output
		self.context = self.net.context
		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.net.name)]

		self.build_net()

	def build_net(self):
		with tf.name_scope(self.net.name):
			self.optimizer = tf.train.AdamOptimizer(name = 'adam', learning_rate = 0.01)
			self.value = tf.placeholder(tf.float32, [None], name = 'y')
			self.mse = tf.losses.mean_squared_error(self.value, tf.squeeze(self.output))
			self.l2 = tf.add_n([tf.nn.l2_loss(t) for t in self.var_list])
			self.loss = self.mse + 1e-3 * self.l2
			self.grad = self.optimizer.compute_gradients(self.loss, self.var_list)
			self.delta_g = tf.add_n([tf.nn.l2_loss(t) for t, _ in self.grad])
			self.train = self.optimizer.apply_gradients(self.grad)
			# self.train = self.optimizer.minimize(self.loss)

	def fit(self, obs, cns, rns):
		feed_dict = {self.input: obs, self.context: cns, self.value: rns}
		iter_num = 1000
		batch_size = obs.shape[0]
		assert obs.shape[0] == rns.shape[0]
		n = obs.shape[0]
		inds = np.arange(n)
		for i in range(iter_num * (n/batch_size)):
			# batch_inds = np.random.choice(inds, size = batch_size, replace = False)
			# loss, _ = self.sess.run([self.loss, self.train], feed_dict = {self.input: obs[batch_inds], self.value: rns[batch_inds]})
			loss, _, delta_g = self.sess.run([self.loss, self.train, self.delta_g], feed_dict = feed_dict)
			sys.stdout.write('%i-th iteration. Vf loss: %f \r'%(i, loss))
			sys.stdout.flush()
			# if delta_g < .05:
			# 	break
		loss = self.sess.run(self.loss, feed_dict = feed_dict)
		print('Value Network updating finished. Final Value Network Loss: %f'%(loss))
			# return loss

	def predict(self, path):
		feed_dict = {self.input: path['observations'], self.context: path['contexts']}
		return self.sess.run(self.output, feed_dict)

	# def build_net(self):
	# 	with tf.name_scope(self.net.name):
	# 		self.optimizer = tf.train.AdamOptimizer(name = 'adam')
	# 		self.value = tf.placeholder(tf.float32, [None], name = 'y')
	# 		self.loss = tf.losses.mean_squared_error(self.value, tf.squeeze(self.output))
	# 		self.train = self.optimizer.minimize(self.loss)

	# def fit(self, obs, rns):
	# 	feed_dict = {self.input: obs, self.value: rns}
	# 	loss, _ = self.sess.run([self.loss, self.train], feed_dict = feed_dict)
	# 	return loss

	# def predict(self, path):
	# 	feed_dict = {self.input: path['observations']}
	# 	return self.sess.run(self.output, feed_dict)
