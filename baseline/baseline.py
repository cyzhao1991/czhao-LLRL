from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys

class Baseline(object):
	def __init__(self, sess, pms):
		self.sess = sess
		self.pms = pms
		self.var_list = []
		self.boost_baseline = False

	def fit(self, path):
		raise NotImplementedError

	def predict(self, path):
		raise NotImplementedError

class BaselineZeros(Baseline):
	def __init__(self, sess, pms):
		super(BaselineZeros, self).__init__(sess, pms)
		# self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.pms.name_scope)]

	def fit(self, obs, rns):
		return None

	def predict(self, path):
		return np.zeros(len(path['rewards']))

class BaselineFcnn(Baseline):
	def __init__(self, net, sess, pms):
		super(BaselineFcnn, self).__init__(sess, pms)
		self.net = net

		self.input = self.net.input
		self.output = self.net.output
		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.net.name)]
		self.build_net()

	def build_net(self):
		with tf.name_scope(self.net.name):
			self.optimizer = tf.train.AdamOptimizer(name = 'adam', learning_rate = 0.001)
			self.value = tf.placeholder(tf.float32, [None], name = 'y')
			self.mse = tf.losses.mean_squared_error(self.value, tf.squeeze(self.output))
			self.l2 = tf.add_n([tf.nn.l2_loss(t) for t in self.var_list])
			self.loss = self.mse + 1e-3 * self.l2
			self.grad = self.optimizer.compute_gradients(self.loss, self.var_list)

			self.gradients = [v[0] for v in self.grad]
			self.gradients_ph = [tf.placeholder(tf.float32, v.shape) for v in self.var_list]
			# self.delta_g = tf.add_n([tf.nn.l2_loss(t) for t, _ in self.grad])
			self.train = self.optimizer.apply_gradients( [(g,v) for g,v in zip(self.gradients_ph, self.var_list)] )
			# self.train = self.optimizer.minimize(self.loss)

	def fit(self, obs, rns, iter_num = 5):
		# feed_dict = {self.input: obs, self.value: rns}
		data_size, _ = obs.shape
		assert obs.shape[0] == rns.shape[0]
		batch_size = 64
		# n = obs.shape[0]
		# inds = np.arange(n)
		for i in range(int(iter_num)):
			idx = np.random.permutation(data_size)
			start = 0
			batch_gradients = []
			for _ in np.arange( np.floor(data_size/batch_size) ):
				minibatch_obs = obs[idx[start:start+batch_size]]
				minibatch_rns = rns[idx[start:start+batch_size]]
				start += batch_size
			# batch_inds = np.random.choice(inds, size = batch_size, replace = False)
			# loss, _ = self.sess.run([self.loss, self.train], feed_dict = {self.input: obs[batch_inds], self.value: rns[batch_inds]})
				feed_dict = {self.input: minibatch_obs, self.value: minibatch_rns}
				loss, gradients = self.sess.run([self.loss, self.gradients], feed_dict = feed_dict)
				batch_gradients.append(gradients)
				# loss, _, delta_g = self.sess.run([self.loss, self.train, self.delta_g], feed_dict = {self.input: obs, self.value: rns})
			update_grad = [np.mean( np.array([g[n] for g in batch_gradients]), axis = 0) for n in range(len(self.var_list))]
			feed_dict = {g_ph: g for g_ph, g in zip(self.gradients_ph, update_grad)}
			_ = self.sess.run(self.train, feed_dict = feed_dict)
			sys.stdout.write('%i-th iteration. Vf loss: %f \r'%(i, loss))
			sys.stdout.flush()
			# if delta_g < .05:
			# 	break
		loss = self.sess.run(self.loss, feed_dict = {self.input: obs, self.value: rns})
		print('Value Network updating finished. Final Value Network Loss: %f'%(loss))
			# return loss

	def predict(self, path):
		feed_dict = {self.input: path['observations']}
		return self.sess.run(self.output, feed_dict)
