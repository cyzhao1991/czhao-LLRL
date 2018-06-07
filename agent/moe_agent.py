from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
from agent import Agent

class MoeAgent(Agent):
	def __init__(self, session, pms, net, saver = None):
		super(MoeAgent, self)__init__(None, session, pms, saver)
		self.net = net
		self.init_vars()

	def init_vars(self):
		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.net.name)]

		self.input_ph = self.net.input
		self.output = self.net.output
		self.label_ph = tf.placeholder(tf.float32, [None, self.net.output_dim], name = 'label_placeholder')
		self.optimizer = tf.train.AdamOptimizer()
		self.mse_loss = self.losses.mean_suqared_error(self.label_ph, self.output)
		self.l2_loss = tf.add_n( [tf.nn.l2_loss(v) for v in self.var_list if 'KB' in v.name] )
		self.l1_loss = tf.add_n( [tf.reduce_sum(tf.abs(v)) for v in self.all_g])

		self.loss = self.mse_loss + .0005 * self.l1_loss + .0005 * self.l2_loss

		self.gradients = self.optimizer.compute_gradients(self.loss, var_list = self.var_list)
		self.train_op = self.optimizer.apply_gradients( self.gradients)

	def learn(self, x, y, iter_num = 5000):
		data_size = x1.shape
		for itern in range(iter_num):
			idx = np.random.permutation(data_size)
			start = 0
			for i in range(100):
				minibatch_x = x[idx[start:start + 256]]
				minibatch_y = y[idx[start:start + 256]]

				start += 256
				feed_dict = {self.input_ph: minibatch_x, self.label_ph: minibatch_y}

				loss, train_op = self.sess.run( [self.loss, self.train_op], feed_dict = feed_dict)

			test_idx = np.random.permutation(data_size)
			feed_dict[self.input_ph] = x[test_idx[:10000]]
			feed_dict[self.label_ph] = y[test_idx[:10000]]
			loss, mse_loss, l1_loss, l2_loss= self.sess.run([self.loss, self.mse_loss, self.l1_loss, self.l2_loss], feed_dict = feed_dict)
			print('Iteration: %i, loss: %3.2f, mse_loss: %3.2f, l2_loss: %3.2f, l1_loss: %3.2f'%(itern, loss, mse_loss, l2_loss, l1_loss))