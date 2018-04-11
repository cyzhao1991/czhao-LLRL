from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.utils import *
from agent.agent import Agent
# from dm_control.rl import control
import matplotlib.pyplot as plt

class MimicAgent(Agent):

	def __init__(self, env, session, pms, net, saver = None):
		super(MimicAgent, self).__init__(env, session, pms, saver)
		self.net = net
		self.init_vars()

	def init_vars(self):
		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.pms.name_scope)]
		# self.input_ph = tf.placeholder( tf.float32, [None, self.pms.obs])
		self.input_ph = self.net.input
		self.output   =self.net.output
		self.output_ph = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'output_placeholder')
		self.optimizer = tf.train.AdamOptimizer()
		self.mse_loss = tf.losses.mean_squared_error(self.output_ph, self.output)
		self.l2_loss = tf.add_n( [tf.nn.l2_loss(v) for v in self.var_list] )
		self.loss = self.mse_loss + 0.001 * self.l2_loss
		self.gradients = self.optimizer.compute_gradients( self.loss, var_list = self.var_list )
		self.gradients = [v[0] for v in self.gradients]
		self.gradients_ph = [tf.placeholder(tf.float32, v.shape) for v in self.var_list]
		self.train_op = self.optimizer.apply_gradients( [(g,v) for g, v in zip( self.gradients_ph, self.var_list)] )
		# self.train_op = self.optimizer.minimize(self.loss)


	def learn(self, x, y):
		data_size, _ = x.shape
		for itern in range(10000):
			idx = np.random.permutation(data_size)
			start = 0
			batch_gradients = []
			for i in range(100):
				minibatch_x = x[idx[start:start+64]]
				minibatch_y = y[idx[start:start+64]]
				start += 64
				feed_dict = {self.input_ph: minibatch_x, self.output_ph: minibatch_y}
				loss, gradients = self.sess.run([self.loss, self.gradients], feed_dict = feed_dict)
				batch_gradients.append(gradients)
				# gradients.append( self.sess.run([self.gradients], feed_dict = self.var_list) )
			update_grad = [np.mean( np.array([g[n] for g in batch_gradients]), axis = 0 ) for n in range(len(self.var_list))]
			# print([v.shape for v in update_grad])
			feed_dict = {g_ph: g for g_ph, g in zip(self.gradients_ph, update_grad)}

			test_idx = np.random.permutation(data_size)
			feed_dict[self.input_ph] = x[test_idx[:10000]]
			feed_dict[self.output_ph] = y[test_idx[:10000]]
			_, loss = self.sess.run([self.train_op, self.loss], feed_dict = feed_dict)
			print('Iteration: %i, loss: %3.2f'%(itern, loss))
		# for _ in range(10000):


class MtlMimicAgent(Agent):

	def __init__(self, env, session, pms, net, saver = None):
		super(MtlMimicAgent, self).__init__(env, session, pms, saver)
		self.net = net
		self.init_vars()

	def init_vars(self):
		self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.pms.name_scope)]
		self.KB_var_list = [v for v in self.var_list if 'KB' in v.name]
		self.s_var_list = [v for v in self.var_list if 's_vector' in v.name]
		# self.input_ph = tf.placeholder( tf.float32, [None, self.pms.obs])
		self.input_ph = self.net.input
		self.context_ph=self.net.context
		self.output   =self.net.output
		self.output_ph = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'output_placeholder')
		self.optimizer = tf.train.AdamOptimizer()
		self.mse_loss = tf.losses.mean_squared_error(self.output_ph, self.output)
		self.l2_loss = 0.001* tf.add_n( [tf.nn.l2_loss(v) for v in self.KB_var_list] )
		self.l1_loss = 0.001 * tf.add_n( [tf.reduce_sum(tf.abs(v)) for v in self.s_var_list] )
		self.loss = self.mse_loss + self.l2_loss + self.l1_loss
		self.gradients = self.optimizer.compute_gradients( self.loss, var_list = self.var_list )
		self.gradients = [v[0] for v in self.gradients]
		self.gradients_ph = [tf.placeholder(tf.float32, v.shape) for v in self.var_list]
		self.train_op = self.optimizer.apply_gradients( [(g,v) for g, v in zip( self.gradients_ph, self.var_list)] )
		# self.train_op = self.optimizer.minimize(self.loss)


	def learn(self, x1, x2, y):
		data_size, _ = x1.shape
		for itern in range(10000):
			idx = np.random.permutation(data_size)
			start = 0
			batch_gradients = []
			for i in range(100):
				minibatch_x1 = x1[idx[start:start+256]]
				minibatch_x2 = x2[idx[start:start+256]]
				minibatch_y  = y[idx[start:start+256]]
				start += 256
				feed_dict = {self.input_ph: minibatch_x1, self.context_ph: minibatch_x2, self.output_ph: minibatch_y}
				loss, gradients = self.sess.run([self.loss, self.gradients], feed_dict = feed_dict)
				batch_gradients.append(gradients)
				# gradients.append( self.sess.run([self.gradients], feed_dict = self.var_list) )
			update_grad = [np.mean( np.array([g[n] for g in batch_gradients]), axis = 0 ) for n in range(len(self.var_list))]
			# print([v.shape for v in update_grad])
			feed_dict = {g_ph: g for g_ph, g in zip(self.gradients_ph, update_grad)}

			test_idx = np.random.permutation(data_size)
			feed_dict[self.input_ph] = x1[test_idx[:10000]]
			feed_dict[self.context_ph] = x2[test_idx[:10000]]
			feed_dict[self.output_ph] = y[test_idx[:10000]]
			_, loss, mse_loss, l1_loss, l2_loss= self.sess.run([self.train_op, self.loss, self.mse_loss, self.l1_loss, self.l2_loss], feed_dict = feed_dict)
			print('Iteration: %i, loss: %3.2f, mse_loss: %3.2f, l2_loss: %3.2f, l1_loss: %3.2f'%(itern, loss, mse_loss, l2_loss, l1_loss))