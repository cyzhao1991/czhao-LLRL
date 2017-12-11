from __future__ import print_function
import tensorflow as tf
import numpy as np

class Net(object):

	def __init__(self, sess):
		self.sess = sess

	def build(self):
		raise NotImplementedError


class Fcnn_2side(Net):

	def __init__(self, sess, input_dim1, input_dim2, output_dim, layer_dim_1, layer_dim_2, layer_dim_o = [], name = None, **kwargs):
		super(Fcnn_2side, self).__init__(sess)
		self.input_dim1 = input_dim1
		self.input_dim2 = input_dim2
		self.output_dim = output_dim
		self.layer_dim_1 = layer_dim_1
		self.layer_dim_2 = layer_dim_2
		self.layer_dim_o = layer_dim_o
		self.name = name

		self.merge_method = kwargs['merge_method'] if 'merge_method' in kwargs.keys() else 'add'
		self.if_bias = kwargs['if_bias'] if 'if_bias' in kwargs.keys() else True

		self.all_layer_dim_1 = np.concatenate([[self.input_dim1], layer_dim_1], axis = 0)
		self.all_layer_dim_2 = np.concatenate([[self.input_dim2], layer_dim_2], axis = 0)
		if self.merge_method is 'concatenate':
			self.all_layer_dim_o = np.concatenate([[self.layer_dim_1[-1] + self.layer_dim_2[-1]], layer_dim_o], axis = 0)
		else:
			assert(self.layer_dim_1[-1] == self.layer_dim_2[-1])
			self.all_layer_dim_o = np.concatenate([[self.layer_dim_1[-1]], layer_dim_o], axis = 0)

		self.input_1, self.input_2, self.output = self.build(name = self.name)

	def build(self, name):

		with tf.name_scope(name):
			self.input_1 = tf.placeholder(tf.float32, [None, self.input_dim1], name = 'input1')
			self.input_2 = tf.placeholder(tf.float32, [None, self.input_dim2], name = 'input2')
			net_1 = []
			net_2 = []
			for i, dim in enumerate(self.all_layer_dim_1):
				


