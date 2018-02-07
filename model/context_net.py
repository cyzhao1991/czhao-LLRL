from __future__ import print_function
import tensorflow as tf
import numpy as np
from net import Net

class Context_Fcnn_Net(Net):

	def __init__(self, sess, input_dim, output_dim, context_dim,layer_dim, module_num, name = None, **kwargs):
		super(Context_Fcnn_Net, self).__init__(sess, input_dim, output_dim, layer_dim, name, **kwargs)
		self.module_num = module_num
		self.context_dim = context_dim
		self.num_of_hidden_layer = len(self.layer_dim)
		self.num_of_layer = self.num_of_hidden_layer + 1

		assert(len(self.if_bias) == self.num_of_hidden_layer + 1)
		assert(len(self.activation_fns_call) == self.num_of_hidden_layer + 1)

		with tf.name_scope(self.name):
			self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name = 'input'))
			self.context = kwargs.get('context_tf', tf.placeholder(tf.float32, [None, self.context_dim], name ='context'))

		self.def_Shared_knowledge(self.name)
		self.def_Task_knowledge(self.name)
		self.output = self.build(self.name)

	def def_Shared_knowledge(self, name):
		self.KB_weights = []
		self.KB_bias = []
		with tf.name_scope(name):
			for i in range( self.num_of_layer ):
				dim_1, dim_2 = self.all_layer_dim[i:i+2]

				self.KB_weights.append( [tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev = 1.0), \
					name = 'KB_weights_h%i_m%i'%(i,j)) for j in range(self.module_num[i])] )
				self.KB_bias.append( [tf.Variable(tf.truncated_normal([dim_2], stddev = 1.0), \
					name = 'KB_bias_h%i_m%i'%(i,j)) if self.if_bias[i] else 0 for j in range(self.module_num[i]) ] )

	def def_Task_knowledge(self, name):
		with tf.name_scope(name):
			self.c_weights = [tf.Variable(tf.truncated_normal([self.context_dim, dim], mean = .1, stddev = .03), \
				name = 's_vector_h%i'%k) for dim,k in zip(self.module_num, range(self.num_of_layer))]
			self.s_vector = [ tf.expand_dims(tf.matmul(self.context, tf.nn.relu(w_c)), 1) for w_c in self.c_weights]
			# self.s_vector = [ tf.expand_dims(tf.matmul(self.context, w_c), 1) for w_c in self.c_weights]


	def build(self, name):
		net = [self.input]
		with tf.name_scope(name):
			for i in range(self.num_of_layer):
				hidden_unit = tf.reduce_mean( tf.stack( [tf.matmul(net[i], weight) + bias for weight, bias in zip(self.KB_weights[i], self.KB_bias[i])] , \
					axis = -1) * self.s_vector[i], axis = -1)
				net.append(self.activation_fns_call[i](hidden_unit))
		return net[-1]