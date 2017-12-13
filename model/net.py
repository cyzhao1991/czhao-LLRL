from __future__ import print_function
import tensorflow as tf
import numpy as np

class Net(object):

	def __init__(self, sess):
		self.sess = sess

	def build(self):
		raise NotImplementedError


class Fcnn(Net):

	def __init__(self, sess, input_dim, output_dim, layer_dim, name = None, **kwargs):
		super(Fcnn, self).__init__(sess)
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.layer_dim = layer_dim
		self.name = name

		self.all_layer_dim = np.concatenate([[self.input_dim], layer_dim, [self.output_dim]], axis = 0)
		self.if_bias = kwargs.get('if_bias', ([True] * len(layer_dim))+[False] )
		self.activation_fns = kwargs.get( 'activation', (['tanh']*len(layer_dim))+['None'])
		with tf.name_scope(self.name):
			self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name = 'input'))

		if len(self.if_bias) == 1:
			self.if_bias *= len(layer_dim) + 1
		if len(self.activation_fns) == 1:
			self.activation_fns *= len(layer_dim) + 1

		act_funcs_dict = {'tanh':, tf.nn.tanh, 'relu':, tf.nn.relu, 'None':lambda x: x}
		self.activation_fns_call = [act_funcs_dict[_] for _ in self.activation_fns]

		assert(self.input.get_shape()[-1].value == input_dim)
		assert(len(self.activation_fns_call) == len(self.layer_dim) + 1)
		assert(len(self.if_bias) == len(self.layer_dim) + 1)

		self.output = self.build(self.input, self.name)

	def build(self, input_tf, name):
		
		with tf.name_scope(name):
			net = [input_tf]
			weights = []
			if np.any(self.if_bias):
				bias = []

			for i, (dim_1, dim_2) in enumerate( zip(self.all_layer_dim[:-1], self.all_layer_dim[1:]) ):
				if self.if_bias[i]:
					weights.append( tf.Variable( tf.truncated_normal([dim_1, dim_2], stddev = 1.0), name = 'theta_%i'%i))
					bias.append( tf.Variable (tf.truncated_normal([dim_2], stddev = 1.0), name = 'bias_%i'%i))
					net.append( self.activation_fns_call( tf.matmul(net[i], weights[i]) + bias[-1] ))
					
				else:
					weights.append( tf.Variable( tf.truncated_normal([dim_1, dim_2], stddev = 1.0), name = 'theta_%i'%i))
					bias.append( tf.Variable (tf.truncated_normal([dim_2], stddev = 1.0), name = 'bias_%i'%i))
					net.append( self.activation_fns_call( tf.matmul(net[i], weights[i]) ))

			return net[-1]



class Modular_Fcnn(Fcnn):

	def __init__(self, sess, input_dim, output_dim, layer_dim, module_num, name = None, **kwargs):
		super(Modular_Fcnn, self).__init__(sess, input_dim, output_dim, layer_dim, name, **kwargs)
		self.module_num = module_num

		with tf.name_scope(self.name):
			self.s_weights = kwargs.get('s_weights', [])
		assert(len(self.module_num) == len(self.layer_dim))

		if len(self.s_weights) == 0:
			self.s_weights = [tf.Variable(tf.ones([_], tf.float32), name = 's_weight_%i'%i) for _ in self.module_dim]

		self.output = self.build(self.input, self.s_weights, self.name)

	def build(self, input_tf, name):
		
		with tf.name_scope(name):
			net = [input_tf]
			weights = []
			if np.any(self.if_bias):
				bias = []

			for i, (dim_1, dim_2) in enumerate( zip(self.all_layer_dim[:-1], self.all_layer_dim[1:]) ):
				if self.if_bias[i]:
					weights.append()


# class Fcnn_2side(Net):

# 	def __init__(self, sess, input_dim1, input_dim2, output_dim, layer_dim_1, layer_dim_2, layer_dim_o = [], name = None, **kwargs):
# 		super(Fcnn_2side, self).__init__(sess)
# 		self.input_dim1 = input_dim1
# 		self.input_dim2 = input_dim2
# 		self.output_dim = output_dim
# 		self.layer_dim_1 = layer_dim_1
# 		self.layer_dim_2 = layer_dim_2
# 		self.layer_dim_o = layer_dim_o
# 		self.name = name

# 		self.merge_method = kwargs.get('merge_method', 'add')
# 		self.if_bias = kwargs.get('if_bias', True)
# 		# self.merge_method = kwargs['merge_method'] if 'merge_method' in kwargs.keys() else 'add'
# 		# self.if_bias = kwargs['if_bias'] if 'if_bias' in kwargs.keys() else True
# 		self.all_layer_dim_1 = np.concatenate([[self.input_dim1], layer_dim_1], axis = 0)
# 		self.all_layer_dim_2 = np.concatenate([[self.input_dim2], layer_dim_2], axis = 0)
# 		if self.merge_method is 'concatenate':
# 			self.all_layer_dim_o = np.concatenate([[self.layer_dim_1[-1] + self.layer_dim_2[-1]], layer_dim_o], axis = 0)
# 		else if self.merge_method in ['add', 'multiply']:
# 			assert(self.layer_dim_1[-1] == self.layer_dim_2[-1])
# 			self.all_layer_dim_o = np.concatenate([[self.layer_dim_1[-1]], layer_dim_o], axis = 0)

# 		self.input_1, self.input_2, self.output = self.build(name = self.name)

# 	def build(self, name):

# 		with tf.name_scope(name):
# 			self.input_1 = tf.placeholder(tf.float32, [None, self.input_dim1], name = 'input1')
# 			self.input_2 = tf.placeholder(tf.float32, [None, self.input_dim2], name = 'input2')
# 			net_1 = [self.input_1]
# 			net_2 = [self.input_2]
# 			weights_1 = []
# 			weights_2 = []
# 			if self.if_bias: 
# 				bias_1 = []
# 				bias_2 = []

# 			for i, (dim_1, dim_2) in enumerate( zip(self.all_layer_dim_1[:-1], self.all_layer_dim_1[1:]) ):
# 				if self.if_bias:
# 					weights_1.append(tf.Variable( tf.truncated_normal([dim_1, dim_2], stddev = 1.0), name = 'weights_1_layer_%i'%(i) ))
# 					bias_1.append( tf.Variable( tf.truncated_normal([dim_2], stddev = 1.), name = 'bias_1_layer_%i'%(i) ))
# 					net_1.append( tf.nn.tanh( tf.matmul(net_1[i], weights_1[i]) + bias_1[i]) )
# 				else:
# 					weights_1.append(tf.Variable( tf.truncated_normal([dim_1, dim_2], stddev = 1.0), name = 'weights_1_layer_%i'%(i) ))
# 					net_1.append( tf.nn.tanh( tf.matmul(net_1[i], weights_1[i]) ) )

# 			for i, (dim_1, dim_2) in enumerate( zip(self.all_layer_dim_2[:-1], self.all_layer_dim_2[1:]) ):
# 				if self.if_bias:
# 					weights_2.append( tf.Variable( tf.truncated_normal([dim_1, dim_2], stddev = 1.0), name = 'weights_2_layer_%i'%(i) ))
# 					bias_2.append( tf.Variable (tf.truncated_normal([dim_2], stddev = 1.), name = 'bias_2_layer_%i'%(i) ))
# 					net_2.append( tf.nn.tanh( tf.matmul(net_2[i], weights_2[i]) + bias_2[i]) )
# 				else:
# 					weights_2.append( tf.Variable( tf.truncated_normal([dim_1, dim_2], stddev = 1.0), name = 'weights_2_layer_%i'%(i) ))
# 					net_2.append( tf.nn.tanh( tf.matmul(net_2[i], weights_2[i]) ) )

# 			net_o = []
# 			weights_o = []
# 			if self.if_bias:
# 				bias_o = []
# 			if self.merge_method is 'concatenate':
# 				net_o.append( tf.concat([net_1[-1],net_2[-1]], axis = 1) )
# 			else if self.merge_method is 'add':
# 				net_o.append( net_1[-1] + net_2[-1] )
# 			else if self.merge_method is 'multiply':
# 				net_o.append( net_1[-1] * net_2[-1] )

# 			for i, (dim_1, dim_2) in enumerate( zip(self.all_layer_dim_o[:-1], self.all_layer_dim_o[1:]) ):
# 				if self.if_bias:
# 					weights_o.append()
