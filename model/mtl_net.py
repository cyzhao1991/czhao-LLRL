from __future__ import print_function
import tensorflow as tf
import numpy as np
from net import Net

# class Mtl_Net(object):

# 	def __init__(self, sess, input_dim, output_dim, layer_dim, name = None, **kwargs):

# 		self.sess = sess
# 		self.input_dim = input_dim
# 		self.output_dim = output_dim
# 		self.layer_dim = layer_dim
# 		self.name = name

# 		self.all_layer_dim = np.concatenate([[self.input_dim], layer_dim, [self.output_dim]], axis = 0).astype(int)
# 		self.if_bias = kwargs.get('if_bias', ([True] * len(layer_dim))+[False] )
# 		self.activation_fns = kwargs.get( 'activation', (['tanh']*len(layer_dim))+['None'])
# 		if len(self.if_bias) == 1:
# 			self.if_bias *= len(layer_dim) + 1
# 		if len(self.activation_fns) == 1:
# 			self.activation_fns *= len(layer_dim) + 1

# 		act_funcs_dict = {'tanh': tf.nn.tanh, 'relu': tf.nn.relu, 'None':lambda x: x}
# 		self.activation_fns_call = [act_funcs_dict[_] for _ in self.activation_fns]

# 	def build(self):
# 		raise NotImplementedError

class Mtl_Fcnn_Net(Net):

	def __init__(self, sess, input_dim, output_dim, layer_dim, module_num, num_of_tasks, name = None, **kwargs):
		super(Mtl_Fcnn_Net, self).__init__(sess, input_dim, output_dim, layer_dim, name, **kwargs)
		self.module_num = module_num
		self.num_of_hidden_layer = len(self.layer_dim)
		self.num_of_tasks = num_of_tasks

		self.def_Shared_knowledge(self.name)
		self.def_Task_knowledge(self.name, self.num_of_tasks)

		with tf.name_scope(self.name):
			self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name = 'input'))

		assert(len(self.if_bias) == self.num_of_hidden_layer + 1)
		assert(len(self.activation_fns_call) == self.num_of_hidden_layer + 1)

		self.input, self.output = self.build( self.input, self.name )

	def def_Shared_knowledge(self, name):
		# self.KB_weights = [None] * self.num_of_hidden_layer
		# self.KB_bias = [None] * self.num_of_hidden_layer
		self.KB_weights = []
		self.KB_bias = []
		with tf.name_scope(name):
			for i in range( self.num_of_hidden_layer ):
				dim_1, dim_2 = self.all_layer_dim[i:i+2]

				self.KB_weights.append( [tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev = 1.0), \
					name = 'KB_weights_h%i_m%i'%(i,j)) for j in range(self.module_num[i])] )
				self.KB_bias.append( [tf.Variable(tf.truncated_normal([dim_2], stddev = 1.0), \
					name = 'KB_bias_h%i_m%i'%(i,j)) if self.if_bias[i] else None for j in range(self.module_num[i]) ] )
				# self.KB_weights.append( [] )
				# self.KB_bias.append( [] if self.if_bias[i] else None )
				# for j in range(self.module_num[i]):
				# 	self.KB_weights[i].append( tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev = 1.0), name = 'KB_weights_i'%))

	def def_Task_knowledge(self, name, num_of_tasks ):
		self.s_vector = []
		self.output_layer_weights = []
		self.output_layer_bias = []
		seed = np.random.randint(1000, 4000)
		with tf.name_scope(name):
			for i in range(num_of_tasks):
				self.s_vector.append( [ tf.Variable(tf.truncated_normal( [self.module_num[j]], stddev = 0.1, seed = seed), \
					name = 's_vector_t%i_h%i'%(i,j) ) for j in range(self.num_of_hidden_layer)] )
				self.output_layer_weights.append( tf.Variable( tf.truncated_normal( [self.layer_dim[-1], self.output_dim ], stddev = 1.0), \
					name = 'output_layer_weights_t%i'%i) )
				self.output_layer_bias.append( tf.Variable (tf.truncated_normal( [self.output_dim], stddev = 1.0), \
					name = 'output_layer_bias_t%i'%i) if self.if_bias[-1] else None)

	def build(self, input_tf, name):

		net = []
		with tf.name_scope(name):
			for i in range(self.num_of_tasks):
				net.append( [input_tf] )
				for j in range(self.num_of_hidden_layer):
					weights = tf.reduce_sum( tf.stack(self.KB_weights[j], axis = -1) * self.s_vector[i][j], axis = -1)
					bias = tf.reduce_sum(tf.stack(self.KB_bias[j], axis = -1) * self.s_vector[i][j], axis = -1) if self.if_bias[j] else 0
					hidden_unit = tf.matmul( net[i][j], weights ) + bias 

					net[i].append( self.activation_fns_call[j]( hidden_unit ) )
				
				output_task = tf.matmul( net[i][-1], self.output_layer_weights[i] )
				output_task += bias if self.if_bias[-1] else 0
				net[i].append( self.activation_fns_call[-1]( output_task) )

		output_list = [ n[-1] for n in net]
		return input_tf, output_list

