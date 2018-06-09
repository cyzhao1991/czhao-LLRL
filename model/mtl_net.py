from __future__ import print_function
import tensorflow as tf
import numpy as np
from net import Net

# class Mtl_Fcnn_Net(Net):

# 	def __init__(self, sess, input_dim, output_dim, layer_dim, module_num, num_of_tasks, name = None, **kwargs):
# 		super(Mtl_Fcnn_Net, self).__init__(sess, input_dim, output_dim, layer_dim, name, **kwargs)
# 		self.module_num = module_num
# 		self.num_of_hidden_layer = len(self.layer_dim)
# 		self.num_of_tasks = num_of_tasks

# 		self.def_Shared_knowledge(self.name)
# 		self.def_Task_knowledge(self.name, self.num_of_tasks)

# 		with tf.name_scope(self.name):
# 			self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name = 'input'))

# 		assert(len(self.if_bias) == self.num_of_hidden_layer + 1)
# 		assert(len(self.activation_fns_call) == self.num_of_hidden_layer + 1)

# 		self.input, self.output = self.build( self.input, self.name )

# 	def def_Shared_knowledge(self, name):
# 		# self.KB_weights = [None] * self.num_of_hidden_layer
# 		# self.KB_bias = [None] * self.num_of_hidden_layer
# 		self.KB_weights = []
# 		self.KB_bias = []
# 		with tf.name_scope(name):
# 			for i in range( self.num_of_hidden_layer ):
# 				dim_1, dim_2 = self.all_layer_dim[i:i+2]

# 				self.KB_weights.append( [tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev = 1.0), \
# 					name = 'KB_weights_h%i_m%i'%(i,j)) for j in range(self.module_num[i])] )
# 				self.KB_bias.append( [tf.Variable(tf.truncated_normal([dim_2], stddev = 1.0), \
# 					name = 'KB_bias_h%i_m%i'%(i,j)) if self.if_bias[i] else None for j in range(self.module_num[i]) ] )
# 				# self.KB_weights.append( [] )
# 				# self.KB_bias.append( [] if self.if_bias[i] else None )
# 				# for j in range(self.module_num[i]):
# 				# 	self.KB_weights[i].append( tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev = 1.0), name = 'KB_weights_i'%))

# 	def def_Task_knowledge(self, name, num_of_tasks ):
# 		self.s_vector = []
# 		self.output_layer_weights = []
# 		self.output_layer_bias = []
# 		seed = np.random.randint(1000, 4000)
# 		with tf.name_scope(name):
# 			for i in range(num_of_tasks):
# 				self.s_vector.append( [ tf.Variable(tf.truncated_normal( [self.module_num[j]], stddev = .1, seed = seed), \
# 					name = 's_vector_t%i_h%i'%(i,j) ) for j in range(self.num_of_hidden_layer)] )
# 				self.output_layer_weights.append( tf.Variable( tf.truncated_normal( [self.layer_dim[-1], self.output_dim ], stddev = 1.0), \
# 					name = 'output_layer_weights_t%i'%i) )
# 				self.output_layer_bias.append( tf.Variable (tf.truncated_normal( [self.output_dim], stddev = 1.0), \
# 					name = 'output_layer_bias_t%i'%i) if self.if_bias[-1] else None)

# 	def build(self, input_tf, name):

# 		net = []
# 		with tf.name_scope(name):
# 			for i in range(self.num_of_tasks):
# 				net.append( [input_tf] )
# 				for j in range(self.num_of_hidden_layer):
# 					weights = tf.reduce_sum( tf.stack(self.KB_weights[j], axis = -1) * self.s_vector[i][j], axis = -1)
# 					bias = tf.reduce_sum(tf.stack(self.KB_bias[j], axis = -1) * self.s_vector[i][j], axis = -1) if self.if_bias[j] else 0
# 					hidden_unit = tf.matmul( net[i][j], weights ) + bias 

# 					net[i].append( self.activation_fns_call[j]( hidden_unit ) )
				
# 				output_task = tf.matmul( net[i][-1], self.output_layer_weights[i] )
# 				output_task += bias if self.if_bias[-1] else 0
# 				net[i].append( self.activation_fns_call[-1]( output_task) )

# 		output_list = [ n[-1] for n in net]
# 		return input_tf, output_list


class MtlFcnnNet(Net):

	def __init__(self, sess, input_dim, output_dim, layer_dim, module_num, num_of_tasks, name = None, **kwargs):
		super(MtlFcnnNet, self).__init__(sess, input_dim, output_dim, layer_dim, name, **kwargs)
		self.module_num = module_num
		self.num_of_hidden_layer = len(self.layer_dim)
		self.num_of_tasks = num_of_tasks

		self.def_shared_knowledge(self.name)
		self.def_task_knowledge(self.name, self.num_of_tasks)
		self.def_task_path(self.name, self.num_of_tasks)

		with tf.name_scope(self.name):
			self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name = 'input'))

		assert(len(self.if_bias) == self.num_of_hidden_layer + 1)
		assert(len(self.activation_fns_call) == self.num_of_hidden_layer + 1)

		self.output, self.all_net = self.build( self.name )

	def def_shared_knowledge(self, name):

		self.shared_weights = [None] * self.num_of_layer
		self.shared_bias = [None] * self.num_of_layer

		with tf.name_scope(name):
			for i in range(self.num_of_layer):
				dim_1, dim_2 = self.all_layer_dim[i:i+2]
				init_v = self.initialize_value[i] if self.initialize_value is not None else .1

				self.shared_weights[i] = [tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev = init_v), \
					name = 'shared_ws_h%i_m%i'%(i,j)) for j in range(self.module_num[i])]
				self.shared_bias[i] = [tf.Variable(tf.truncated_normal([dim_2], stddev = init_v), \
					name = 'shared_bs_h%i_m%i'%(i,j)) if self.if_bias[i] else 0 for j in range(self.module_num[i]) ]

	def def_task_knowledge(self, name, num_of_tasks):

		self.task_weights = [None] * self.num_of_layer
		self.task_bias = [None] * self.num_of_layer
		with tf.name_scope(name):
			for i in range(self.num_of_layer):
				dim_1, dim_2 = self.all_layer_dim[i:i+2]
				init_v = self.initialize_value[i] if self.initialize_value is not None else .1

				self.task_weights[i] = [tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev = init_v), \
					name = 'task_ws_h%i_t%i'%(i,j)) for j in range(num_of_tasks)]
				self.task_bias[i] = [tf.Variable(tf.truncated_normal([dim_2], stddev = init_v), \
					name = 'task_bs_h%i_t%i'%(i,j)) if self.if_bias[i] else 0 for j in range(num_of_tasks) ]

	def def_task_path(self, name, num_of_tasks):

		self.task_path = [None] * self.num_of_layer
		with tf.name_scope(name):
			for i in range(self.num_of_layer):
				self.task_path[i] = [tf.Variable( tf.truncated_normal([self.module_num[i]+1], stddev = 1.), \
					name = 'task_path_h%i_t%i'%(i,j)) for j in range(num_of_tasks)]


	def build(self, name):
		
		net = [None] * self.num_of_tasks
		with tf.name_scope(name):
			for j in range(self.num_of_tasks):
				net[j] = [None] * self.num_of_layer
				net[j][0] = self.input
				for i in range(self.num_of_layer):
					tmp_in = net[i][j]
					shared_h = [tf.activation_fns_call[i](tmp_in @ w + b) for w, b in zip(self.shared_weights[i], self.shared_bias[i])]
					task_h = tf.activation_fns_call[i](tmp_in @ self.task_weights[i][j] + self.task_bias[i][j])
					net[j][i] = tf.reduce_mean( tf.stack(shared_h + [task_h], axis = 2) * tf.reshape(self.task_path[i][j], [1,1,-1]), axis = -1)
		all_output = [n[-1] for n in net]
		return all_output, net


