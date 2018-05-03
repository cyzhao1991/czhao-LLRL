from __future__ import print_function
import tensorflow as tf
import numpy as np
from model.net import Net

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

				init_v = self.initialize_value[i] if self.initialize_value is not None else .1

				self.KB_weights.append( [tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev = init_v), \
					name = 'KB_weights_h%i_m%i'%(i,j)) for j in range(self.module_num[i])] )
				self.KB_bias.append( [tf.Variable(tf.truncated_normal([dim_2], stddev = init_v), \
					name = 'KB_bias_h%i_m%i'%(i,j)) if self.if_bias[i] else 0 for j in range(self.module_num[i]) ] )

	def def_Task_knowledge(self, name):
		# if self.context_dim <= self.module_num[0]:
		# 	diagnal_init = np.eye(self.context_dim)
		# 	diagnal_init = np.pad(diagnal_init, ((0,0),(0, self.module_num[0] - self.context_dim)), 'constant').astype(np.float32)
		# else:
		# 	raise ValueError('module_num too small')
		with tf.name_scope(name):
			self.c_weights = [tf.Variable(tf.truncated_normal([self.context_dim, dim], mean = 0., stddev = 1.), \
				name = 's_vector_h%i'%k) for dim,k in zip(self.module_num, range(self.num_of_layer))]
			# self.c_weights = [tf.Variable( (diagnal_init + .1 * np.random.rand(self.context_dim, dim)).astype(np.float32), \
			# 	name = 's_vector_h%i'%k) for dim,k in zip(self.module_num, range(self.num_of_layer))]
			# self.s_vector = [ tf.expand_dims(tf.matmul(self.context, tf.nn.relu(w_c)), 1) for w_c in self.c_weights]
			self.s_vector = [ tf.expand_dims(tf.matmul(self.context, w_c), 1) for w_c in self.c_weights]
			# self.s_vector = [ tf.expand_dims(tf.matmul(self.context, tf.sigmoid(w_c)), 1) for w_c in self.c_weights]

	def build(self, name):
		net = [self.input]
		with tf.name_scope(name):
			for i in range(self.num_of_layer):
				hidden_unit = tf.reduce_sum( tf.stack( [tf.matmul(net[i], weight) + bias for weight, bias in zip(self.KB_weights[i], self.KB_bias[i])] , \
					axis = -1) * self.s_vector[i], axis = -1)
				net.append(self.activation_fns_call[i](hidden_unit))
		return net[-1]

class Context_Fcnn_Net_2(Net):

	def __init__(self, sess, input_dim, output_dim, context_dim,layer_dim, module_num, name = None, **kwargs):
		super(Context_Fcnn_Net_2, self).__init__(sess, input_dim, output_dim, layer_dim, name, **kwargs)
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

				init_v = self.initialize_value[i] if self.initialize_value is not None else .1

				self.KB_weights.append( [tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev = init_v), \
					name = 'KB_weights_h%i_m%i'%(i,j)) for j in range(self.module_num[i])] )
				self.KB_bias.append( [tf.Variable(tf.truncated_normal([dim_2], stddev = init_v), \
					name = 'KB_bias_h%i_m%i'%(i,j)) if self.if_bias[i] else 0 for j in range(self.module_num[i]) ] )

	def def_Task_knowledge(self, name):
		# if self.context_dim <= self.module_num[0]:
		# 	diagnal_init = np.eye(self.context_dim)
		# 	diagnal_init = np.pad(diagnal_init, ((0,0),(0, self.module_num[0] - self.context_dim)), 'constant').astype(np.float32)
		# else:
		# 	raise ValueError('module_num too small')
		with tf.name_scope(name):
			self.c_weights = [tf.Variable(tf.truncated_normal([self.context_dim, dim], mean = 0., stddev = 1.), \
				name = 's_vector_h%i'%k) for dim,k in zip(self.module_num, range(self.num_of_layer))]
			# self.c_weights = [tf.Variable( (diagnal_init + .1 * np.random.rand(self.context_dim, dim)).astype(np.float32), \
			# 	name = 's_vector_h%i'%k) for dim,k in zip(self.module_num, range(self.num_of_layer))]
			self.s_vector = [ tf.expand_dims(tf.matmul(self.context, tf.nn.relu(w_c)), 1) for w_c in self.c_weights]
			# self.s_vector = [ tf.expand_dims(tf.matmul(self.context, w_c), 1) for w_c in self.c_weights]
			# self.s_vector = [ tf.expand_dims(tf.matmul(self.context, tf.sigmoid(w_c)), 1) for w_c in self.c_weights]

	def build(self, name):
		net = [self.input]
		with tf.name_scope(name):
			for i in range(self.num_of_layer):
				hidden_unit = tf.reduce_sum( tf.stack( [tf.matmul(net[i], weight) + bias for weight, bias in zip(self.KB_weights[i], self.KB_bias[i])] , \
					axis = -1) * self.s_vector[i], axis = -1)
				net.append(self.activation_fns_call[i](hidden_unit))
		return net[-1]

class one_hot_Fcnn_net(Net):

	def __init__(self, sess, input_dim, output_dim, task_num, layer_dim, module_num, name = None, **kwargs):
		super(one_hot_Fcnn_net, self).__init__(sess, input_dim, output_dim, layer_dim, name, **kwargs)
		self.module_num = module_num
		self.task_num = task_num
		self.num_of_hidden_layer = len(self.layer_dim)
		self.num_of_layer = self.num_of_hidden_layer + 1

		with tf.name_scope(self.name):
			self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name = 'input'))
			self.context = kwargs.get('context_tf', tf.placeholder(tf.float32, [None, self.context_dim], name ='context'))

		self.def_Shared_knowledge(self.name)
		self.def_Task_knowledge(self.name)
		self.output = self.build(self.name)

	# def def_Shared_knowledge(self, name):
	# 	self.KB_weights = []
	# 	self.KB_bias = []
	# 	with tf.name_scope(name):
	# 		for i in range( self.num_of_layer ):


# class Progressive_Context_Net(Net):
	
# 	def __init__(self, sess, input_dim, output_dim, context_dim, layer_dim, module_num, name = None, **kwargs):
# 		super(Progressive_Context_Net, self).__init__(sess, input_dim, output_dim, layer_dim, name, **kwargs)
# 		self.module_num = module_num
# 		self.context_dim = context_dim
# 		self.num_of_hidden_layer = len(self.layer_dim)
# 		self.num_of_layer = self.num_of_hidden_layer + 1

# 		assert(len(self.if_bias) == self.num_of_hidden_layer + 1)
# 		assert(len(self.activation_fns_call) == self.num_of_hidden_layer + 1)

# 		with tf.name_scope(self.name):
# 			self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name = 'input'))
# 			self.context = kwargs.get('context_tf', tf.placeholder(tf.float32, [None, self.context_dim], name ='context'))

# 		self.def_based_knowledge(self.name)
# 		self.def_Shared_knowledge(self.name)
# 		self.def_Task_knowledge(self.name)
# 		self.output = self.build(self.name)

# 	def def_based_knowledge(self, name):
# 		self.


class Concat_Context_Fcnn_Net(Net):

	def __init__(self, sess, input_dim, output_dim, context_dim, layer_dim, name = None, **kwargs):
		super(Concat_Context_Fcnn_Net, self).__init__(sess, input_dim, output_dim, layer_dim, name, **kwargs)
		# self.module_num = module_num
		self.context_dim = context_dim
		self.all_layer_dim[0] = self.input_dim + self.context_dim
		self.num_of_hidden_layer = len(self.layer_dim)
		self.num_of_layer = self.num_of_hidden_layer + 1

		assert(len(self.if_bias) == self.num_of_hidden_layer + 1)
		assert(len(self.activation_fns_call) == self.num_of_hidden_layer + 1)

		with tf.name_scope(self.name):
			self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name = 'input'))
			self.context = kwargs.get('context_tf', tf.placeholder(tf.float32, [None, self.context_dim], name ='context'))

		self.output = self.build(self.name)

	def build(self, name):
		with tf.name_scope(name):
			net = [tf.concat( [self.input, self.context], axis = 1) ]
			weights = []
			if np.any(self.if_bias):
				bias = []

			for i, (dim_1, dim_2) in enumerate( zip(self.all_layer_dim[:-1], self.all_layer_dim[1:]) ):
				if self.if_bias[i]:
					init_v = self.initialize_value[i] if self.initialize_value is not None else .1
					weights.append( tf.Variable( tf.truncated_normal([dim_1, dim_2], stddev = init_v), name = 'theta_%i'%i))
					bias.append( tf.Variable (tf.truncated_normal([dim_2], stddev = init_v), name = 'bias_%i'%i))
					net.append( self.activation_fns_call[i]( tf.matmul(net[i], weights[i]) + bias[-1] ))
					
				else:
					# print(dim_1,dim_2)
					init_v = self.initialize_value[i] if self.initialize_value is not None else .1
					weights.append( tf.Variable( tf.truncated_normal([dim_1, dim_2], stddev = init_v), name = 'theta_%i'%i))
					net.append( self.activation_fns_call[i]( tf.matmul(net[i], weights[i]) ))

			return net[-1]


# class Sparse_Fcnn_Net(Net):

# 	def __init__(self, sess, input_dim, output_dim, layer_dim, module_num, name = None, **kwargs):
# 		super(Context_Fcnn_Net, self).__init__(sess, input_dim, output_dim, layer_dim, name, **kwargs)
# 		self.module_num = module_num
# 		# self.context_dim = context_dim
# 		self.num_of_hidden_layer = len(self.layer_dim)
# 		self.num_of_layer = self.num_of_hidden_layer + 1

# 		assert(len(self.if_bias) == self.num_of_hidden_layer + 1)
# 		assert(len(self.activation_fns_call) == self.num_of_hidden_layer + 1)

# 		with tf.name_scope(self.name):
# 			self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name = 'input'))
# 			# self.context = kwargs.get('context_tf', tf.placeholder(tf.float32, [None, self.context_dim], name ='context'))

# 		self.def_Shared_knowledge(self.name)
# 		self.def_Task_knowledge(self.name)
# 		self.output = self.build(self.name)

# 	def def_Shared_knowledge(self, name):
# 		self.KB_weights = []
# 		self.KB_bias = []
# 		with tf.name_scope(name):
# 			for i in range( self.num_of_layer ):
# 				dim_1, dim_2 = self.all_layer_dim[i:i+2]

# 				init_v = self.initialize_value[i] if self.initialize_value is not None else .1

# 				self.KB_weights.append( [tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev = init_v), \
# 					name = 'KB_weights_h%i_m%i'%(i,j)) for j in range(self.module_num[i])] )
# 				self.KB_bias.append( [tf.Variable(tf.truncated_normal([dim_2], stddev = init_v), \
# 					name = 'KB_bias_h%i_m%i'%(i,j)) if self.if_bias[i] else 0 for j in range(self.module_num[i]) ] )

# 	def def_Task_knowledge(self, name):
# 		# if self.context_dim <= self.module_num[0]:
# 		# 	diagnal_init = np.eye(self.context_dim)
# 		# 	diagnal_init = np.pad(diagnal_init, ((0,0),(0, self.module_num[0] - self.context_dim)), 'constant').astype(np.float32)
# 		# else:
# 		# 	raise ValueError('module_num too small')
# 		with tf.name_scope(name):
# 			self.c_weights = [tf.Variable(tf.truncated_normal([self.context_dim, dim], mean = 0., stddev = 1.), \
# 				name = 's_vector_h%i'%k) for dim,k in zip(self.module_num, range(self.num_of_layer))]
# 			# self.c_weights = [tf.Variable( (diagnal_init + .1 * np.random.rand(self.context_dim, dim)).astype(np.float32), \
# 			# 	name = 's_vector_h%i'%k) for dim,k in zip(self.module_num, range(self.num_of_layer))]
# 			self.s_vector = [ tf.expand_dims(tf.matmul(self.context, tf.nn.relu(w_c)), 1) for w_c in self.c_weights]
# 			# self.s_vector = [ tf.expand_dims(tf.matmul(self.context, w_c), 1) for w_c in self.c_weights]
# 			# self.s_vector = [ tf.expand_dims(tf.matmul(self.context, tf.sigmoid(w_c)), 1) for w_c in self.c_weights]

# 	def build(self, name):
# 		net = [self.input]
# 		with tf.name_scope(name):
# 			for i in range(self.num_of_layer):
# 				hidden_unit = tf.reduce_sum( tf.stack( [tf.matmul(net[i], weight) + bias for weight, bias in zip(self.KB_weights[i], self.KB_bias[i])] , \
# 					axis = -1) * self.s_vector[i], axis = -1)
# 				net.append(self.activation_fns_call[i](hidden_unit))
# 		return net[-1]
