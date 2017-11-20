from __future__ import print_function
import numpy as np
import tensorflow as tf

class KnowledgeBase(object):

	def __init__(self, layer_shape, num_of_latent, layer_type = 'fcnn', name = 'KB', seed = None):

		self.layer_shape = layer_shape
		self.num_of_latent = num_of_latent
		self.layer_type = layer_type
		self.name = name

		self.num_of_layer = len(layer_shape) - 1

		self.L_weights = []
		self.L_bias = []
		self.build(name = name, seed = seed)
		self.build(name = name+'_target', seed = seed)

	def build(self, name, seed = None):

		with tf.name_scope(name):
			for i in range(self.num_of_layer):
				self.L_weights.append([])
				self.L_bias.append([])
				n_in, n_out = self.layer_shape[i:i+2]
				for j in range(self.num_of_latent):
					self.L_weights[i].append( tf.Variable(tf.truncated_normal([n_in, n_out], stddev = 1.0, seed = seed), name = 'L_weights_layer%i_latent%i'%(i, j), trainable = True) )
					self.L_bias[i].append( tf.Variable(tf.truncated_normal([n_out], stddev = 1.0, seed = seed), name = 'L_bias_layer%i_latent%i'%(i,j), trainable = True) )


	def variable_list(self):
		return [v for v in tf.trainable_variables() if self.name in v.name and 'target' not in v.name]

	def target_variable_list(self):
		return [v for v in tf.trainable_variables() if self.name in v.name and 'target' in v.name]

class PathPolicy(object):

	def __init__(self, input_dim, output_dim, knowledge_base = None, layer_shape = None, num_of_latent = None, seed = None, \
		layer_type = 'fcnn', name = 'task_0', allow_context = False, context_dim = None, context_layer = 0, pre_defined_context = None, final_layer_act_function = None):

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.name = name
		self.final_layer_act_function = final_layer_act_function
		if knowledge_base is None:
			self.KB = KnowledgeBase(layer_shape, num_of_latent, layer_type, name + 'KB')
			self.layer_shape = layer_shape
			self.num_of_latent = num_of_latent
			self.num_of_layer = len(layer_shape) - 1
			self.layer_type = layer_type
		else:
			self.KB = knowledge_base
			self.layer_shape = self.KB.layer_shape
			self.num_of_latent = self.KB.num_of_latent
			self.num_of_layer = self.KB.num_of_layer
			self.layer_type = self.KB.layer_type

		self.s_weights = []
		self.s_bias = []
		self.hidden_units = [None] * (self.num_of_layer + 1)
		self.weights_task = [None] * self.num_of_layer
		self.bias_task = [None] * self.num_of_layer
		self.input_weights = None
		self.input_bias = None
		self.output_weights = None

		self.allow_context = allow_context
		if allow_context:
			self.context_dim = context_dim
			self.context_layer = context_layer
			self.context_weights = None
			self.context_bias = None
			self.input, self.output, self.context = self.build(name = name, pre_defined_context = pre_defined_context, seed = seed)
			self.target_input, self.target_output, self.target_context = self.build(name = name + '_target', pre_defined_context = pre_defined_context, seed = seed)
		else:
			self.input, self.output = self.build(name = name, pre_defined_context = pre_defined_context, seed = seed)
			self.target_input, self.target_output = self.build(name = name + '_target', pre_defined_context = pre_defined_context, seed = seed)

	def build(self, name, pre_defined_context = None, seed = None):
		
		with tf.name_scope(name):
			inputs = tf.placeholder(tf.float32, [None, self.input_dim], name = 'input')
			self.input_weights = tf.Variable( tf.truncated_normal([self.input_dim, self.layer_shape[0]], stddev = 1.0, seed = seed), name = 'weights_input', trainable = True)
			self.input_bias = tf.Variable( tf.truncated_normal([self.layer_shape[0]], stddev = 1.0, seed = seed), name = 'bias_input', trainable = True)
			self.hidden_units[0] = tf.nn.relu( tf.matmul(inputs, self.input_weights) + self.input_bias )
			if self.allow_context:
				if pre_defined_context is not None:
					contexts = pre_defined_context
				else:
					contexts = tf.placeholder(tf.float32, [None, self.context_dim], name = 'context')
				self.context_weights = tf.Variable( tf.truncated_normal([self.context_dim, self.layer_shape[self.context_layer]], stddev = 1.0, seed = seed), name = 'weights_context',  trainable = True)
				self.context_bias = tf.Variable( tf.truncated_normal([self.layer_shape[self.context_layer]], stddev = 1.0, seed = seed), name = 'bias_context', trainable = True)
				context_out = tf.nn.relu( tf.matmul(contexts, self.context_weights) + self.context_bias)
			for i in range(self.num_of_layer):
				self.s_weights.append( tf.Variable( tf.truncated_normal([self.num_of_latent], stddev = 1.0, seed = seed), name = 's_weights_layer%i'%(i), trainable = True) )
				self.s_bias.append( tf.Variable( tf.truncated_normal([self.num_of_latent], stddev = 1.0, seed = seed), name = 's_bias_layer%i'%(i), trainable = True) )
				self.weights_task[i] = tf.add_n([self.s_weights[i][j] * self.KB.L_weights[i][j] for j in range(self.num_of_latent)])
				self.bias_task[i] = tf.add_n( [self.s_bias[i][j] * self.KB.L_bias[i][j] for j in range(self.num_of_latent)])
				if i == self.context_layer:
					self.hidden_units[i+1] = tf.nn.relu( tf.matmul( (self.hidden_units[i]+context_out), self.weights_task[i]) + self.bias_task[i])
				else:
					self.hidden_units[i+1] = tf.nn.relu( tf.matmul(self.hidden_units[i], self.weights_task[i]) + self.bias_task[i])

			self.output_weights = tf.Variable( tf.truncated_normal([self.layer_shape[-1], self.output_dim], stddev = 1.0, seed = seed), name = 'weights_output', trainable = True)				
			outputs = tf.matmul( self.hidden_units[-1], self.output_weights )
			if self.final_layer_act_function == 'tanh':
				outputs = tf.nn.tanh(outputs) * 10.
			if self.allow_context:
				return inputs, outputs, contexts
			else:
				return inputs, outputs

	def variable_list(self):
		return [v for v in tf.trainable_variables() if self.name in v.name and 'KB' not in v.name and 'target' not in v.name]

	def target_variable_list(self):
		return [v for v in tf.trainable_variables() if self.name in v.name and 'KB' not in v.name and 'target' in v.name]


# class LearnerWarpper(object):

# 	def __init__(self, knowledge_base, path_policy_list, learning_rate_kb = 0.01, learning_rate_s = 0.001, tau = 0.001, name = ''):
# 		self.KB = knowledge_base
# 		self.pp_list = path_policy_list

# 		self.num_of_tasks = len(path_policy_list)

# 		self.action_list = [None] * self.num_of_tasks
# 		self.loss_list = [None] * self.num_of_tasks

# 		with tf.name_scope(name):
# 			self.optimizer_KB = tf.train.AdamOptimizer(learning_rate = learning_rate_kb, name = name + '_kb_adam')
# 			self.optimizer_s = tf.train.AdamOptimizer(learning_rate = learning_rate_s, name = name + '_s_adam')

