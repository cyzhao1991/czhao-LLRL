from __future__ import print_function
import tensorflow as tf
import numpy as np

class FcnnMTL(object):

	def __init__(self, sess, input_dim, output_dim, num_of_layers, dim_of_units, num_of_latent, bias = True ,name = 'fcnn'):#, weight_s = None):
		self.sess = sess
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.num_of_layers = num_of_layers
		self.dim_of_units = dim_of_units
		self.num_of_latent = num_of_latent

		self.dim_of_all = np.array(self.dim_of_units)
		self.dim_of_all = np.insert(self.dim_of_all, 0, input_dim)
		self.dim_of_all = np.append(self.dim_of_all, output_dim)

		self.input, self.output = self.build(name = name)
		# self.weight_s = weight_s

	def build(self, name = 'fcnn'):

		with tf.name_scope(name):
			inputs = [None for _ in range(self.num_of_layers+1)]
			inputs[0] = tf.placeholder(tf.float32, [None, self.input_dim])
			weights_L = [[None for _ in range(self.num_of_latent)] for _ in range(self.num_of_layers+1)]
			# if self.weight_s is None:
			# weights_s = tf.Variable( tf.truncated_normal( [self.num_of_layers + 1, self.num_of_latent], stddev = 1.0), name = 'weight_s', trainable = True) 
			# else:
				# weights_s = self.weight_s
			weights_s = [None for _ in range(self.num_of_layers+1)]
			bias_L = [[None for _ in range(self.num_of_latent)] for _ in range(self.num_of_layers+1)]
			summed_weights = [None for _ in range(self.num_of_layers+1)]
			summed_bias = [None for _ in range(self.num_of_layers+1)]

			for i in range(self.num_of_layers + 1):
				for j in range(self.num_of_latent):
					weights_L[i][j] = tf.Variable(tf.truncated_normal( [self.dim_of_all[i], self.dim_of_all[i+1]], stddev = 1.0, dtype = tf.float32), name = 'weights_L_layer%i_latent%i'%(i,j), trainable = True)
					# print(self.sess.run(tf.shape(weights_L[0][0])), i, j)
					bias_L[i][j] = tf.Variable(tf.truncated_normal( [self.dim_of_all[i+1]], stddev = 1., dtype = tf.float32), name = 'weights_bias_layer%i_latent%i'%(i,j), trainable = True)
				weights_s[i] = tf.Variable(np.zeros(self.num_of_latent), name = 'weights_s_layer%i'%(i), trainable = True, dtype = tf.float32)

			for i in range(self.num_of_layers):
				summed_weights[i] = tf.add_n([w_L * weights_s[i][j] for w_L, j in zip(weights_L[i], range(self.num_of_latent))])
				summed_bias[i]    = tf.add_n([w_L * weights_s[i][j] for w_L, j in zip(bias_L[i], range(self.num_of_latent))])
				inputs[i+1] = tf.nn.relu( tf.matmul( inputs[i], summed_weights[i] ) + summed_bias[i] )

			summed_weights[-1] = tf.add_n([w_L * weights_s[-1][j] for w_L, j in zip(weights_L[-1], range(self.num_of_latent))])
			output = tf.matmul(inputs[self.num_of_layers], summed_weights[-1])
			# for i in range(self.num_of_layers):
			# 	weights[i] = tf.Variable(tf.truncated_normal([ self.num_of_latent, self.dim_of_all[i], self.dim_of_all[i+1]], stddev = 1.0), name = 'weights'+str(i), trainable = True)
			# 	bias[i] = tf.Variable(tf.truncated_normal([ self.num_of_latent, self.dim_of_all[i+1]], stddev = 0.), name = 'bias'+str(i), trainable = True)
			# 	tmp_weight = tf.tensordot(weights[i], weights_s[i], axes = [[0],[0]])
			# 	tmp_bias = tf.tensordot(bias[i], weights_s[i], axes = [[0],[0]])
			# 	inputs[i+1] = tf.nn.relu( tf.matmul( inputs[i], tmp_weight) + tmp_bias )


			# weights[self.num_of_layers] = tf.Variable(tf.truncated_normal([self.num_of_latent, self.dim_of_all[self.num_of_layers], self.output_dim], stddev = 1.0)\
			# 	, name = 'weights'+str(self.num_of_layers),trainable = True)
			# tmp_weight = tf.tensordot(weights[self.num_of_layers], weights_s[self.num_of_layers], axes = [[0],[0]])
			# output = tf.matmul( inputs[self.num_of_layers], tmp_weight ) 
		
		return inputs[0], output

	def predict(self, inputs, name = 'fcnn'):
		with tf.name_scope(name):
			return self.sess.run(self.output, feed_dict = {self.input: inputs})

	def place_holder(self):
		return self.input, self.output

	# def replace_weight_s(self, weight_s):
	# 	self.weight_s = weight_s