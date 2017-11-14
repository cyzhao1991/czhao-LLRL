import tensorflow as tf
import numpy as np

class Fcnn_LS_MTL(object):

	def __init__(self, sess, input_dim, output_dim, num_of_layers, dim_of_units, num_of_latent, num_of_tasks ,bias = True ,name = 'fcnn'):#, weight_s = None):
		self.sess = sess
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.num_of_layers = num_of_layers
		self.dim_of_units = dim_of_units
		self.num_of_latent = num_of_latent
		self.num_of_tasks = num_of_tasks

		self.dim_of_all = np.array(self.dim_of_units)
		self.dim_of_all = np.insert(self.dim_of_all, 0, input_dim)
		self.dim_of_all = np.append(self.dim_of_all, output_dim)

		self.input, self.output = self.build(name = name)
		# self.weight_s = weight_s

	def build(self, name = 'fcnn'):

		with tf.name_scope(name):
			inputs ={}
			inputs[0] = tf.placeholder(tf.float32, [None, self.input_dim])
			weights = {}
			# if self.weight_s is None:
			weights_s = {}
			for t in range(self.num_of_tasks):
				weights_s[t] = tf.Variable( tf.truncated_normal( [self.num_of_layers + 1, self.num_of_latent], stddev = 1.0), name = 'weight_s_%i'%t, trainable = True)

			# else:
				# weights_s = self.weight_s
			bias = {}

			output_list = []
			for t in range(self.num_of_tasks):
				for i in range(self.num_of_layers):
					weights[i] = tf.Variable(tf.truncated_normal([ self.num_of_latent, self.dim_of_all[i], self.dim_of_all[i+1]], stddev = 1.0), name = 'weights'+str(i), trainable = True)
					bias[i] = tf.Variable(tf.truncated_normal([ self.num_of_latent, self.dim_of_all[i+1]], stddev = 0.), name = 'bias'+str(i), trainable = True)
					tmp_weight = tf.tensordot(weights[i], weights_s[t][i], axes = [[0],[0]])
					tmp_bias = tf.tensordot(bias[i], weights_s[t][i], axes = [[0],[0]])
					inputs[i+1] = tf.nn.relu( tf.matmul( inputs[i], tmp_weight) + tmp_bias )


				weights[self.num_of_layers] = tf.Variable(tf.truncated_normal([self.num_of_latent, self.dim_of_all[self.num_of_layers], self.output_dim], stddev = 1.0)\
					, name = 'weights'+str(self.num_of_layers),trainable = True)
				tmp_weight = tf.tensordot(weights[self.num_of_layers], weights_s[t][self.num_of_layers], axes = [[0],[0]])
				output = tf.matmul( inputs[self.num_of_layers], tmp_weight )
				output_list.append(output)
		
		return inputs[0], tf.stack( output_list, axis = 0 )

	def predict(self, inputs, name = 'fcnn'):
		with tf.name_scope(name):
			return self.sess.run(self.output, feed_dict = {self.input: inputs})

	def place_holder(self):
		return self.input, self.output

	# def replace_weight_s(self, weight_s):
	# 	self.weight_s = weight_s