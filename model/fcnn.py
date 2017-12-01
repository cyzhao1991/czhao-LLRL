import tensorflow as tf
import numpy as np

class Fcnn(object):

	def __init__(self, sess, input_dim, output_dim, num_of_layers, dim_of_units, bias = True ,name = 'fcnn'):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.num_of_layers = num_of_layers
		self.dim_of_units = dim_of_units
		self.sess = sess

		self.dim_of_all = np.array(self.dim_of_units)
		self.dim_of_all = np.insert(self.dim_of_all, 0, input_dim)
		self.dim_of_all = np.append(self.dim_of_all, output_dim)

		self.input, self.output = self.build(name = name)


	def build(self, name = 'fcnn'):

		with tf.name_scope(name):
			inputs = {}
			inputs[0] = tf.placeholder(tf.float32, [None, self.input_dim], name = 'input')
			weights = {}
			bias = {}
			for i in range(self.num_of_layers):
				weights[i] = tf.Variable(tf.truncated_normal([self.dim_of_all[i], self.dim_of_all[i+1]], stddev = 1.0), name = 'weights'+str(i), trainable = True)
				bias[i] = tf.Variable(tf.truncated_normal([self.dim_of_all[i+1]], stddev = 0.), name = 'bias'+str(i), trainable = True)
				inputs[i+1] = tf.nn.relu( tf.matmul( inputs[i], weights[i]) + bias[i] )


			weights[self.num_of_layers] = tf.Variable(tf.truncated_normal([self.dim_of_all[self.num_of_layers], self.output_dim], stddev = 1.0), name = 'weights'+str(self.num_of_layers),trainable = True)

			output = tf.matmul( inputs[self.num_of_layers], weights[self.num_of_layers] ) 

		return inputs[0], output

	def predict(self, inputs, name = 'fcnn'):
		with tf.name_scope(name):
			return self.sess.run(self.output, feed_dict = {self.input: inputs})

	def place_holder(self):
		return self.input, self.output


	# [v for v in tf.global_variables() if 'weights1' in v.name]
	# sess.run(v.eval())
