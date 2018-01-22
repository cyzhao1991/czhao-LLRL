import tensorflow as tf
import numpy as np
import scipy.signal


def discount(x, gamma):

	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis = 0)[::-1]

def log_likelihood(x, means, logstds):
	zs = (x - means)/tf.exp(logstds)
	return -tf.reduce_sum(logstds, -1) - .5 *tf.reduce_sum(tf.square(zs), -1) - .5*means.get_shape()[-1].value * np.log(2*np.pi)

def flatten_var(var_list):
	return tf.concat([tf.reshape(var, [tf.size(var)]) for var in var_list], axis = 0)

def set_from_flat(var_list, x):
	start = 0
	assigns = []
	for var in var_list:
		shape = var.get_shape().as_list()
		size = np.prod(shape)		
		assigns.append(tf.assign(var, tf.reshape(x[start:start + size], shape)))
		start += size
	return assigns

def kl_sym(mean_1, logstd_1, mean_2, logstd_2):
	std_1 = tf.exp(logstd_1)
	std_2 = tf.exp(logstd_2)
	numerator = tf.square(mean_1 - mean_2) + tf.square(std_1) - tf.square(std_2)
	denominator = 2 * tf.square(std_2) + 1e-8
	kl = tf.reduce_sum(numerator/denominator + logstd_2 - logstd_1, -1)
	return kl

def kl_sym_firstfixed(mean, logstd):
	m_1, ls_1 = map(tf.stop_gradient, [mean, logstd])
	m_2, ls_2 = mean, logstd
	return kl_sym(m_1, ls_1, m_2, ls_2)

def linesearch(f, x, fullstep, max_backtracks, max_kl):
	fval, kl = f(x)
	for step_frac in .3**np.arange(max_backtracks):
		new_x = x + step_frac*fullstep
		newfval, newkl = f(new_x)
		if newfval < fval and newkl < max_kl:
			# print('valid gradient')
			return new_x
	return x
