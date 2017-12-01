import tensorflow as tf

class Paras_base(object):

	flags = tf.app.flags
	flags.DEFINE_integer('max_iter', 200, 'maximum training iteration')
	flags.DEFINE_integer('max_time_step', 500, 'maximum time step per episode')
	flags.DEFINE_integer('num_of_paths', 1000, 'number of paths in each ')

	flags.DEFINE_boolean('render', Flase, 'whether to render')
	