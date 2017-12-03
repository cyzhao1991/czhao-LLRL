import tensorflow as tf

class Paras_base(object):

	flags = tf.app.flags
	flags.DEFINE_integer('max_iter', 200, 'maximum training iteration')
	flags.DEFINE_integer('max_time_step', 500, 'maximum time step per episode')
	flags.DEFINE_integer('num_of_paths', 1000, 'number of paths in each ')

	flags.DEFINE_float('discount', 0.99, 'discount factor for env')
	flags.DEFINE_float('max_std', 2.6, 'maximum std for action distribution')
	flags.DEFINE_float('min_std', 0.1, 'minimum std for action distribution')

	flags.DEFINE_boolean('render', False, 'whether to render')
	flags.DEFINE_boolean('train_flag', True, 'whether in training process')
	flags.DEFINE_boolean('center_adv', True, 'whether normalize advantages')

	pms = flags.FLAGS
	