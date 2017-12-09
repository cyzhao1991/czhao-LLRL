import tensorflow as tf

class Paras_base(object):

	flags = tf.app.flags
	flags.DEFINE_integer('max_iter', 200, 'maximum training iteration')
	flags.DEFINE_integer('max_time_step', 500, 'maximum time step per episode')
	flags.DEFINE_integer('num_of_paths', 1000, 'number of paths in each ')
	flags.DEFINE_integer('obs_shape', 4, 'dimension of observation')
	flags.DEFINE_integer('action_shape', 1, 'dimension of action')
	flags.DEFINE_integer('cg_iters', 30, 'max_iteration for conjugate gradient')
	flags.DEFINE_integer('max_backtracks',10, 'maximum number for backward linesearch')
	flags.DEFINE_integer('save_model_iters',10, 'save model per # of iterations')

	flags.DEFINE_float('discount', 0.99, 'discount factor for env')
	flags.DEFINE_float('max_std', 2.6, 'maximum std for action distribution')
	flags.DEFINE_float('min_std', 0.1, 'minimum std for action distribution')
	flags.DEFINE_float('subsample_factor', 1.0, 'subsample factor while training')
	flags.DEFINE_float('cg_damping', 0.001 ,'damping factor for conjugate gradient')
	flags.DEFINE_float('max_kl', .1, 'maximum kl divergence per update')
	flags.DEFINE_float('max_action', 1.0, 'maximum action')

	flags.DEFINE_boolean('render', False, 'whether to render')
	flags.DEFINE_boolean('train_flag', True, 'whether in training process')
	flags.DEFINE_boolean('center_adv', True, 'whether normalize advantages')
	flags.DEFINE_boolean('linesearch', True, 'if linesearch')
	flags.DEFINE_boolean('save_model', True, 'if save model')

	flags.DEFINE_string('name_scope', 'trpo', 'name scope for agent')
	flags.DEFINE_string('env_name', 'cartpole', 'name of environment')
	flags.DEFINE_string('save_dir', './Data/checkpoint/', 'checkpoint saving directory')

	pms = flags.FLAGS
	