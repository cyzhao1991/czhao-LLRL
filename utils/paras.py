import tensorflow as tf

class Paras_base(object):

	flags = tf.app.flags
	flags.DEFINE_integer('max_iter', 200, 'maximum training iteration')
	flags.DEFINE_integer('max_time_step', 200, 'maximum time step per episode')
	flags.DEFINE_integer('num_of_paths', 100, 'number of paths in each ')
	flags.DEFINE_integer('obs_shape', 4, 'dimension of observation')
	flags.DEFINE_integer('action_shape', 1, 'dimension of action')
	flags.DEFINE_integer('cg_iters', 30, 'max_iteration for conjugate gradient')
	flags.DEFINE_integer('max_backtracks',10, 'maximum number for backward linesearch')
	flags.DEFINE_integer('save_model_iters',10, 'save model per # of iterations')
	flags.DEFINE_integer('max_total_time_step', 0, 'max time step per iteration')
	flags.DEFINE_integer('pre_iter', 0, 'pre iter for saving model')

	flags.DEFINE_float('discount', 0.99, 'discount factor for env')
	flags.DEFINE_float('max_std', 2.6, 'maximum std for action distribution')
	flags.DEFINE_float('min_std', 0.01, 'minimum std for action distribution')
	flags.DEFINE_float('subsample_factor', 1.0, 'subsample factor while training')
	flags.DEFINE_float('cg_damping', 0.1 ,'damping factor for conjugate gradient')
	flags.DEFINE_float('max_kl', .01, 'maximum kl divergence per update')
	flags.DEFINE_float('max_action', 10.0, 'maximum action')
	flags.DEFINE_float('gae_lambda', 0.97, 'lambda factor for GAE')
	flags.DEFINE_float('l1_regularizer', .001, 'alpha for l1 regularizer')
	flags.DEFINE_float('l1_column_reg', 0.005, 'alpha for l1 column regularizer')

	flags.DEFINE_boolean('render', False, 'whether to render')
	flags.DEFINE_boolean('train_flag', True, 'whether in training process')
	flags.DEFINE_boolean('center_adv', True, 'whether normalize advantages')
	flags.DEFINE_boolean('linesearch', True, 'if linesearch')
	flags.DEFINE_boolean('save_model', True, 'if save model')
	flags.DEFINE_boolean('gae_flag', True, 'if using GAE')
	flags.DEFINE_boolean('with_context', False, 'if with context input')
	flags.DEFINE_boolean('independent_std', True, 'if with context input')


	flags.DEFINE_string('name_scope', 'trpo', 'name scope for agent')
	flags.DEFINE_string('env_name', 'cartpole', 'name of environment')
	flags.DEFINE_string('save_dir', './Data/checkpoint/', 'checkpoint saving directory')
	flags.DEFINE_string('contextual_method', 'meta_s_network', 'how to encode contexts')

	pms = flags.FLAGS
	

	'''
		DDPG Paras
	'''
	flags.DEFINE_integer('buffer_size', 100000, 'maximum size of replay buffer')
	flags.DEFINE_integer('batchsize', 64, 'size of minibatch')
	flags.DEFINE_integer('warm_up_size', 500, 'initial steps to start with')
	flags.DEFINE_float('actor_learning_rate', 0.001, 'actor learning rate')
	flags.DEFINE_float('critic_learning_rate', 0.0001, 'critic learning rate')
	flags.DEFINE_float('tau', 0.001, 'soft target updates tau')

	'''
	PPO Paras
	'''
	flags.DEFINE_float('cliprange', 0.2, 'cliprange epsilon')
	flags.DEFINE_float('entropy_coeff', .01, 'entropy co-efficient c2')
	flags.DEFINE_float('vf_coeff', 1., 'value loss co-efficient c1')
	flags.DEFINE_float('ppo_lr', 3e-4, 'learning rate for ppo')

	flags.DEFINE_integer('nbatch', 32, 'number of batches per iteration')
	flags.DEFINE_integer('nepochs', 10, 'number of updating epochs')

	'''
	others
	'''
	flags.DEFINE_integer('context_shape', 3, 'dimension of context vector')
# class DDPG_Paras_base(object):
# 	flags = tf.app.flags
# 	flags.DEFINE_integer('max_iter', 20000, 'maximum training iteration')
# 	flags.DEFINE_integer('max_time_step', 200, 'maximum time step per episode')
# 	flags.DEFINE_integer('obs_shape', 4, 'dimension of observation')
# 	flags.DEFINE_integer('action_shape', 1, 'dimension of action')
# 	flags.DEFINE_integer('cg_iters', 30, 'max_iteration for conjugate gradient')
# 	flags.DEFINE_integer('save_model_iters',100, 'save model per # of iterations')

# 	flags.DEFINE_float('discount', 0.99, 'discount factor for env')
# 	flags.DEFINE_float('max_action', 1.0, 'maximum action')
	
# 	flags.DEFINE_boolean('render', False, 'whether to render')
# 	flags.DEFINE_boolean('train_flag', True, 'whether in training process')
# 	flags.DEFINE_boolean('save_model', True, 'if save model')
# 	flags.DEFINE_boolean('with_context', False, 'if with context input')

# 	flags.DEFINE_string('name_scope', 'trpo', 'name scope for agent')
# 	flags.DEFINE_string('env_name', 'cartpole', 'name of environment')
# 	flags.DEFINE_string('save_dir', './Data/checkpoint/', 'checkpoint saving directory')

# 	pms = flags.FLAGS
# 	
