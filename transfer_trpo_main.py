from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse
# from utils.krylov import cg
# from utils.utils import *
# from agent.agent import Agent

from actor.context_actor import Context_Gaussian_Actor
from agent.context_trpo import Context_TRPO_Agent
from baseline.baseline import BaselineZeros, BaselineFcnn
from baseline.context_baseline import Context_Baseline
# from env.cartpole import CartPoleEnv
# from gym.envs.mujoco.reacher import ReacherEnv
# from gym.envs.mujoco.walker2d import Walker2dEnv
from dm_control.suite import walker

from model.context_net import Context_Fcnn_Net, Concat_Context_Fcnn_Net
from utils.paras import Paras_base

def main(gpu_num, exp_num, env = None, **kwargs):

	mod_num = kwargs.get('mod_num', 5)
	num_of_paths = kwargs.get('num_of_paths', 10)
	num_of_tasks = kwargs.get('num_of_tasks', 10)
	suffix = kwargs.get('suffix', '')
	# task_num = kwarg.get('task_num', 0)
	# num_of_paths = kwarg.get('num_of_paths', 100)
	dir_name = 'Data/dm_control/transfer_mtl/%s/exp%i%s/'%('walker_walk',exp_num, suffix)
	# dir_name = '/disk/scratch/chenyang/Data/context_trpo_path10/mod_%i_exp%i/'%(mod_num, exp_num)
	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)

	# with open('log.txt', 'a') as text_file:
	# 	text_file.write('gpu %i exp %i started.\n'%(gpu_num, exp_num))
	
	# gravity_list = np.arange(0.2, 2.1, .2) * 9.8
	# env_paras_list = [(g, 1.0, 0.1) for g in gravity_list]
	# # random.shuffle(env_paras_list)
	# env_paras_list = env_paras_list[0:num_of_tasks]
	# env_list = []
	# [env_list.append(CartPoleEnv(g, mc, mp)) for g,mc,mp in env_paras_list]
	# num_of_envs = len(env_list)

	# for env1,g in zip(env_list, env_paras_list):
	# 	env1.context = np.array([1, g[0]], dtype = np.float64)



	# goal_list = np.zeros([0,2])
	# while len(goal_list) < 10:
	# 	goal = np.random.rand(1,2) * 0.4 - 0.2
	# 	if np.linalg.norm(goal) < 0.2:
	# 		goal_list = np.append(goal_list, goal, axis = 0)
	# context_list = np.insert(goal_list * 10, 0, 1, axis = 1)
	# goal_list = np.insert(goal_list, 0, 1, axis = 1)
	# ori_g = np.array([0., 0., -9.8])
	# delta_g = np.array( [[i, 0., j] for i in [-1., 0., 1.] for j in [-2.5, 0., 2.5]] )
	# context_g = ori_g + delta_g
	# context_list = np.insert(context_g, 0, 1, axis = 1)
	# env = Walker2dEnv()
	env = walker.walk()
	context_range = np.array([10., 0., 10.])

	act_spec = env.action_spec()
	obs_spec = env.observation_spec()
	act_size = act_spec.shape[0]
	max_action = act_spec.maximum
	obs_size = np.sum(np.sum([s.shape for s in obs_spec.values()])) + 1

	with tf.device('/gpu:%i'%(gpu_num)):
		pms = Paras_base().pms
		pms.save_model = True
		pms.save_dir = dir_name

		pms.obs_shape = obs_size
		pms.action_shape = act_size
		pms.max_action = max_action
		pms.num_of_paths = num_of_paths
		pms.max_iter = 500
		pms.max_time_step = 1000
		pms.max_total_time_step = 45000
		pms.subsample_factor = 0.1
		pms.max_kl = 0.1
		pms.min_std = 0.01
		pms.env_name = 'walker_stand'
		# pms.contextual_method = 'concatenate'
		# pms.l1_regularizer = 0.01
		pms.context_shape = 4
		# pms.max_time_step = 500
		# pms.env_name = 'reacher'
		pms.independent_std = False
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.per_process_gpu_memory_fraction = 0.3
		config.gpu_options.allow_growth = True

		sess = tf.Session(config = config)

		# tf_goal_list = tf.constant(goal_list, tf.float32, name = 'goal_list')
		# tf_context_list = tf.constant(context_list, tf.float32, name = 'goal_list')
		# actor_net = Concat_Context_Fcnn_Net(sess, pms.obs_shape, pms.action_shape, pms.context_shape, [64,64], name = pms.name_scope,\
		# 	if_bias = [False], activation = ['tanh', 'tanh', 'None'], init = [1., 1., .01])
		actor_net = Context_Fcnn_Net(sess, pms.obs_shape, pms.action_shape, pms.context_shape, [100,50,25], [mod_num,mod_num,mod_num,mod_num],\
			name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh', 'tanh','None'], init = [.1, .1, .1, 0.01])
		actor = Context_Gaussian_Actor(actor_net, sess, pms)
		
		baseline_net = Concat_Context_Fcnn_Net(sess, pms.obs_shape, 1, pms.context_shape, [100,50,25], name = 'baseline',\
			if_bias = [False], activation = ['relu', 'relu','relu','None'], init = [.1, .1, .1, .1])
		baseline = Context_Baseline(baseline_net, sess, pms)
		# baseline = BaselineZeros(sess, pms)

		learn_agent = Context_TRPO_Agent(env, actor, baseline, sess, pms, [None], context_range = context_range)

	saver = tf.train.Saver()
	learn_agent.saver = saver
	with tf.device('/gpu:%i'%(gpu_num)):
		sess.run(tf.global_variables_initializer())
	model_file = 'Data/dm_control/stl/walker_run/exp4/walker_run-iter490.ckpt'
	learn_agent.saver.restore(sess, model_file)
	learned_s_vector = np.array(sess.run(learn_agent.task_var_list))


	learn_agent.update_vars()
	pdb.set_trace()
	tf.get_default_graph().finalize()



	saving_result = learn_agent.learn()

	sess.close()

	filename = dir_name + 'shelve_result'
	my_shelf = shelve.open(filename, 'n')
	my_shelf['saving_result'] = saving_result

	# my_shelf['goal_list'] = goal_list

	my_shelf.close()
	# with open('log.txt', 'a') as text_file:
	# 	text_file.write('gpu %i exp %i finished.\n'%(gpu_num, exp_num))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default = 0, type = int)
	parser.add_argument('--exp', default = 0, type = int)
	parser.add_argument('--mod', default = 5, type = int)
	parser.add_argument('--path', default = 10, type= int)
	parser.add_argument('--opt', default = '', type = str)
	args = vars(parser.parse_args())
	gpu_num = args['gpu']
	exp_num = args['exp']
	mod_num = args['mod']
	num_of_paths = args['path']
	suffix = args['opt']
	main(gpu_num, exp_num, mod_num = mod_num, num_of_paths = num_of_paths, suffix = suffix)
