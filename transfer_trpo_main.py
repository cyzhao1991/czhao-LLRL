from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse

from actor.actor import GaussianActor
from agent.trpo import TRPOagent
from baseline.baseline import BaselineZeros, BaselineFcnn

import gym
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.reacher import ReacherEnv
from model.net import *
from utils.paras import Paras_base

def main(gpu_num, exp_num, speed = None, **kwargs):
	tf.reset_default_graph()
	# gpu_num = 0
	# exp_num = 0
	# speed = 1.
	# wind = 0.
	wind = 0.
	dir_name = '/disk/scratch/chenyang/new_Data/dm_control/stl/walker_s%1.1f/w%1.1fg0.0/exp0/'%(speed,wind)
	dir_name = 'new_Data/dm_control/stl_ft/walker_s%1.1f/w%1.1fg0.0/exp%i/'%(speed, wind, exp_num)
	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)

	# with open('log.txt', 'a') as text_file:
	# 	text_file.write('gpu %i exp %i started.\n'%(gpu_num, exp_num))


	# while True:
	# 	goal = np.random.rand(1,2) * 0.4 - 0.2
	# 	if np.linalg.norm(goal) < 0.2:
	# 		break

	# env = walker.run()
	env = Walker2dEnv()
	env.reward_type = 'bound'
	env.target_value = speed
	env.model.opt.gravity[0] += wind
	# env.model.opt.gravity[2] += gravity
	# env = cartpole.balance()
	# act_spec = env.action_spec()
	# obs_spec = env.observation_spec()
	act_size = env.action_space.shape[0]
	max_action = env.action_space.high
	# obs_size = np.sum(np.sum([s.shape for s in obs_spec.values()])) + 1
	obs_size = env.observation_space.shape[0]
	with tf.device('/gpu:%i'%(gpu_num)):
		pms = Paras_base().pms
		pms.save_model = True
		pms.save_dir = dir_name
		pms.train_flag = True
		pms.render = False
		# action_size = env.action_spec().shape[0]
		# observation_size = env.observation_spec().shape[0]
		# max_action = env.action_space.high[0]
		pms.obs_shape = obs_size
		pms.action_shape = act_size
		pms.max_action = max_action
		#pms.num_of_paths = num_of_paths
		pms.max_iter = 501
		pms.max_time_step = 1000
		pms.subsample_factor = .1
		pms.max_kl = 0.01
		pms.min_std = 0.01
		pms.env_name = 'walker'
		pms.max_total_time_step = 10000
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.per_process_gpu_memory_fraction = 0.1
		config.gpu_options.allow_growth = True
		sess = tf.Session(config = config)
		# pms.render = True
		# actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100,50,25], name = pms.name_scope, if_bias = [True], activation = ['tanh', 'tanh', 'tanh','None'], init = [1. ,1., 1. ,.01])
		actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], name = pms.name_scope, if_bias = [True], activation = ['tanh', 'tanh','tanh', 'None'], init = [.1, .1 ,.1,.01])

		load_model_name = '/disk/scratch/chenyang/new_Data/dm_control/stl/walker_s1.0/w0.0g0.0/exp0/walker-iter1000.ckpt'
		# load_model_name = 'new_Data/dm_control/stl/walker_s1.0/w0.0g0.0/exp0/walker-iter1000.ckpt'

		tmp_saver = tf.train.Saver()
		tmp_saver.restore(sess, load_model_name)

		actor = GaussianActor(actor_net, sess, pms)

		baseline_net = Fcnn(sess, pms.obs_shape, 1, [100, 50, 25], name = 'baseline', if_bias = [True], activation = ['relu', 'relu','relu','None'], init = [.1, .1, .1, .1])
		baseline = BaselineFcnn(baseline_net, sess, pms)

		learn_agent = TRPOagent(env, actor, baseline, sess, pms, [None], goal = None)
		learn_agent.init_vars()
		learn_agent.boost_baseline = True

	saver = tf.train.Saver(max_to_keep = 101)
	learn_agent.saver = saver
	var_list = [v for v in tf.global_variables()]
	sess.run([v.initializer for v in var_list if not sess.run(tf.is_variable_initialized(v))])


	# # sess.run(tf.global_variables_initializer())
	saving_result = learn_agent.learn()

	sess.close()

	filename = dir_name + 'shelve_result'
	my_shelf = shelve.open(filename, 'n')
	my_shelf['saving_result'] = saving_result
	# my_shelf['goal'] = goal
	my_shelf.close()

'''
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default = 0, type = int)
	parser.add_argument('--exp', default = 0, type = int)
	parser.add_argument('--path', default = 10, type= int)
	args = vars(parser.parse_args())
	gpu_num = args['gpu']
	exp_num = args['exp']
	num_of_paths = args['path']

	main(gpu_num, exp_num, num_of_paths = num_of_paths)
'''