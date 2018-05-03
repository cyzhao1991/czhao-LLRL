from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse
# from utils.krylov import cg
# from utils.utils import *
# from agent.agent import Agent

from actor.actor import GaussianActor
from agent.trpo import TRPOagent
from baseline.baseline import BaselineZeros, BaselineFcnn

# from dm_control.suite import walker
# from dm_control.suite import cartpole

import gym
from gym.envs.mujoco.walker2d import Walker2dEnv

from model.net import *
from utils.paras import Paras_base
'''
def main(gpu_num, exp_num, env = None, **kwargs):
'''
# task_num = kwargs.get('task_num', 0)
# num_of_paths = kwargs.get('num_of_paths', 10)
# speed = kwargs.get('speed', 0.)
# wind = kwargs.get('wind', 0.)
# gravity = kwargs.get('gravity', 0.)
SPEED = 1.
GRAVITY = 0.
WIND = 4.
exp_num = 0
speed = SPEED
gravity = GRAVITY
wind = WIND

dir_name = 'Data/dm_control/stl/walker_s%1.1f/w%1.1fg%1.1f/exp%i/'%(speed, wind, gravity, exp_num)
# dir_name = '/disk/scratch/chenyang/Data/dm_control/stl/walker_s%1.1f/w%1.1fg%1.1f/exp%i/'%(speed, wind, gravity, exp_num)
# dir_name = '/disk/scratch/chenyang/Data/trpo_stl/task_%i_exp%i/'%(task_num, exp_num)
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
env.model.opt.gravity[2] += gravity
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
	pms.num_of_paths = num_of_paths
	pms.max_iter = 1000
	pms.max_time_step = 1000
	pms.subsample_factor = 1.
	pms.max_kl = 0.01
	pms.min_std = 0.01
	pms.env_name = 'walker'
	pms.max_total_time_step = 4096
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.1
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	# pms.render = True
	# actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100,50,25], name = pms.name_scope, if_bias = [True], activation = ['tanh', 'tanh', 'tanh','None'], init = [1. ,1., 1. ,.01])
	actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh','tanh', 'None'], init = [.1, .1 ,.1,.01])
	actor = GaussianActor(actor_net, sess, pms)

	baseline_net = Fcnn(sess, pms.obs_shape, 1, [100, 50, 25], name = 'baseline', if_bias = [False], activation = ['relu', 'relu','relu','None'], init = [.1, .1, .1, .1])
	baseline = BaselineFcnn(baseline_net, sess, pms)

	learn_agent = TRPOagent(env, actor, baseline, sess, pms, [None], goal = None)

saver = tf.train.Saver(max_to_keep = 100)
learn_agent.saver = saver
sess.run(tf.global_variables_initializer())
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