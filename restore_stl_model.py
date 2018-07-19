from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse
# from utils.krylov import cg
# from utils.utils import *
# from agent.agent import Agent

# from actor.context_actor import Context_Gaussian_Actor
# from agent.context_trpo import Context_TRPO_Agent
# from baseline.context_baseline import Context_Baseline

# from env.cartpole import CartPoleEnv
# from gym.envs.mujoco.reacher import ReacherEnv

# from model.context_net import Context_Fcnn_Net, Concat_Context_Fcnn_Net
from utils.paras import Paras_base

from actor.actor import GaussianActor
from agent.trpo import TRPOagent
from baseline.baseline import *
# from env.cartpole import CartPoleEnv
# from gym.envs.mujoco.reacher import ReacherEnv
# from gym.envs.mujoco.walker2d import Walker2dEnv
from model.net import *
from gym.envs.mujoco.walker2d import Walker2dEnv
# from dm_control.suite import walker
# from dm_control.suite import cartpole

tf.reset_default_graph()
'''
contextual policy
'''
mod_num = 5 #kwargs.get('mod_num', 10)
num_of_paths = 10#kwargs.get('num_of_paths', 10)
num_of_tasks = 10#kwargs.get('num_of_tasks', 10)
# task_num = kwarg.get('task_num', 0)
# num_of_paths = kwarg.get('num_of_paths', 100)

# gravity_list = np.arange(0.2, 2.1, .2) * 9.8
# env_paras_list = [(g, 1.0, 0.1) for g in gravity_list]
# # random.shuffle(env_paras_list)
# env_paras_list = env_paras_list[0:num_of_tasks]

# env_list = []
# [env_list.append(CartPoleEnv(g, mc, mp)) for g,mc,mp in env_paras_list]
# num_of_envs = len(env_list)

# test_gravity_list = np.random.rand(10)* 20 + 10
# test_env_list = [CartPoleEnv(g, 1., .1) for g in gravity_list]

# for env1,g in zip(env_list, env_paras_list):
# 	env1.context = np.array([1, g[0]], dtype = np.float64)
# for env1,g in zip(test_env_list, test_gravity_list):
# 	env1.context = np.array([1, g], dtype = np.float64)

# goal_list = np.zeros([0,2])
# while len(goal_list) < 10:
# 	goal = np.random.rand(1,2) * 0.4 - 0.2
# 	if np.linalg.norm(goal) < 0.2:
# 		goal_list = np.append(goal_list, goal, axis = 0)
# context_list = np.insert(goal_list * 10, 0, 1, axis = 1)
'''
waler2d context setup
'''
# ori_g = np.array([0., 0., -9.8])
# delta_g = np.array( [[i, 0., j] for i in [-1., 0., 1.] for j in [-2.5, 0., 2.5]] )
# context_g = ori_g + delta_g
# context_list = np.insert(context_g, 0, 1, axis = 1)
# env = Walker2dEnv()
# env = walker.stand()
# context_range = np.array([2., 0., 2.])
# env_contexts = np.array([[i, 0, j] for i in [-1. , 0., 1.] for j in [-1., 0., 1. ]])
# act_spec = env.action_spec()
# obs_spec = env.observation_spec()
# act_size = act_spec.shape[0]
# max_action = act_spec.maximum
# obs_size = np.sum(np.sum([s.shape for s in obs_spec.values()])) + 1	
'''
Build Network
'''

# with tf.device('/gpu:%i'%(0)):
# 	pms = Paras_base().pms
# 	pms.save_model = True
# 	# pms.save_dir = dir_name
# 	# env = env_list[0]
# 	# env = ReacherEnv()
# 	# env = gym.make('Pendulum-v0')
# 	# action_size = env.action_space.shape[0]
# 	# observation_size = env.observation_space.shape[0]
# 	# max_action = env.action_space.high[0]
# 	pms.obs_shape = obs_size
# 	pms.action_shape = act_size
# 	pms.max_action = max_action
# 	pms.num_of_paths = num_of_paths
# 	pms.subsample_factor = 0.1
# 	# pms.l1_regularizer = 0.01
# 	pms.context_shape = 4
# 	pms.max_time_step = 1000
# 	pms.env_name = 'walker_stand'
# 	# pms.contextual_method = 'concatenate'
# 	pms.independent_std = False
# 	config = tf.ConfigProto(allow_soft_placement = True)
# 	config.gpu_options.per_process_gpu_memory_fraction = 0.3
# 	config.gpu_options.allow_growth = True
# 	sess = tf.Session(config = config)

# 	# tf_goal_list = tf.constant(goal_list, tf.float32, name = 'goal_list')
# 	# tf_context_list = tf.constant(context_list, tf.float32, name = 'goal_list')
# 	# actor_net = Concat_Context_Fcnn_Net(sess, pms.obs_shape, pms.action_shape, pms.context_shape, [64,64], name = pms.name_scope,\
# 	# 	if_bias = [False], activation = ['tanh', 'tanh', 'None'], init = [1., 1., .01])
# 	actor_net = Context_Fcnn_Net(sess, pms.obs_shape, pms.action_shape, pms.context_shape, [100,50,25], [mod_num,mod_num,mod_num, mod_num],\
# 		name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh','tanh','None'])
# 	actor = Context_Gaussian_Actor(actor_net, sess, pms)
# 	baseline_net = Concat_Context_Fcnn_Net(sess, pms.obs_shape, 1, pms.context_shape, [100,50,25], name = 'baseline',\
# 		if_bias = [False], activation = ['tanh', 'tanh', 'tanh', 'None'], init = [1., 1., .1, .1])
# 	baseline = Context_Baseline(baseline_net, sess, pms)
# 	# baseline = BaselineZeros(sess, pms)

# 	learn_agent = Context_TRPO_Agent(env, actor, baseline, sess, pms, [None], context_range = context_range)

# saver = tf.train.Saver()
# learn_agent.saver = saver
# model_file = './Data/dm_control/context_mtl/walker_stand/exp0/walker_stand-iter430.ckpt'
# learn_agent.saver.restore(sess, model_file)
# learn_agent.pms.render = False
# learn_agent.pms.train_flag = False

# stats = [learn_agent.get_single_path(i) for i in range(9)]
# [print(len(s['rewards']), np.sum(s['rewards'])) for s in stats]
# print(np.mean([np.sum(s['rewards']) for s in stats]))
# print('--------------------------------')
# learn_agent.env = test_env_list
# stats = [learn_agent.get_single_path(i) for i in range(10)]
# [print(len(s['rewards']), np.sum(s['rewards'])) for s in stats]
# print('--------------------------------')
# learn_agent.env = env_list
# stats = [learn_agent.get_single_path(i) for i in range(10)]
# [print(len(s['rewards']), np.sum(s['rewards'])) for s in stats]


'''
stl
'''

num_of_paths = 10
dir_name = 'Data/dm_control/stl/walker_walk/w0.0g-5.0/exp0/'
# dir_name = '/disk/scratch/chenyang/Data/trpo_stl/task_%i_exp%i/'%(task_num, exp_num)
# with open('log.txt', 'a') as text_file:
# 	text_file.write('gpu %i exp %i started.\n'%(gpu_num, exp_num))
# gravity_list = np.arange(0.2, 2.1, .2) * 9.8
# env_list = [CartPoleEnv(g, 1.0, 0.1) for g in gravity_list]
# env = env_list[1]
# env = ReacherEnv()
# env = walker.stand()
# act_spec = env.action_spec()
# obs_spec = env.observation_spec()
# act_size = act_spec.shape[0]
# max_action = act_spec.maximum
# obs_size = np.sum(np.sum([s.shape for s in obs_spec.values()])) + 1

SPEED = 2.
GRAVITY = 0.
WIND = 0.
exp_num = 2
speed = SPEED
gravity = GRAVITY
wind = WIND

# env = Walker2dEnv()
# env.reward_type = 'bound'
# env.target_value = speed
# env.model.opt.gravity[0] += wind
# env.model.opt.gravity[2] += gravity

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
# myshelf = shelve.open('Data/checkpoint/stl/exp2/shelve_result')
# goal = myshelf['goal']
# goal = np.array([[ 0.05421805,  0.12854432]])
with tf.device('/gpu:%i'%(0)):
	pms = Paras_base().pms
	pms.save_model = True
	pms.save_dir = dir_name
	# env = CartPoleEnv() if env is None else env
	# env = gym.make('Pendulum-v0')
	pms.obs_shape = obs_size
	pms.action_shape = act_size
	pms.max_action = max_action
	pms.num_of_paths = num_of_paths
	pms.subsample_factor = 0.1
	pms.max_time_step = 1000
	pms.env_name = 'walker_stand'
	pms.train_flag = True
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.1
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)

	actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh','tanh', 'None'], init = [1., 1. ,1.,.01])
	actor = GaussianActor(actor_net, sess, pms)

	baselinet_net = Fcnn(sess, pms.obs_shape, 1, [100, 50, 25], name = 'baseline', if_bias = [False], activation = ['relu', 'relu','relu','None'], init = [.1, .1, .1, .1])
	# baseline = BaselineZeros(sess, pms)
	baseline = BaselineFcnn(baselinet_net, sess, pms)


	learn_agent = TRPOagent(env, actor, baseline, sess, pms, [None], goal = None)
	learn_agent.init_vars()
saver = tf.train.Saver()
learn_agent.saver = saver
sess.run(tf.global_variables_initializer())
# learn_agent.get_single_path()
# model_file = '../Data/arv/trpo_stl_Jan28/task_1_exp3/cartpole-iter190.ckpt'
# model_file = 'Data/checkpoint/stl/exp0_nogoal/cartpole-iter990.ckpt'
learn_agent.pms.render = False
learn_agent.pms.train_flag = False

wind_list = [-4, -2, -1, 0, 1, 2, 4]
overall_mean = np.zeros([7,7])
overall_std = np.zeros([7,7])
speed = -1.5
wind = 0.
gravity = 0.
exp_num = 0
# model_file = 'Data/dm_control/finetune/walker_s%1.1f/w%1.1fg%1.1f/exp3/walker-iter990.ckpt'%(speed, wind, gravity)
# model_file = 'Data/dm_control/stl(con)/walker_s%1.1f/w%1.1fg%1.1f/exp%i/walker-iter500.ckpt'%(speed, wind, gravity, exp_num)
model_file = 'Data/dm_control/stl/walker_s2.0/w0.0g0.0/exp0/walker-iter990.ckpt'
learn_agent.saver.restore(sess, model_file)
env.target_value = 2.

'''
for i, wind in enumerate(wind_list):
	model_file = 'Data/dm_control/finetune/walker_s%1.1f/w%1.1fg%1.1f/exp2/walker-iter990.ckpt'%(speed, wind, gravity)
	# model_file = 'Data/dm_control/stl/walker_s1.0/w0.0g0.0/exp0/walker-iter990.ckpt'
	learn_agent.saver.restore(sess, model_file)
	for j, task_wind in enumerate(wind_list):
		env.model.opt.gravity[0] = task_wind
		env.target_value = 1.0#task_speed
		stats = [learn_agent.get_single_path() for _ in range(10)]
		overall_mean[i][j] = np.mean([np.sum(s['rewards']) for s in stats])
		overall_std[i][j]  = np.std( [np.sum(s['rewards']) for s in stats])
		print(i,j)
'''
# print([np.sum(s['rewards']) for s in stats])
# # # env.render(close = True)
# print(np.mean( [np.sum(s['rewards']) for s in stats] ) )
