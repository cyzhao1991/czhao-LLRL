from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse, pdb 

from utils.paras import Paras_base
from actor.mtl_actor import MtlGaussianActor
from agent.vanilla_mtl import PpoMtl
from baseline.baseline import *
from model.mtl_net import MtlFcnnNet
from model.net import Fcnn
from gym.envs.mujoco.walker2d import Walker2dEnv
# from agent.mimic_agent import *

# from actor.context_actor import Context_Gaussian_Actor
# from model.context_net import Context_Fcnn_Net, Concat_Context_Fcnn_Net
# # from agent.context_trpo import Context_TRPO_Agent
# from agent.context_trpo_ver2 import Context_TRPO_Agent

tf.reset_default_graph()


GRAVITY = 0.
WIND = 0.
gpu_num = 0
exp_num = 1
# speed = SPEED #if speed is None else speed
gravity = GRAVITY
wind = WIND

dir_name = 'ppo_Data/dm_control/mtl/walker/multiwind/exp%i/'%(exp_num)
# dir_name = '/disk/scratch/chenyang/Data/dm_control/mtl/walker/w%1.1fg%1.1f/exp%i/'%(speed, wind, gravity, exp_num)
# dir_name = '/disk/scratch/chenyang/Data/trpo_stl/task_%i_exp%i/'%(task_num, exp_num)
if not os.path.isdir(dir_name):
	os.makedirs(dir_name)

env = Walker2dEnv()
env.reward_type = 'bound'
env.target_value = 0.
env.model.opt.gravity[0] += wind
env.model.opt.gravity[2] += gravity
act_size = env.action_space.shape[0]
max_action = env.action_space.high
obs_size = env.observation_space.shape[0]

mod_num = 1

s_list = [-2., 0., 2.]
task_contexts = [{'speed': 2., 'wind': s} for s in s_list]
# envs = [Walker2dEnv() for _ in range(len(s_list))]
# for e, s in zip(envs, s_list):
# 	e.reward_type = 'bound'
# 	e.target_value = s
# 	e.model.opt.gravity[0] += wind
# 	e.model.opt.gravity[2] += gravity



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
	pms.max_time_step = 1024
	pms.subsample_factor = .1
	pms.max_kl = 0.01
	pms.min_std = 0.01
	pms.nbatch = 4096 * len(s_list)
	pms.env_name = 'walker'
	pms.max_total_time_step = 1024 * 32
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.1
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	# pms.render = True
	# actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100,50,25], name = pms.name_scope, if_bias = [True], activation = ['tanh', 'tanh', 'tanh','None'], init = [1. ,1., 1. ,.01])
	actor_net = MtlFcnnNet(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], [mod_num,mod_num,mod_num,mod_num], len(s_list),name = pms.name_scope, \
		if_bias = [True], activation = ['tanh', 'tanh','tanh', 'None'], init = [.1, .1 ,.1,.01])
	actor = MtlGaussianActor(actor_net, sess, pms)

	baseline_nets = [Fcnn(sess, pms.obs_shape, 1, [100, 50, 25], name = 'baseline_task%i'%i, if_bias = [True], \
		activation = ['relu', 'relu','relu','None'], init = [.1, .1, .1, .1]) for i in range(len(s_list))]
	baseline = [BaselineFcnn(b, sess, pms) for b in baseline_nets]

	learn_agent = PpoMtl(env, actor, baseline, sess, pms, saver = [None], env_contexts =task_contexts)
	path_vector = [v for v in tf.trainable_variables() if 'path' in v.name]

	learn_agent.task_var_lists = [[v for v in tf.trainable_variables() if ('t%i'%i in v.name and 'path' not in v.name)] for i in range(3)]
	learn_agent.var_list = sum(learn_agent.task_var_lists,learn_agent.shared_var_list)
	# learn_agent.init_vars()
	learn_agent.init_vars()

saver = tf.train.Saver(max_to_keep = 101)
learn_agent.saver = saver


'''
	Learn Model
'''
with tf.device('/gpu:%i'%(gpu_num)):
	sess.run(tf.global_variables_initializer())
	[sess.run(tf.assign(v, np.array([1.,1.]).astype(np.float32) ) ) for v in path_vector if 'h3' not in v.name]
	[sess.run(tf.assign(v, np.array([0.,1.]).astype(np.float32) ) ) for v in path_vector if 'h3' in v.name]
	saving_result = learn_agent.learn()


sess.close()

filename = dir_name + 'shelve_result'
my_shelf = shelve.open(filename, 'n')
my_shelf['saving_result'] = saving_result
# my_shelf['goal'] = goal
my_shelf.close()


'''
	Restore Model
'''
# exp_num = 0
# dir_name = 'ppo_Data/dm_control/mtl/walker/multispeed/exp%i/'%(exp_num)
# filename = dir_name+'walker-iter500.ckpt'
# if 'wind' in dir_name:
# 	learn_agent.env_contexts = [{'speed': 2., 'wind':s} for s in s_list]
# elif 'speed' in dir_name:
# 	learn_agent.env_contexts = [{'speed': s, 'wind':0.} for s in s_list]

# saver.restore(sess, filename)
# pms.train_flag = False
# pms.render = True
