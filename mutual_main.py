from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse
# from utils.krylov import cg
# from utils.utils import *
# from agent.agent import Agent

from utils.paras import Paras_base
from actor.mtl_actor import MtlGaussianActor
from agent.vanilla_mtl import PpoMtl
from baseline.baseline import *
from model.mtl_net import MtlFcnnNet2
from model.net import Fcnn
from gym.envs.mujoco.walker2d import Walker2dEnv


# def main(gpu_num, exp_num, env = None, **kwargs):

tf.reset_default_graph()
speed = 2.
wind = 0.
gpu_num = 0
exp_num = 0
# task_num = kwargs.get('task_num', 0)
# num_of_paths = kwargs.get('num_of_paths', 10)
# dir_name = '/disk/scratch/chenyang/ppo_Data/dm_control/stl/walker_s%1.1f/w%1.1fg0.0/exp0/'%(speed,wind)
dir_name = '/disk/scratch/chenyang/new_ppo_Data/dm_control/mutual_stl/walker_s%1.1f/w%1.1fg0.0/exp%i'%(speed, wind, exp_num)
# dir_name = 'Data/ppo/stl/%s_exp%i/'%('half_cheetah',exp_num)
# dir_name = '/disk/scratch/chenyang/Data/trpo_stl/task_%i_exp%i/'%(task_num, exp_num)
if not os.path.isdir(dir_name):
	os.makedirs(dir_name)

env = Walker2dEnv()
env.reward_type = 'bound'
env.target_value = speed
env.model.opt.gravity[0] += wind
act_size = env.action_space.shape[0]
max_action = env.action_space.high
obs_size = env.observation_space.shape[0]

mod_num = 0

s_list = [0.,0.]
task_contexts = [{'speed': speed, 'wind': s} for s in s_list]

with tf.device('/gpu:%i'%(gpu_num)):
	pms = Paras_base().pms
	pms.save_model = True
	pms.save_dir = dir_name
	# env = HalfCheetahEnv() if env is None else env
	# env = gym.make('Pendulum-v0')


	pms.train_flag = True
	pms.render = False

	pms.obs_shape = obs_size
	pms.action_shape = act_size
	pms.max_action = max_action

	pms.max_iter = 501
	pms.max_time_step = 1024
	# pms.subsample_factor = .1
	pms.name_scope = 'ppo'
	# pms.max_kl = 0.01
	pms.min_std = 0.01
	pms.env_name = 'walker'
	pms.max_total_time_step = 1024 * 32 / 2
	# pms.nbatch = 4096 * 3
	pms.batchsize = 4096 * 3 

	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.1
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	actor_net = MtlFcnnNet2(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], [mod_num,mod_num,mod_num,mod_num], 2, name = pms.name_scope,\
		if_bias = [True], activation = ['tanh', 'tanh','tanh', 'None'], init = [.1, .1 ,.1,.01])
	# actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], name = pms.name_scope, if_bias = [True], activation = ['tanh', 'tanh', 'tanh', 'None'], init = [1. ,1.,1., .01])
	actor = MtlGaussianActor(actor_net, sess, pms)
	baseline_nets = [Fcnn(sess, pms.obs_shape, 1, [100, 50, 25], name = 'baseline_task%i'%i, if_bias = [True], \
		activation = ['relu', 'relu','relu','None'], init = [.1, .1, .1, .1]) for i in range(len(s_list))]
	baseline = [BaselineFcnn(b, sess, pms) for b in baseline_nets]
# 	baseline = BaselineFcnn(baselinet_net, sess, pms)

	# learn_agent = PPOagent(env, actor, baseline, sess, pms, [None])

	learn_agent = PpoMtl(env, actor, baseline, sess, pms, saver = [None], env_contexts =task_contexts)
	path_vector = [v for v in tf.trainable_variables() if 'path' in v.name]

	learn_agent.task_var_lists = [[v for v in tf.trainable_variables() if ('t%i'%i in v.name and 'path' not in v.name)] for i in range(2)]
	learn_agent.var_list = sum(learn_agent.task_var_lists,learn_agent.shared_var_list)


	learn_agent.init_mutual_vars()

saver = tf.train.Saver(max_to_keep = 101)
learn_agent.saver = saver

'''
	Learn Model
'''
with tf.device('/gpu:%i'%(gpu_num)):
	sess.run(tf.global_variables_initializer())
	[sess.run(tf.assign(v, np.array([1.]).astype(np.float32) ) ) for v in path_vector]
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
