from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse

from utils.paras import Paras_base
from actor.actor import GaussianActor
from agent.trpo import TRPOagent
from baseline.baseline import *
from model.net import *
from gym.envs.mujoco.walker2d import Walker2dEnv
tf.reset_default_graph()
env = Walker2dEnv()
env.reward_type = 'bound'
env.target_value = 1.
default_context = np.array([0., 0., -9.8])
# env.model.opt.gravity[0] += default_context[0] + 0.
# env.model.opt.gravity[2] += default_context[2] - 5.

act_size = env.action_space.shape[0]
max_action = env.action_space.high
obs_size = env.observation_space.shape[0]

with tf.device('/gpu:%i'%(0)):
	pms = Paras_base().pms
	pms.save_model = True
	# pms.save_dir = dir_name
	pms.obs_shape = obs_size
	pms.action_shape = act_size
	pms.max_action = max_action
	pms.num_of_paths = 10
	pms.subsample_factor = 0.1
	pms.max_time_step = 1000
	pms.env_name = 'walker_stand'
	pms.train_flag = False
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

saver = tf.train.Saver()
learn_agent.saver = saver
sess.run(tf.global_variables_initializer())

model_file_prefix = 'Data/dm_control/stl/'
model_file_suffix = '/exp0/walker-iter990.ckpt'
# w_list = [-3., -1.5, 0., 1.5, 3.]
w_list = [-4, -2, -1, 0, 1, 2, 4]
task_name_list = ['walker_s%1.1f/w0.0g0.0'%w for w in w_list]
for w, task_name in zip(w_list, task_name_list):
	# env.model.opt.gravity[0] = w
	env.target_value = w
	model_file_name = model_file_prefix + task_name + model_file_suffix
	learn_agent.saver.restore(sess, model_file_name)
	stats = []
	for i in range(1000):
		stats.append(learn_agent.get_single_path())
		sys.stdout.write('%i-th path sampled. \r'%i)
		sys.stdout.flush()
	#stats = [learn_agent.get_single_path() for _ in range(1000)]
	obs = np.array( [s['observations'] for s in stats] )
	acs = np.array( [s['actions'] for s in stats] )
	res = np.array( [s['rewards'] for s in stats] )
	print( (np.mean([np.sum(a) for a in res]) , np.std([np.sum(a) for a in res]) ))

	data_filename = 'Data/mimic_data/stl/%s_exp0.npz'%task_name
	np.savez(data_filename, obs = obs, acs = acs, res = res)