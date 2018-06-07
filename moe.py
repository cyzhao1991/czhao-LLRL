from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse, pdb 

from utils.paras import Paras_base
from gym.envs.mujoco.walker2d import Walker2dEnv
from agent.moe_agent import MoeAgent
from model.context_net import MoeNet


s_list = [-4., -2., -1., 0., 1., 2., 4.]
task_name_list =['walker_s%1.1f/w0.0g0.0'%w for w in w_list]
filename_list = ['Data/mimic_data/stl/%s_exp0.npz'%task_name for task_name in task_name_list]

all_obs = []
all_acs = []

for filename, s in zip(filename_list, s_list):
	pre_data = np.load(filename)

	obs = np.concatenate(pre_data['obs'], axis = 0)
	acs = np.concatenate(pre_data['acs'], axis = 0)
	n,_ = obs.shape
	cos = np.ones([n, 1]) * s
	all_obs.append( np.stack([obs, cos], axis = 1) )
	all_acs.append(acs)

all_obs = np.concatenate( all_obs, axis = 0 )
all_acs = np.concatenate( all_acs, axis = 0 )

print(all_obs.shape)
print(all_acs.shape)
print(all_con.shape)

env = Walker2dEnv()
env.reward_type = 'bound'
env.target_value = 0.
default_context = np.array([0., 0., -9.8])
# env.model.opt.gravity[0] += default_context[0] + 0.
# env.model.opt.gravity[2] += default_context[2] - 5.
mod_num = 10

act_size = env.action_space.shape[0]
obs_size = env.observation_space.shape[0]

with tf.device('/gpu:%i'%(0)):
	pms = Paras_base().pms
	pms.save_model = True
	# pms.save_dir = dir_name
	pms.obs_shape = obs_size
	pms.action_shape = act_size
	pms.env_name = 'walker'
	pms.train_flag = False
	pms.context_shape = 7
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.1
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)

	# actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh','tanh', 'None'], init = [1., 1. ,1.,.01])
	actor_net = MoeNet(sess, pms.obs_shape, pms.action_shape, [100,50,25], [mod_num,mod_num,mod_num,mod_num],\
			name = pms.name_scope, if_bias = [True], activation = ['relu', 'relu', 'relu','None'], init = [.1, .1, .1, .01])
	agent = MoeAgent(sess, pms, actor_net)

	sess.run(tf.global_variables_initializer())
	tf.get_default_graph().finalize()
mimic_agent.learn(all_obs, all_acs)
saver = tf.train.Saver()
model_name = 'Data/mimic_data/multi_speed_moe_1.ckpt'
saver.save(sess, model_name)