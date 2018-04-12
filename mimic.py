from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse, pdb 

from utils.paras import Paras_base
from actor.actor import GaussianActor
from agent.trpo import TRPOagent
from baseline.baseline import *
from model.net import *
from gym.envs.mujoco.walker2d import Walker2dEnv
from agent.mimic_agent import *

from actor.context_actor import Context_Gaussian_Actor
from model.context_net import Context_Fcnn_Net, Concat_Context_Fcnn_Net

w_list = [-3., -1.5, 0., 1.5, 3. ]
w_list_2 = [-3., 0., 1.5]
task_name_list = ['w%1.1fg%1.1f'%(w, 0) for w in w_list]
task_name_list2 = ['w%1.1fg0.0'%w for w in w_list_2]
filename_list = ['Data/mimic_data/stand_%s.npz'%task_name for task_name in task_name_list]
filename_list2 = ['Data/mimic_data/walk_%s.npz'%task_name for task_name in task_name_list2]
all_obs = []
all_acs = []
all_con = []
for filename, w in zip(filename_list, w_list):
	pre_data = np.load(filename)
	context = np.array([1., 0., w])
	obs = np.concatenate(pre_data['obs'], axis = 0)
	acs = np.concatenate(pre_data['acs'], axis = 0)
	n,_ = obs.shape
	con = np.tile(context, [n,1])

	all_obs.append(obs)
	all_acs.append(acs)
	all_con.append(con)

# for filename, w in zip(filename_list2, w_list_2):
# 	pre_data = np.load(filename)
# 	context = np.array([0., 1., w])
# 	obs = np.concatenate(pre_data['obs'], axis = 0)
# 	acs = np.concatenate(pre_data['acs'], axis = 0)
# 	n,_ = obs.shape
# 	con = np.tile(context, [n,1])

# 	all_obs.append(obs)
# 	all_acs.append(acs)
# 	all_con.append(con)

all_obs = np.concatenate( all_obs, axis = 0 )
all_acs = np.concatenate( all_acs, axis = 0 )
all_con = np.concatenate( all_con, axis = 0 )

print(all_obs.shape)
print(all_acs.shape)
print(all_con.shape)

env = Walker2dEnv()
env.reward_type = 'bound'
env.target_value = 0.
default_context = np.array([0., 0., -9.8])
# env.model.opt.gravity[0] += default_context[0] + 0.
# env.model.opt.gravity[2] += default_context[2] - 5.
mod_num = 5

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
	pms.max_time_step = 500
	pms.env_name = 'walker_stand'
	pms.train_flag = False
	pms.context_shape = 3
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.1
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)

	# actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh','tanh', 'None'], init = [1., 1. ,1.,.01])
	actor_net = Context_Fcnn_Net(sess, pms.obs_shape, pms.action_shape, pms.context_shape, [100,50,25], [mod_num,mod_num,mod_num,mod_num],\
			name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh', 'tanh','None'], init = [.1, .1, .1, .01])
	mimic_agent = MtlMimicAgent(env, sess, pms, actor_net)

	sess.run(tf.global_variables_initializer())
mimic_agent.learn(all_obs, all_con, all_acs)
saver = tf.train.Saver()
model_name = 'Data/mimic_data/stand_mtl_mimic_2.ckpt'
saver.save(sess, model_name)