from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse, pdb, cv2

from utils.paras import Paras_base
from actor.actor import GaussianActor
from agent.trpo import TRPOagent
from baseline.baseline import *
from model.net import *
from gym.envs.mujoco.walker2d import Walker2dEnv
from agent.mimic_agent import *

from actor.context_actor import Context_Gaussian_Actor
from model.context_net import Context_Fcnn_Net, Concat_Context_Fcnn_Net
# from agent.context_trpo import Context_TRPO_Agent
from agent.context_trpo_ver1 import Context_TRPO_Agent

tf.reset_default_graph()

WIND = 3.
dir_name = 'Data/dm_control/finetune/mtl_walker_s1.0/w%1.1fg0.0/exp0/'%WIND
if not os.path.isdir(dir_name):
	os.makedirs(dir_name)
env = Walker2dEnv()
env.reward_type = 'bound'
env.target_value = 1.
default_context = np.array([0., 0., -9.8])
env.model.opt.gravity[0] = WIND
# env.model.opt.gravity[2] += default_context[2] - 5.
mod_num = 5

act_size = env.action_space.shape[0]
max_action = env.action_space.high
obs_size = env.observation_space.shape[0]

with tf.device('/gpu:%i'%(0)):
	pms = Paras_base().pms
	pms.save_model = True
	pms.save_dir = dir_name
	pms.obs_shape = obs_size
	pms.action_shape = act_size
	pms.max_action = max_action
	pms.max_total_time_step = 4096
	pms.max_kl = 0.01
	pms.min_std = 0.01
	pms.max_iter = 500	
	pms.subsample_factor = 1.
	pms.max_time_step = 1000
	pms.env_name = 'walker'
	pms.train_flag = True
	pms.context_shape = 7
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.1
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)

	# actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh','tanh', 'None'], init = [1., 1. ,1.,.01])
	actor_net = Context_Fcnn_Net(sess, pms.obs_shape, pms.action_shape, pms.context_shape, [100,50,25], [mod_num,mod_num,mod_num,mod_num],\
			name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh', 'tanh','None'], init = [.1, .1, .1, .01])
	# mimic_agent = MtlMimicAgent(env, sess, pms, actor_net)

	# sess.run(tf.global_variables_initializer())
# mimic_agent.learn(all_obs, all_con, all_acs)
saver = tf.train.Saver()
# model_name = 'Data/mimic_data/multi_wind_mimic_0.ckpt'
# saver.save(sess, model_name)
# saver.restore(sess, model_name)

# learned_var_list = [v for v in tf.trainable_variables()]
# learned_s = sess.run([v for v in learned_var_list if 's_vector' in v.name])
# learned_s_nonzero = [np.abs(s) < 0.01 for s in learned_s]
# inactive_module = [np.all(s, axis = 0) for s in learned_s_nonzero]
# inactive_module_index = np.nonzero(inactive_module)
# inactive_module_name = ['h%i_m%i'%(i,j) for i,j in zip(inactive_module_index[0], inactive_module_index[1])]

#actor.shared_var_list = [v for v in ]

with tf.device('/gpu:%i'%(0)):

	# test_variable = tf.Variable(np.array([0.]).astype(np.float32), tf.float32, name = 'test_variable')

	actor = Context_Gaussian_Actor(actor_net, sess, pms)
	baseline_net = Fcnn(sess, pms.obs_shape, 1, [100, 50, 25], name = 'baseline', if_bias = [False], activation = ['relu', 'relu','relu','None'], init = [.1, .1, .1, .1])
	baseline = BaselineFcnn(baseline_net, sess, pms)
	# actor.shared_var_list = [v for v in learned_var_list if np.any([name in v.name for name in inactive_module_name])]
	# actor.shared_var_list.append(actor.action_logstd)
	# actor.var_list = actor.shared_var_list + actor.task_var_list
	learn_agent = Context_TRPO_Agent(env,actor, baseline,sess, pms, [None], env_contexts =np.array([[0,0,0,1,0,0,0]]))

# all_var_list = tf.global_variables()
# all_var_initialized = sess.run([tf.is_variable_initialized(v) for v in all_var_list])
# not_initialized_vars = [v for v, f in zip(all_var_list, all_var_initialized) if not f]
# sess.run(tf.variables_initializer(not_initialized_vars))
learn_agent.saver = saver
#sess.run([v.initializer for v in all_var_list[-5:]])
var_list  = [v for v in tf.trainable_variables()]
s_vars = [v for v in var_list if 's_vector' in v.name]
kb_vars = [v for v in var_list if 'KB' in v.name]
l2_loss = [tf.nn.l2_loss(v) for v in kb_vars]
logstd = learn_agent.actor.action_logstd
# logstd = [v for v in var_list if 'logstd' in v.name]
np.set_printoptions(precision = 3)
sess.run(tf.global_variables_initializer())
# model_name = dir_name + 'walker-iter490.ckpt'
# learn_agent.saver.restore(sess, model_name)


learned_s = sess.run(s_vars)
learned_s_nonzero = [np.abs(s) < 0.01 for s in learned_s]
zero_out_s = np.array([np.where(s_nz, 0, s) for s_nz, s in zip(learned_s_nonzero, learned_s)])
# sess.run([tf.assign(s, zo_s) for s, zo_s in zip(s_vars, zero_out_s)])

l1 = []
l0 = []
WIND = -3.
# dir_name = 'Data/dm_control/finetune/mtl_walker_s1.0/w%1.1fg0.0/exp0/'%WIND
dir_name = 'Data/dm_control/finetune_ver1/mtl_walker_s1.0/w%1.1fg0.0/test_exp0/'%WIND
for i in range(100):
	model_name = dir_name + 'walker-iter%i.ckpt'%(i)
	learn_agent.saver.restore(sess, model_name)
	all_s = sess.run(s_vars)

	l1.append(np.sum( [np.sum(np.abs(s)) for s in all_s] ) )
	l0.append(np.sum( [np.sum( np.max(np.abs(s), axis = 0)) for s in all_s] ) )

	new_s = np.array([s[3] for s in all_s])
	new_l2 = np.array(sess.run( l2_loss ))

	if i is not 0:
		dif_s = np.max( np.abs(new_s - old_s) )
		dif_l2 = np.max( new_l2 - old_l2 )

	old_s = new_s
	old_l2 = new_l2

	# cv2.imshow('heatmap', np.abs(new_s)/2. )
	# cv2.imshow('heatmap2', (np.abs(new_s)>.01).astype(np.float64))
	# cv2.waitKey(100)



	# print('------------%i-th iteration---------'%i)
	# print(new_s)
	# print(new_l2)
	# if i is not 0:
	# 	print(dif_s)
	# 	print(dif_l2)
	# print(np.array([s[3] for s in all_s]))
	# print(np.array(sess.run( l2_loss )))
plt.figure(1)
plt.plot(l1)
plt.figure(2)
plt.plot(l0)
# plt.legend(['l1','l0'])
plt.show()

	# print(sess.run(logstd))
	# print(sess.run(test_variable))
'''

saving_result = learn_agent.learn()
sess.close()

filename = dir_name + 'shelve_result'
myshelf = shelve.open(filename, 'n')
myshelf['saving_result'] = saving_result
myshelf.close()
'''
#actor.shared_var_list = [v for v in learned_var_list if np.any([name in v.name for name in inactive_module_name])]
