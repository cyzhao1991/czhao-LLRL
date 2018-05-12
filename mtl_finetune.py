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
from agent.context_trpo import Context_TRPO_Agent

tf.reset_default_graph()

WIND = 3.
dir_name = 'Data/dm_control/finetune/mtl_walker_s1.0/w%1.1fg0.0/exp5/'%WIND
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
	pms.max_total_time_step = 50000
	pms.max_kl = 0.01
	pms.min_std = 0.01
	pms.max_iter = 500
	pms.subsample_factor = .1
	pms.max_time_step = 1000
	pms.env_name = 'walker'
	pms.train_flag = True
	pms.context_shape = 7
	pms.l1_regularizer= 0.002
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.1
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)

	# actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh','tanh', 'None'], init = [1., 1. ,1.,.01])
	actor_net = Context_Fcnn_Net(sess, pms.obs_shape, pms.action_shape, pms.context_shape, [100,50,25], [mod_num,mod_num,mod_num,mod_num],\
			name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh', 'tanh','None'], init = [.1, .1, .1, .01])
	mimic_agent = MtlMimicAgent(env, sess, pms, actor_net)

	sess.run(tf.global_variables_initializer())
# mimic_agent.learn(all_obs, all_con, all_acs)
saver = tf.train.Saver(max_to_keep = 100)
model_name = 'Data/mimic_data/multi_wind_mimic_0.ckpt'
# saver.save(sess, model_name)
saver.restore(sess, model_name)

learned_var_list = [v for v in tf.trainable_variables()]
s_var_list = [v for v in learned_var_list if 's_vector' in v.name]
learned_s = sess.run(s_var_list)
learned_s_nonzero = [np.abs(s) < 0.01 for s in learned_s]
zero_out_s = np.array([np.where(s_nz, 0, s) for s_nz, s in zip(learned_s_nonzero, learned_s)])
one_out_s = [np.where(s_nz, 1, s) for s_nz, s in zip(learned_s_nonzero, learned_s)]

m,n,l = zero_out_s.shape

inactive_module = [np.all(s, axis = 0) for s in learned_s_nonzero]
inactive_module_index = np.nonzero(inactive_module)
inactive_module_name = ['h%i_m%i'%(i,j) for i,j in zip(inactive_module_index[0], inactive_module_index[1])]
inactive_module_list = [v for v in learned_var_list if np.any([name in v.name for name in inactive_module_name])]

zero_one_s = np.array(zero_out_s)
for i,j in zip(inactive_module_index[0], inactive_module_index[1]):
	zero_one_s[i][:,j] = 1.

# for i, j in zip(inactive_module_index[0], inactive_module_index[1]):
# 	zero_out_s[i, :, j] = 1.
sess.run([tf.assign(s, zo_s) for s, zo_s in zip(s_var_list, zero_out_s)])
# sess.run([tf.assign(s, zo_s) for s, zo_s in zip(s_var_list, zero_one_s)])

# s_var_list = [tf.where(s_nz, s, tf.stop_gradient(s)) for s, s_nz in zip(s_var_list, learned_s_nonzero)]
# actor_net.c_weights = s_var_list
# actor_net.def_Task_knowledge(actor_net.name)
# actor_net.output = actor_net.build(actor_net.name)

# sess.run( [tf.assign(v, 0.001*np.random.rand(v.shape[0], v.shape[1]).astype(np.float32)) for v in inactive_module_list] )
pms.save_model_iters = 10

# value_to_assign = sess.run(tf.trainable_variables())

# tf.reset_default_graph()

# sess = tf.Session(config = config)
with tf.device('/gpu:%i'%(0)):
	# actor_net = Context_Fcnn_Net(sess, pms.obs_shape, pms.action_shape, pms.context_shape, [100,50,25], [mod_num,mod_num,mod_num,mod_num],\
	# 		name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh', 'tanh','None'], init = [.1, .1, .1, .01])
	# var_to_assign = tf.trainable_variables()
	# inactive_module_list = [v for v in learned_var_list if np.any([name in v.name for name in inactive_module_name])]

	test_variable = tf.Variable(np.array([1.]).astype(np.float32), tf.float32, name = 'test_variable')

	actor = Context_Gaussian_Actor(actor_net, sess, pms)
	baseline_net = Fcnn(sess, pms.obs_shape, 1, [100, 50, 25], name = 'baseline', if_bias = [False], activation = ['relu', 'relu','relu','None'], init = [.1, .1, .1, .1])
	baseline = BaselineFcnn(baseline_net, sess, pms)
	actor.shared_var_list = inactive_module_list
	# sess.run([var.initializer for var in actor.shared_var_list])
	actor.shared_var_list.append(actor.action_logstd)
	# actor.task_var_list = s_var_list
	actor.var_list = actor.shared_var_list + actor.task_var_list
	learn_agent = Context_TRPO_Agent(env,actor, baseline,sess, pms, [None], env_contexts =np.array([[0,0,0,1,0,0,0]]))

# sess.run(tf.global_variables_initializer())
# sess.run([tf.assign(var,np.array(val).astype(np.float32)) for var,val in zip(var_to_assign, value_to_assign)])
all_var_list = tf.global_variables()
all_var_initialized = sess.run([tf.is_variable_initialized(v) for v in all_var_list])
not_initialized_vars = [v for v, f in zip(all_var_list, all_var_initialized) if not f]
sess.run(tf.variables_initializer(not_initialized_vars))
saver = tf.train.Saver(all_var_list, max_to_keep = 100)
# new_saver = tf.train.Saver(var_list = all_var_list, max_to_keep = 100, saver_def = saver.as_saver_def())
learn_agent.saver = saver
#sess.run([v.initializer for v in all_var_list[-5:]])


saving_result = learn_agent.learn()
sess.close()

filename = dir_name + 'shelve_result'
myshelf = shelve.open(filename, 'n')
myshelf['saving_result'] = saving_result
myshelf.close()

#actor.shared_var_list = [v for v in learned_var_list if np.any([name in v.name for name in inactive_module_name])]