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
# from agent.mimic_agent import *
from actor.actor import GaussianActor
# from actor.mtl_actor import MtlGaussianActor
from model.mtl_net import MtlFcnnNet
from agent.trpo import TRPOagent

def main(gpu_num, exp_num, speed = None, **kwargs):

	tf.reset_default_graph()
	wind = 0.
	dir_name = '/disk/scratch/chenyang/new_Data/dm_control/tl_prog/walker_s%1.1f/w%1.1fg0.0/exp%i/'%(speed,wind,exp_num)
	# dir_name = 'new_Data/dm_control/stl_ft/walker_s%1.1f/w%1.1fg0.0/exp%i/'%(speed, wind, exp_num)

	# dir_name = '/disk/scratch/chenyang/Data/dm_control/mtl_prog/walker_s%1.1f/w%1.1fg0.0/exp%i'%(SPEED, WIND, exp_num)
	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)
	env = Walker2dEnv()
	env.reward_type = 'bound'
	env.target_value = SPEED
	default_context = np.array([0., 0., -9.8])
	env.model.opt.gravity[0] = 0.
	# env.model.opt.gravity[2] += default_context[2] - 5.
	mod_num = 1

	act_size = env.action_space.shape[0]
	max_action = env.action_space.high
	obs_size = env.observation_space.shape[0]

	with tf.device('/gpu:%i'%(gpu_num)):
		pms = Paras_base().pms
		pms.save_model = True
		pms.save_dir = dir_name
		pms.obs_shape = obs_size
		pms.action_shape = act_size
		pms.max_action = max_action
		pms.max_total_time_step = 10000
		pms.max_kl = 0.01
		pms.min_std = 0.01
		pms.max_iter = 501
		pms.subsample_factor = .1
		pms.max_time_step = 1000
		pms.env_name = 'walker'
		pms.train_flag = True
		# pms.context_shape = 7
		pms.l1_regularizer= 0.
		pms.l1_column_reg = 0.
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.per_process_gpu_memory_fraction = 0.1
		config.gpu_options.allow_growth = True
		sess = tf.Session(config = config)

		actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], name = pms.name_scope, if_bias = [True], activation = ['tanh', 'tanh','tanh', 'None'], init = [1., 1. ,1.,.01])

	saver = tf.train.Saver(max_to_keep = 100)
	model_name = '/disk/scratch/chenyang/new_Data/dm_control/stl/walker_s1.0/w0.0g0.0/exp0/walker-iter1000.ckpt'
	# model_name = 'new_Data/dm_control/stl/walker_s3.0/w0.0g0.0/exp0/walker-iter1000.ckpt'

	saver.restore(sess, model_name)

	learned_var_list = [v for v in tf.trainable_variables()]
	pre_learned_var = sess.run(learned_var_list)
	sess.close()
	tf.reset_default_graph()
	with tf.device('/gpu:%i'%(gpu_num)):
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.per_process_gpu_memory_fraction = .1
		config.gpu_options.allow_growth = True
		sess = tf.Session(config = config)

		actor_net = MtlFcnnNet(sess, pms.obs_shape, pms.action_shape, [100, 50, 25], [mod_num,mod_num,mod_num,mod_num], 1, name = pms.name_scope, \
			if_bias = [True], activation = ['tanh', 'tanh','tanh', 'None'], init = [.1, .1 ,.1,.01])
		actor_net.output = actor_net.output[0]
		# actor = MtlGaussianActor(actor_net, sess, pms)
		actor = GaussianActor(actor_net, sess, pms)

		baseline_net = Fcnn(sess, pms.obs_shape, 1, [100, 50, 25], name = 'baseline', if_bias = [True], \
			activation = ['relu', 'relu','relu','None'], init = [.1, .1, .1, .1]) 
		baseline = BaselineFcnn(baseline_net, sess, pms)

		learn_agent = TRPOagent(env, actor, baseline, sess, pms, saver = None)
		learn_agent.var_list = [v for v in tf.trainable_variables() if 'shared' not in v.name and 'baseline' not in v.name]
		learn_agent.init_vars()
		learn_agent.boost_baseline = True
		sess.run(tf.global_variables_initializer())
	shared_w_varlist = [v for v in tf.trainable_variables()]
	sess.run([tf.assign(var, value) for var, value in zip(shared_w_varlist, pre_learned_var)])
	sess.run([tf.assign(var, np.array([1., .1]).astype(np.float32) ) for var in learn_agent.var_list if 'task_path' in var.name])

	saver = tf.train.Saver(max_to_keep = 101)
	learn_agent.saver = saver

	saving_result = learn_agent.learn()
	sess.close()

	filename = dir_name + 'shelve_result'
	myshelf = shelve.open(filename, 'n')
	myshelf['saving_result'] = saving_result
	myshelf.close()


# # if __name__ == '__main__':
# # 	main()
