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
# from env.cartpole import CartPoleEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from model.net import *
from utils.paras import Paras_base

def main(gpu_num, exp_num, env = None, **kwargs):

	task_num = kwargs.get('task_num', 0)
	num_of_paths = kwargs.get('num_of_paths', 10)
	dir_name = 'Data/checkpoint/stl/%s_exp%i/'%('half_cheetah',exp_num)
	# dir_name = '/disk/scratch/chenyang/Data/trpo_stl/task_%i_exp%i/'%(task_num, exp_num)
	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)

	# with open('log.txt', 'a') as text_file:
	# 	text_file.write('gpu %i exp %i started.\n'%(gpu_num, exp_num))


	# while True:
	# 	goal = np.random.rand(1,2) * 0.4 - 0.2
	# 	if np.linalg.norm(goal) < 0.2:
	# 		break

	env = HalfCheetahEnv()


	with tf.device('/gpu:%i'%(gpu_num)):
		pms = Paras_base().pms
		pms.save_model = True
		pms.save_dir = dir_name
		env = Walker2dEnv() if env is None else env
		# env = gym.make('Pendulum-v0')
		action_size = env.action_space.shape[0]
		observation_size = env.observation_space.shape[0]
		max_action = env.action_space.high[0]
		pms.obs_shape = observation_size
		pms.action_shape = action_size
		pms.max_action = max_action
		pms.num_of_paths = num_of_paths
		pms.max_iter = 200
		pms.max_time_step = 1000
		pms.subsample_factor = .1
		pms.max_kl = 0.01
		pms.min_std = 0.1
		pms.env_name = 'half_cheetah'
		pms.max_total_time_step = 5000
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.per_process_gpu_memory_fraction = 0.1
		config.gpu_options.allow_growth = True
		sess = tf.Session(config = config)

		actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [64,64], name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh', 'None'], init = [1. ,1., .01])
		actor = GaussianActor(actor_net, sess, pms)

		baselinet_net = Fcnn(sess, pms.obs_shape, 1, [64,64], name = 'baseline', if_bias = [False], activation = ['tanh','tanh','None'], init = [1. ,1., .1])
		# baseline = BaselineZeros(sess, pms)
		baseline = BaselineFcnn(baselinet_net, sess, pms)

		learn_agent = TRPOagent(env, actor, baseline, sess, pms, [None], goal = None)

	saver = tf.train.Saver()
	learn_agent.saver = saver
	sess.run(tf.global_variables_initializer())
	saving_result = learn_agent.learn()

	sess.close()

	filename = dir_name + 'shelve_result'
	my_shelf = shelve.open(filename, 'n')
	my_shelf['saving_result'] = saving_result
	my_shelf['goal'] = goal
	my_shelf.close()
	# with open('log.txt', 'a') as text_file:
	# 	text_file.write('gpu %i exp %i finished.\n'%(gpu_num, exp_num))

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
