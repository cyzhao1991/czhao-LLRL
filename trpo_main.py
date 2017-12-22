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
from baseline.baseline import BaselineZeros
from env.cartpole import CartPoleEnv
from model.net import *
from utils.paras import Paras_base

def main(gpu_num, exp_num, env = None):
	dir_name = 'Data/checkpoint/'
	# dir_name = '/disk/scratch/chenyang/Data/trpo_stl/exp%i/'%(exp_num)
	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)

	with open('log.txt', 'a') as text_file:
		text_file.write('gpu %i exp %i started.\n'%(gpu_num, exp_num))

	with tf.device('/gpu:%i'%(gpu_num)):
		pms = Paras_base().pms
		pms.save_model = True
		pms.save_dir = dir_name
		env = CartPoleEnv() if env is None else env
		# env = gym.make('Pendulum-v0')
		action_size = env.action_space.shape[0]
		observation_size = env.observation_space.shape[0]
		max_action = env.action_space.high[0]
		pms.obs_shape = observation_size
		pms.action_shape = action_size
		pms.max_action = max_action
		pms.num_of_paths = 100
		
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.20
		sess = tf.Session(config = config)

		actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100,50,25], name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh', 'None', 'None'])
		actor = GaussianActor(actor_net, sess, pms)
		baseline = BaselineZeros(sess, pms)

		learn_agent = TRPOagent(env, actor, baseline, sess, pms, [None])

	saver = tf.train.Saver()
	learn_agent.saver = saver
	sess.run(tf.global_variables_initializer())
	saving_result = learn_agent.learn()

	sess.close()

	filename = dir_name + 'shelve_result'
	my_shelf = shelve.open(filename, 'n')
	my_shelf['saving_result'] = saving_result
	my_shelf.close()
	with open('log.txt', 'a') as text_file:
		text_file.write('gpu %i exp %i finished.\n'%(gpu_num, exp_num))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default = 0, type = int)
	parser.add_argument('--exp', default = 0, type = int)
	args = vars(parser.parse_args())
	gpu_num = args['gpu']
	exp_num = args['exp']
	main(gpu_num, exp_num)
