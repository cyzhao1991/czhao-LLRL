from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse
# from utils.krylov import cg
# from utils.utils import *
# from agent.agent import Agent

from actor.context_actor import Context_Gaussian_Actor
from agent.context_trpo import Context_TRPO_Agent
from baseline.baseline import BaselineZeros
from env.cartpole import CartPoleEnv
from model.context_net import Context_Fcnn_Net
from utils.paras import Paras_base

def main(gpu_num, exp_num, env = None, **kwargs):

	mod_num = kwargs.get('mod_num', 10)
	num_of_paths = kwargs.get('num_of_paths', 10)
	num_of_tasks = kwargs.get('num_of_tasks', 10)
	# task_num = kwarg.get('task_num', 0)
	# num_of_paths = kwarg.get('num_of_paths', 100)
	# dir_name = 'Data/checkpoint/'
	dir_name = '/disk/scratch/chenyang/Data/context_trpo/mod_%i_exp%i/'%(mod_num, exp_num)
	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)

	with open('log.txt', 'a') as text_file:
		text_file.write('gpu %i exp %i started.\n'%(gpu_num, exp_num))
	
	gravity_list = np.arange(0.2, 2.1, .2) * 9.8
	env_paras_list = [(g, 1.0, 0.1) for g in gravity_list]
	# random.shuffle(env_paras_list)
	env_paras_list = env_paras_list[0:num_of_tasks]
	env_list = []
	[env_list.append(CartPoleEnv(g, mc, mp)) for g,mc,mp in env_paras_list]
	num_of_envs = len(env_list)

	for env1,g in zip(env_list, env_paras_list):
		env1.context = np.array([1, g[0]], dtype = np.float64)


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
		pms.num_of_paths = num_of_paths
		pms.subsample_factor = 0.01
		pms.context_shape = 2
		
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.05
		config.gpu_options.allow_growth = True
		sess = tf.Session(config = config)

		actor_net = Context_Fcnn_Net(sess, pms.obs_shape, pms.action_shape, pms.context_shape, [100,50,25], [mod_num,mod_num,mod_num,mod_num],\
			name = pms.name_scope, if_bias = [False], activation = ['tanh', 'tanh', 'tanh', 'tanh'])
		actor = Context_Gaussian_Actor(actor_net, sess, pms)
		baseline = BaselineZeros(sess, pms)

		learn_agent = Context_TRPO_Agent(env_list, actor, baseline, sess, pms, [None])

	saver = tf.train.Saver()
	learn_agent.saver = saver
	sess.run(tf.global_variables_initializer())

	tf.get_default_graph().finalize()


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
	parser.add_argument('--mod', default = 10, type = int)
	parser.add_argument('--path', default = 10, type= int)
	args = vars(parser.parse_args())
	gpu_num = args['gpu']
	exp_num = args['exp']
	mod_num = args['mod']
	num_of_paths = args['path']
	main(gpu_num, exp_num, mod_num = mod_num, num_of_paths = num_of_paths)
