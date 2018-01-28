from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse, pdb, random

from model.mtl_net import Mtl_Fcnn_Net
from actor.mtl_actor import Mtl_Gaussian_Actor
from utils.paras import Paras_base
from agent.trpo_mtl_v1 import TRPO_MTLagent
from baseline.baseline import BaselineZeros
from env.cartpole import CartPoleEnv

def main(gpu_num, exp_num, env = None, **kwarg):

	# dir_name = '/home/chenyang/Documents/coding/Data/checkpoint/'
	dir_name = '/disk/scratch/chenyang/Data/trpo_mtl_v1/mod%i_exp%i/'%(mod_num, exp_num)
	if not os.path.isdir(dir_name):
		os.mkdir(dir_name)

	with open('log.txt', 'a') as text_file:
		text_file.write('mtl gpu %i exp %i started.\n'%(gpu_num, exp_num))

	# gravity_list = [0., 4.9, 9.8]
	# mass_cart = [0.1, 0.5, 1.0]
	# mass_pole = [0.1, 0.5, 1.0]
	# env_paras_list = [(g, mc, mp) for g in gravity_list for mc in mass_cart for mp in mass_pole]
	# random.shuffle(env_paras_list)
	# env_paras_list = env_paras_list[:10]
	# env_list = []
	# [env_list.append(CartPoleEnv(g, mc, mp)) for g,mc,mp in env_paras_list]
	# random.shuffle(env_list)
	# env_list = env_list[:10]
	mod_num = kwargs.get('mod_num', 10)
	num_of_paths = kwargs.get('num_of_paths', 100)
	num_of_tasks = kwargs.get('num_of_tasks', 10)
	
	gravity_list = np.arange(0.2, 2.1, .2) * 9.8
	env_paras_list = [(g, 1.0, 0.1) for g in gravity_list]
	random.shuffle(env_paras_list)
	env_paras_list = env_paras_list[0:num_of_tasks]
	env_list = []
	[env_list.append(CartPoleEnv(g, mc, mp)) for g,mc,mp in env_paras_list]
	num_of_envs = len(env_list)

	with tf.device('/gpu:%i'%(gpu_num)):
		pms = Paras_base().pms
		# print(pms.max_iter)
		pms.save_model = True
		pms.save_dir = dir_name
		# pms.save_dir = '/home/chenyang/Documents/coding/Data/checkpoint/'
		action_size = env_list[0].action_space.shape[0]
		observation_size = env_list[0].observation_space.shape[0]
		max_action = env_list[0].action_space.high[0]
		pms.obs_shape = observation_size
		pms.action_shape = action_size
		pms.max_action = max_action
		pms.num_of_paths = num_of_paths
		pms.with_context = False
		pms.name_scope = 'mtl_trpo'

		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.2
		config.gpu_options.allow_growth = True
		sess = tf.Session(config = config)

		# weight_net = Mtl_Fcnn_Net(sess, 30, [], name = pms.name_scope+'_weight', if_bias = [True])
		# w_out = weight_net.output
		# s_weights = [ tf.slice(w_out, [0, 0], [1, 10]), tf.slice(w_out, [0, 10], [1,10]), tf.slice(w_out, [0,20],[1,10]), weight_net.input[0] ]
		actor_net = Mtl_Fcnn_Net(sess, pms.obs_shape,  pms.action_shape, [100,50,25], [mod_num,mod_num,mod_num], num_of_envs, name = pms.name_scope, \
			if_bias = [False], activation_fns = ['tanh', 'tanh', 'tanh', 'tanh'])
		# actor_net.context = weight_net.input
		# actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100,50,25], name = pms.name_scope, if_bias = [False], activation_fns = ['tanh', 'tanh', 'None', 'None'])
		# pdb.set_trace()
		# final_layer_collection = []
		# actor_collection = []
		# for t in range(len(env_list)):
		# 	tmp_name = pms.name_scope + '_task_%i'%t
		# 	with tf.name_scope( tmp_name ):
		# 		final_layer_collection.append( Net(sess, pms.obs_shape, pms.action_shape, [], name = tmp_name) )
		# 		final_layer_collection[t].weights = tf.Variable( tf.truncated_normal([25, pms.action_shape], stddev = 1.), name = 'theta_%i'%(-1))
		# 		final_layer_collection[t].input = actor_mid_net.input
		# 		final_layer_collection[t].context = weight_net.input
		# 		final_layer_collection[t].output = tf.matmul( actor_mid_net.output, final_layer_collection[t].weights )
		# 		actor_collection.append( GaussianActor(final_layer_collection[t], sess, pms) )
		# actor = GaussianActor(actor_net, sess, pms)
		# baseline = [BaselineZeros(sess, pms) for _ in env_list]
		actor = Mtl_Gaussian_Actor(actor_net, sess, pms, num_of_envs)
		baseline = BaselineZeros(sess, pms)
		
		learn_agent = TRPO_MTLagent(env_list, actor, baseline, sess, pms, [None])

	saver = tf.train.Saver()
	learn_agent.saver = saver

	with tf.device('/gpu:%i'%(gpu_num)):
		sess.run(tf.global_variables_initializer())
		tf.get_default_graph().finalize()

		saving_result = learn_agent.learn()
	# filename = '/home/chenyang/Documents/coding/Data/checkpoint/shelve_result'
	filename = dir_name + 'shelve_result'
	my_shelf = shelve.open(filename, 'n')
	my_shelf['saving_result'] = saving_result
	# my_shelf['env_list'] = env_list
	my_shelf['env_paras_list'] = env_paras_list
	my_shelf.close()
	with open('log.txt', 'a') as text_file:
		text_file.write('mtl gpu %i exp %i finished.\n'%(gpu_num, exp_num))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default = 0, type = int)
	parser.add_argument('--exp', default = 0, type = int)
	parser.add_argument('--mod', default = 10, type = int)
	args = vars(parser.parse_args())
	gpu_num = args['gpu']
	exp_num = args['exp']
	mod_num = args['mod']
	main(gpu_num, exp_num, mod_num = mod_num)
