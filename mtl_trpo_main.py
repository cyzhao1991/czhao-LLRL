from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse, pdb
# from utils.krylov import cg
# from utils.utils import *
# from agent.agent import Agent

from actor.actor import GaussianActor
from agent.trpo_mtl import TRPO_MTLagent
from baseline.baseline import BaselineZeros
from env.cartpole import CartPoleEnv
from model.net import *
from utils.paras import Paras_base

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default = 0, type = int)
parser.add_argument('--exp', default = 0, type = int)
args = vars(parser.parse_args())
gpu_num = args['gpu']
exp_num = args['exp']

dir_name = '/home/chenyang/Documents/coding/Data/checkpoint/'
# dir_name = '/disk/scratch/chenyang/Data/trpo_mtl/gpu%iexp%i/'%(gpu_num, exp_num)
if not os.path.isdir(dir_name):
	os.mkdir(dir_name)

with open('log.txt', 'a') as text_file:
	text_file.write('gpu %i exp %i started.\n'%(gpu_num, exp_num))

gravity_list = [0., 4.9, 9.8]
mass_cart = [0.1, 0.5, 1.0]
mass_pole = [0.1, 0.5, 1.0]
env_paras_list = [(g, mc, mp) for g in gravity_list for mc in mass_cart for mp in mass_pole]
env_list = []
[env_list.append(CartPoleEnv(g, mc, mp)) for g,mc,mp in env_paras_list]
env = env_list[0]
with tf.device('/gpu:%i'%(gpu_num)):
	pms = Paras_base().pms
	# print(pms.max_iter)
	pms.save_model = False
	pms.save_dir = dir_name
	# pms.save_dir = '/home/chenyang/Documents/coding/Data/checkpoint/'
	action_size = env.action_space.shape[0]
	observation_size = env.observation_space.shape[0]
	max_action = env.action_space.high[0]
	pms.obs_shape = observation_size
	pms.action_shape = action_size
	pms.max_action = max_action
	pms.num_of_paths = 1000
	pms.with_context = True

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 1.
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)

	weight_net = Fcnn(sess, len(env_paras_list[0]), 30, [], name = 'weight_network', if_bias = [True])
	w_out = weight_net.output
	s_weights = [ tf.slice(w_out, [0, 0], [-1, 10]), tf.slice(w_out, [0, 10], [-1,10]), tf.slice(w_out, [0,20],[-1,10]) ]
	actor_mid_net = Modular_Fcnn(sess, pms.obs_shape, 25, [100,50], [10,10,10], name = pms.name_scope+'_shared', \
		if_bias = [False], activation_fns = ['tanh', 'tanh', 'tanh', 'None'], s_weights = s_weights)
	# actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [100,50,25], name = pms.name_scope, if_bias = [False], activation_fns = ['tanh', 'tanh', 'None', 'None'])

	final_layer_collection = []
	actor_collection = []
	for t in range(len(env_list)):
		tmp_name = pms.name_scope + '_task_%i'%t
		with tf.name_scope( tmp_name ):
			final_layer_collection.append( Net(sess, pms.obs_shape, pms.action_shape, [], name = tmp_name) )
			final_layer_collection[t].weights = tf.Variable( tf.truncated_normal([25, pms.action_shape], stddev = 1.), name = 'theta_%i'%(-1))
			final_layer_collection[t].input = actor_mid_net.input
			final_layer_collection[t].context_input = weight_net.input
			final_layer_collection[t].output = tf.matmul( actor_mid_net.output, final_layer_collection[t].weights )
			actor_collection.append( GaussianActor(final_layer_collection[t], sess, pms) )

	pdb.set_trace()
	# actor = GaussianActor(actor_net, sess, pms)
	baseline = [BaselineZeros(sess, pms) for _ in env_list]

	learn_agent = TRPO_MTLagent(env_list, actor_list, baseline_list, sess, pms)

	sess.run(tf.global_variables_initializer())

	saving_result = learn_agent.learn()
# filename = '/home/chenyang/Documents/coding/Data/checkpoint/shelve_result'
filename = dir_name + 'shelve_result'
my_shelf = shelve.open(filename, 'n')
my_shelf['saving_result'] = saving_result
my_shelf.close()
with open('log.txt', 'a') as text_file:
	text_file.write('gpu %i exp %i finished.\n'%(gpu_num, exp_num))
