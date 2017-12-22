from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve, argparse

from actor.actor import DeterministicActor
from agent.ddpg import DDPGagent
from utils.replay_buffer import ReplayBuffer
from utils.ounoise import OUNoise
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
		action_size = env.action_space.shape[0]
		observation_size = env.observation_space.shape[0]
		max_action = env.action_space.high[0]
		pms.obs_shape = observation_size
		pms.max_iter = 1000000
		pms.action_shape = action_size
		pms.max_action = max_action
		pms.num_of_paths = 100
		pms.name_scope = 'ddpg'
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.20
		sess = tf.Session(config = config)

		state_ph = tf.placeholder(tf.float32, [None, pms.obs_shape])
		action_ph = tf.placeholder(tf.float32, [None, pms.action_shape])
		critic_input_ph = tf.concat([state_ph, action_ph], axis = 1)
		actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [400, 300], name = pms.name_scope+'_actor_r', if_bias = [False], activation = ['relu', 'relu', 'None'], input_tf = state_ph)
		actor_target_net = Fcnn(sess, pms.obs_shape, pms.action_shape, [400, 300], name = pms.name_scope+'_actor_t', if_bias = [False], activation = ['relu', 'relu', 'None'], input_tf = state_ph)
		critic_net = Fcnn(sess, pms.obs_shape+pms.action_shape, 1, [400, 300], name = pms.name_scope+'_critic_r', if_bias = [False], activation = ['relu', 'relu', 'None'], input_tf = critic_input_ph)
		critic_target_net = Fcnn(sess, pms.obs_shape+pms.action_shape, 1, [400, 300], name = pms.name_scope+'_critic_t', if_bias = [False], activation = ['relu', 'relu', 'None'], input_tf = critic_input_ph)
		critic_net.state_ph = state_ph
		critic_net.action_ph = action_ph
		print('sth')
		actor = DeterministicActor(actor_net, sess, pms)
		actor_target = DeterministicActor(actor_net, sess, pms)

		replay_buffer = ReplayBuffer(buffer_size = pms.buffer_size)
		ounoise = OUNoise(pms.action_shape)
		learn_agent = DDPGagent(env, actor, critic_net, actor_target, critic_target_net, replay_buffer, ounoise, sess, pms, [None])
	
	saver = tf.train.Saver()
	learn_agent.saver = saver
	sess.run(tf.global_variables_initializer())
	saving_result = learn_agent.learn()
	sess.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default = 0, type = int)
	parser.add_argument('--exp', default = 0, type = int)
	args = vars(parser.parse_args())
	gpu_num = args['gpu']
	exp_num = args['exp']
	main(gpu_num, exp_num)