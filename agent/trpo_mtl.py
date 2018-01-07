from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.krylov import cg
from utils.utils import *
from agent import Agent
from trpo import TRPOAgent

class TRPO_MTLagent(Agent):

	def __init__(self, env_list, actor_list, baseline_list, session, flags, saver = None):
		super(TRPO_MTLagent, self).__init__(env_list, actor_list, baseline_list, session, flags, saver)

		self.init_vars()

		print('Building Network')

	def init_vars(self):
		name_scope = self.pms.name_scope
		self.agent_list = []
		for i in range(len(env_list)):
			suffix = 'task_%i'%i
			self.pms.name_scope = name_scope+suffix
			self.agent_list.append( TRPOAgent(env_list[i], actor_list[i], baseline_list[i], sess, self.pms, self.saver) )

			# with tf.name_scope(self.pms.name_scope + suffix):
			# 	obs = self.actor_list[i].input_ph
			# 	context = self.actor_list[i].net.context_input
			# 	advant = tf.placeholder(tf.float32, [None], name = 'advantages')
			# 	action = tf.placeholder(tf.float32, [None, self.pms.action_shape])
			# 	old_dist_mean = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = )

	def get_all_paths(self, num_of_paths = None, prefix = None, verbose = True):
		all_paths = []
		t = time.time()
		task_num = 0
		if verbose:
			print(prefix + 'Gathering Samples')
		for trpo_a in self.agent_list:
			tmp_t = time.time()
			all_paths.append(self.get_paths(num_of_paths, verbose = False))
			if verbose:
				sys.stdout.write('%i-th task sampled. simulation time: %f \r'%(task_num, time.time()-tmp_t))
				sys.stdout.flush()
			task_num += 1
		if verbose:
			print('%i tasks sampled. Total time used: %f.'%(task_num, time.time()-t))
		return all_paths

	def process_paths(self, all_paths):
		return [trpo_a.process_paths(paths) for trpo_a, paths in zip(self.agent_list, all_paths)]

	def gradients_shared(self, all_sample_data):
		var_list = [v ]

		pass
	
	def gradients_task(self, sample_data, task_index):
		pass


	def train_paths_shared(self, paths_list):
		pass

	def train_paths_task(self, paths, task_index):
		sample_data = self.agent_list[task_index].process_paths(paths)
		obs_source = sample_data['observations']
		act_source = sample_data['actions']
		adv_source = sample_data['advantages']
		n_samples = sample_data['total_time_step']
		actor_info_source = sample_data['actor_infos']
		
		episode_rewards = np.array([np.sum(path['rewards']) for path in paths])
		train_number = int(1./self.pms.subsample_factor)
		step_gradients = []

		flat_theta_prev = self.sess.run(flatten_var(self.actor.var_list))



	def learn(self):

		for iter_num in range(self.pms.max_iter):
			print('\n******************* Iteration %i *******************'%iter_num)
			t = time.time()
			paths = self.get_all_paths()
			sample_time = time.time() - t
			t = time.time()
