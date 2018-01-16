from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.krylov import cg
from utils.utils import *
from agent import Agent
from trpo import TRPOagent

class TRPO_MTLagent(TRPOagent):

	def __init__(self, env_list, actor, baseline, session, flags, saver = None):
		super(TRPO_MTLagent, self).__init__(env_list, actor, baseline, session, flags, saver)
		self.shared_paras = [v for v in self.actor.var_list if v.name.startswith(self.pms.name_scope + '_shared')]
		self.weight_paras = [v for v in self.actor.var_list if v.name.startswith(self.pms.name_scope + '_weight')]
		self.env_list = env_list
		self.task_context_mat = np.eye(len(self.env_list))

	def get_single_path(self, env_index = 0):

		observations = []
		actions = []
		rewards = []
		actor_infos = []
		state = self.env_list[env_index].reset()

		if self.pms.render:
			self.env_list[env_index].render()

		for _ in range(self.pms.max_time_step):
			action, actor_info = self.actor.get_action(state, contexts = self.task_context_mat[env_index])
			action = [action] if len(np.shape(action)) == 0 else action
			next_state, reward, terminal, _ = self.env_list[env_index].step(action)
			observations.append(state)
			actions.append(action)
			rewards.append(reward)
			actor_infos.append(actor_info)
			if terminal:
				break
			state = next_state
			if self.pms.render:
				self.env_list[env_index].render()
		path = dict(observations = np.array(observations), actions = np.array(actions), rewards = np.array(rewards), actor_infos = actor_infos)
		return path

	def get_paths(self, num_of_paths = None, prefix = '', verbose = True):
		if num_of_paths is None:
			num_of_paths = self.pms.num_of_paths
		paths = []
		t = time.time()
		if verbose:
			print(prefix+'Gathering Samples')
		for env_index in range(len(self.env_list)):
			paths.append([])
			for i in range(num_of_paths):
				paths[-1].append(self.get_single_path(env_index))
				if verbose:
					sys.stdout.write('%i-th path sampled. simulation time: %f \r'%(i + env_index*num_of_paths, time.time()-t))
					sys.stdout.flush()
		if verbose:
			print('%i paths sampled. Total time used: %f.'%(num_of_paths*len(self.env_list), time.time()-t))
		return paths

	def process_all_paths(self, all_paths):
		all_sample_data = []
		for paths, task_context in zip(all_paths, self.task_context_mat):
			all_sample_data.append(self.process_paths(paths))
			all_sample_data[-1]['contexts'] = np.array(np.tile(task_context, (all_sample_data[-1]['total_time_step'], 1)))
		return all_sample_data

	def train_paths(self, all_paths):
		all_sample_data = self.process_all_paths(all_paths)
		updated_flat_theta = []
		s_loss = []
		kl_div = []
		avg_rtn = []
		t_t_step = []
		for sample_data, paths in zip(all_sample_data,all_paths):
			obs_source = sample_data['observations']
			con_source = sample_data['contexts']
			act_source = sample_data['actions']
			adv_source = sample_data['advantages']
			n_samples = sample_data['total_time_step']
			actor_info_source = sample_data['actor_infos']
			# con_source = sample_data
			episode_rewards = np.array([np.sum(path['rewards']) for path in paths])
			train_number = int(1./self.pms.subsample_factor)
			step_gradients = []

			flat_theta_prev = self.sess.run(flatten_var(self.actor.var_list))

			for iteration in range(train_number):
				inds = np.random.choice(n_samples, int(np.floor(n_samples*self.pms.subsample_factor)), replace = False)
				obs_n = obs_source[inds]
				con_n = con_source[inds]
				act_n = act_source[inds]
				adv_n = adv_source[inds]
				act_dis_mean_n = np.array([a_info['mean'] for a_info in actor_info_source[inds]])
				act_dis_logstd_n = np.array([a_info['logstd'] for a_info in actor_info_source[inds]])

				feed_dict = {self.obs: obs_n,
							 self.contexts: con_n,
							 self.advant: adv_n,
							 self.action: act_n,
							 self.old_dist_mean: act_dis_mean_n[:,np.newaxis],
							 self.old_dist_logstd: act_dis_logstd_n[:,np.newaxis]
							 }

				def fisher_vector_product(p):
					feed_dict[self.flat_tangent] = p
					return self.sess.run(self.flat_fvp, feed_dict = feed_dict) + self.pms.cg_damping * p

				g = self.sess.run(self.flat_surr_grad, feed_dict = feed_dict)
				# print([g, np.amax(g), np.amin(g)])
				step_gradient = cg(fisher_vector_product, -g, cg_iters = self.pms.cg_iters)
				sAs = step_gradient.dot( fisher_vector_product(step_gradient) )
				inv_stepsize = np.sqrt( sAs/(2.*self.pms.max_kl) )
				fullstep_gradient = step_gradient / (inv_stepsize + 1e-8)

				def loss_function(x):
					self.sess.run(set_from_flat(self.actor.var_list, x))
					surr_loss, kl = self.sess.run([self.surr_loss, self.kl], feed_dict = feed_dict)
					# self.sess.run(set_from_flat(self.actor.var_list, flat_theta_prev))
					return surr_loss, kl
				if self.pms.linesearch:
					flat_theta_new = linesearch(loss_function, flat_theta_prev, fullstep_gradient, self.pms.max_backtracks, self.pms.max_kl)
				else:
					flat_theta_new = flat_theta_prev + fullstep_gradient
				self.sess.run(set_from_flat(self.actor.var_list, flat_theta_prev))
				step_gradients.append(flat_theta_new - flat_theta_prev)
			flat_theta_new = flat_theta_prev + np.nanmean(step_gradients, axis = 0)
			surrgate_loss, kl_divergence = loss_function(flat_theta_new)
			updated_flat_theta.append(flat_theta_new)
			s_loss.append(surrgate_loss)
			kl_div.append(kl_divergence)
			avg_rtn.append(np.mean(episode_rewards))
			t_t_step.append(n_samples)

		stats = dict(
			surrgate_loss = s_loss,
			kl_divergence = kl_div,
			average_return = avg_rtn,
			total_time_step = t_t_step
			)
		return updated_flat_theta, flat_theta_prev, stats


	def learn(self):
		'''
		saving_result = dict(
			average_return = [],
			sample_time = [],
			total_time_step = [],
			train_time = [],
			surrgate_loss = [],
			kl_divergence = [],
			iteration_number = []
			)
		'''
		dict_keys = ['average_return', 'sample_time', 'total_time_step', \
			'train_time', 'surrgate_loss', 'kl_divergence', 'iteration_number', 's_vector']
		saving_result = dict([(v, []) for v in dict_keys])

		for iter_num in range(self.pms.max_iter):
			print('\n******************* Iteration %i *******************'%iter_num)
			t = time.time()
			paths = self.get_paths()
			sample_time = time.time() - t
			t = time.time()
			theta, theta_old, stats = self.train_paths(paths)
			self.sess.run(set_from_flat(self.actor.var_list, np.mean(theta, axis = 0)))
			train_time = time.time() - t
			s_vector = self.sess.run(self.weight_paras)
			print(s_vector)
			for k, v in stats.iteritems():
				print("%-20s: %15.5f"%(k,np.mean(v)))

			save_value_list = [stats['average_return'], sample_time, stats['total_time_step'], \
				train_time, stats['surrgate_loss'], stats['kl_divergence'], iter_num, s_vector]

			[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]

			if self.pms.save_model and iter_num % self.pms.save_model_iters == 0:
				self.save_model(self.pms.save_dir + self.pms.env_name + '-iter%i'%(iter_num))

		return saving_result
