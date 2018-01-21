from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.krylov import cg
from utils.utils import *
from agent import Agent

class TRPO_MTLagent(Agent):

	def __init__(self, env_list, mtl_actor, mtl_baseline, session, pms, saver = None):
		super(TRPO_MTLagent, self).__init__(env_list, session, pms, saver)
		self.actor = mtl_actor
		self.baseline = mtl_baseline
		assert( len(env_list) == self.actor.num_of_tasks )
		self.num_of_tasks = len(env_list)

		self.var_list = self.actor.var_list
		self.shared_var_list = self.actor.shared_var_list
		self.task_var_list = self.actor.task_var_list

		# self.shared_var_num = np.sum([])

		self.init_vars()
		print('Building Network')

	def init_vars(self):
		with tf.name_scope(self.pms.name_scope):
			self.obs = self.actor.input_ph
			self.advant = tf.placeholder(tf.float32, [None], name = 'advantages')
			self.action = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'action')
			self.old_dist_mean = tf.placeholder( tf.float32, [None, self.pms.action_shape], name = 'old_dist_mean')
			self.old_dist_logstd = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_logstd')

			self.new_dist_mean_list = self.actor.output_net
			self.new_dist_logstd_list = self.actor.action_logstd
			new_dist_info = zip(self.new_dist_mean_list, self.new_dist_logstd_list)

			# list of new logli
			logli_new_list = [log_likelihood(self.action, new_mean, new_logstd) for new_mean, new_logstd in new_dist_info]
			# scaler of old logli
			logli_old = log_likelihood(self.action, self.old_dist_mean, self.old_dist_logstd)
			batchsize = tf.cast(tf.shape(self.obs)[0], tf.float32)
			self.flat_tangent = tf.placeholder(tf.float32, shape = [None], name = 'flat_tangent')

			self.ratio_list = [tf.exp(ln - logli_old) for ln in logli_new_list]
			self.surr_loss_list = [-tf.reduce_mean(ratio * self.advant) for ratio in self.ratio_list]
			self.kl_list = [tf.reduce_mean(kl_sym(self.old_dist_mean, self.old_dist_logstd, new_mean, new_logstd)) for new_mean, new_logstd in new_dist_info]
			kl_firstfixed_list = [kl_sym_firstfixed(new_mean, new_logstd) for new_mean, new_logstd in new_dist_info]

			# gradients computed for shared & task weights separate
			# surr_grad_shared_list = [tf.gradients(surr_loss, self.shared_var_list) for surr_loss in self.surr_loss_list]
			# self.flat_surr_grad_shared_list = map(flatten_var, surr_grad_shared_list) #[flatten_var(surr_grad) for surr_grad in surr_grad_shared_list]
			# grads_shared_list = [tf.gradients(kl, self.shared_var_list) for kl in kl_firstfixed_list]
			# flat_grad_shared_list = map(flatten_var, grads_shared_list) #[flatten_var(grads) / batchsize for grads in grads_shared_list]
			# self.fvp_shared_list = [ tf.gradients(tf.reduce_sum(flat_g * self.flat_tangent / batchsize), self.shared_var_list) for flat_g in flat_grad_shared_list]
			# self.flat_fvp_shared_list = map(flatten_var, self.fvp_shared_list)

			# surr_grad_task_list = [tf.gradients(surr_loss, task_var) for surr_loss, task_var in zip(self.surr_loss_list, self.task_var_list)]
			# self.flat_surr_grad_task_list = map(flatten_var, surr_grad_task_list)
			# grads_task_list = [tf.gradients(kl, task_var) for kl, task_var in zip(kl_firstfixed_list, self.task_var_list)]
			# flat_grad_task_list = map(flatten_var, grads_task_list)
			# self.fvp_task_list = [tf.gradients(tf.reduce_sum( flat_g * self.flat_tangent / batchsize), task_var) for flat_g, task_var in zip(flat_grad_task_list, self.task_var_list)]
			# self.flat_fvp_task_list = map(flatten_var, self.fvp_task_list)

			# gradients computed for shared & task weights at same time
			surr_grad_task_list = [tf.gradients(surr_loss, self.shared_var_list + task_var) for surr_loss, task_var in zip(self.surr_loss_list, self.task_var_list)]
			self.flat_surr_grad_task_list = map(flatten_var, surr_grad_task_list)
			grads_task_list = [tf.gradients(kl, self.shared_var_list + task_var) for kl, task_var in zip(kl_firstfixed_list, self.task_var_list)]
			flat_grad_task_list = map(flatten_var, grads_task_list)
			self.fvp_task_list = [tf.gradients(tf.reduce_sum( flat_g * self.flat_tangent / batchsize), self.shared_var_list + task_var) for flat_g, task_var in zip(flat_grad_task_list, self.task_var_list)]
			self.flat_fvp_task_list = map(flatten_var, self.fvp_task_list)


	def get_single_path(self, task_index):
		observations = []
		actions = []
		rewards = []
		actor_infos = []
		state = self.env_list[task_index].reset()

		if self.pms.render:
			self.env_list[task_index].render()

		for _ in range(self.pms.max_time_step):
			action, actor_info = self.actor.get_action(state, task_index)
			action = [action] if len(np.shape(action)) == 0 else action
			next_state, reward, terminal, _ = self.env_list[task_index].step(action)
			observations.append(state)
			actions.append(action)
			rewards.append(reward)
			actor_infos.append(actor_info)
			if terminal:
				break
			state = next_state
			if self.pms.render:
				self.env_list[task_index].render()
		path = dict(observations = np.array(observations), actions = np.array(actions), rewards = np.array(rewards), actor_infos = actor_infos)
		return path

	def get_paths(self, num_of_paths = None, prefix = '', verbose = True):
		if num_of_paths is None:
			num_of_paths = self.pms.num_of_paths
		paths = []
		t = time.time()
		if verbose:
			print(prefix + 'Gathering Samples')
		for task_index in range(self.num_of_tasks):
			paths.append([])
			for i in range(num_of_paths):
				path[task_index].append(self.get_single_path( task_index ))
				if verbose:
					sys.stdout.write('%i-th path sampled. simulation time: %f \r'%(i + env_index*num_of_paths, time.time()-t))
					sys.stdout.flush()
		if verbose:
			print('All paths sampled. Total sampled paths: %i. Total time usesd: %f.'%(num_of_paths * self.num_of_tasks), time.time() - t)
		return paths

	def process_paths(self, paths):
		total_time_step = 0
		for path in paths:
			total_time_step += len(path['rewards'])
			path['baselines'] = self.baseline.predict(path)
			path['returns'] = discount(path['rewards'], self.pms.discount)
			if not self.pms.gae_flag:
				path['advantages'] = path['returns'] - path['baselines']
			else:
				b = np.append(path['baselines'], path['baselines'][-1])
				deltas = path['rewards'] + self.pms.discount * b[1:] - b[:-1]
				path['advantages'] = discount(deltas, self.pms.discount * self.pms.gae_lambda)

		observations = np.concatenate([path['observations'] for path in paths])
		actions = np.concatenate([path['actions'] for path in paths])
		rewards = np.concatenate([path['rewards'] for path in paths])
		advantages = np.concatenate([path['advantages'] for path in paths])
		actor_infos = np.concatenate([path['actor_infos'] for path in paths])
		if self.pms.center_adv:
			advantages -= np.mean(advantages)
			advantages /= (np.std(advantages) + 1e-8)

		sample_data = dict(
			observations = observations,
			actions = actions,
			rewards = rewards,
			advantages = advantages,
			actor_infos = actor_infos,
			paths = paths,
			total_time_step = total_time_step
		)

		self.baseline.fit(paths)

		return sample_data

	def train_paths(self, all_paths):
		updated_flat_theta = []
		s_loss = []
		kl_div = []
		avg_rtn = []
		t_t_step = []
		all_sample_data = []
		shared_var_num = np.sum( self.sess.run([tf.size(v) for v in self.shared_var_list]) )

		for task_i, paths in enumerate(all_paths):
			sample_data = self.process_paths(paths)
			all_sample_data.append(sample_data)
			obs_source = sample_data['observations']
			act_source = sample_data['actions']
			adv_source = sample_data['advantages']
			n_samples = sample_data['total_time_step']
			actor_info_source = sample_data['actor_infos']
			
			episode_rewards = np.array([np.sum(path['rewards']) for path in paths])
			train_number = int(1./self.pms.subsample_factor)
			step_gradients = []

			avg_rtn.append(np.mean(episode_rewards))
			t_t_step.append(n_samples)

			flat_theta_prev = self.sess.run(flatten_var(self.shared_var_list + self.task_var_list[task_i]))

			for iteration in range(train_number):
				inds = np.random.choice(n_samples, int(np.floor(n_samples*self.pms.subsample_factor)), replace = False)
				obs_n = obs_source[inds]
				act_n = act_source[inds]
				adv_n = adv_source[inds]
				act_dis_mean_n = np.array([a_info['mean'] for a_info in actor_info_source[inds]])
				act_dis_logstd_n = np.array([a_info['logstd'] for a_info in actor_info_source[inds]])

				feed_dict = {self.obs: obs_n,
							 self.advant: adv_n,
							 self.action: act_n,
							 self.old_dist_mean: act_dis_mean_n[:,np.newaxis],
							 self.old_dist_logstd: act_dis_logstd_n[:,np.newaxis]
							 }

				# def fisher_vector_product(p, task_index, shared = True):
				# 	feed_dict[self.flat_tangent] = p
				# 	if shared:
				# 		return self.sess.run(self.flat_fvp_shared_list[task_index], feed_dict = feed_dict) + self.pms.cg_damping * p
				# 	else:
				# 		return self.sess.run(self.flat_fvp_task_list[task_index], feed_dict = feed_dict) + self.pms.cg_damping * p

				# g_shared = self.sess.run(self.flat_surr_grad_shared_list[task_index], feed_dict = feed_dict)
				# fisher_shared_fn = lambda p: fisher_vector_product(p, task_index, True)
				# step_gradients = cg
			
				def fisher_vector_product(p, task_index):
					feed_dict[self.flat_tangent]= p 
					return self.sess.run(self.flat_fvp_task_list[task_index], feed_dict = feed_dict) + self.pms.cg_damping * p

				g = self.sess.run(self.flat_surr_grad_task_list[task_i], feed_dict = feed_dict)
				fisher_fn = lambda p: fisher_vector_product(p, task_i)
				step_gradient = cg(fisher_fn, -g, cg_iters = self.pms.cg_iters)
				sAs = step_gradient.dot( fisher_vector_product(step_gradient, task_i) )
				inv_stepsize = np.sqrt(sAs/ (2.*self.pms.max_kl))
				fullstep_gradient = step_gradient / (inv_stepsize + 1e-8)

				def loss_function(x, task_index):
					self.sess.run(set_from_flat(self.shared_var_list+self.task_var_list[task_index], x))
					surr_loss, kl = self.sess.run( [self.surr_loss_list[task_index], self.kl_list[task_index]], feed_dict = feed_dict )
					return surr_loss, kl

				loss_fn = lambda x: loss_function(x, task_i)
				if self.pms.linesearch:
					flat_theta_new = linesearch(loss_fn, flat_theta_prev, fullstep_gradient, self.pms.max_backtracks, self.pms.max_kl)
				else:
					flat_theta_new = flat_theta_prev + fullstep_gradient

				self.sess.run( set_from_flat(self.shared_var_list+self.task_var_list[task_i], flat_theta_prev) )
				step_gradients.append(flat_theta_new - flat_theta_prev)

			updated_flat_theta.append(flat_theta_new)
			

		new_shared_flat_theta = np.mean([v[:shared_var_num] for v in updated_flat_theta], axis = 0)
		self.sess.run( set_from_flat(self.shared_var_list, new_shared_flat_theta) )
		self.sess.run( [set_from_flat( task_var, theta[shared_var_num:] ) for task_var, theta in zip(self.task_var_list, updated_flat_theta)] )

		for i, sample_data in enumerate(all_sample_data):
			act_dis_mean_n = np.array(a_info['mean'] for a_info in sample_data['actor_infos'])
			act_dis_logstd_n = np.array(a_info['logstd'] for a_info in sample_data['actor_infos'])
			feed_dict = {self.obs: sample_data['observations'],
						 self.advant: sample_data['advantages'],
						 self.action: sample_data['actions'],
						 self.old_dist_mean: act_dis_mean_n[:,np.newaxis],
						 self.old_dist_logstd: act_dis_logstd_n[:,np.newaxis]
						 }
			sl, kl = self.sess.run([self.surr_loss_list[i], self.kl_list[i]], feed_dict = feed_dict)

			s_loss.append(sl)
			kl_div.append(kl)

		stats = dict(
			surrgate_loss = s_loss,
			kl_divergence = kl_div,
			average_return = avg_rtn,
			total_time_step = t_t_step
			)
		return updated_flat_theta, flat_theta_prev, stats

	def learn(self):
		dict_keys = ['average_return', 'sample_time', 'total_time_step', \
			'train_time', 'surrgate_loss', 'kl_divergence', 'iteration_number', 's_vector']
		saving_result = dict([(v, []) for v in dict_keys])

		s_vector_var_list = [v for v in self.var_list if 's_vector' in v.name]

		for iter_num in range(self.pms.max_iter):
			print('\n******************* Iteration %i *******************'%iter_num)
			t = time.time()
			paths = self.get_paths()
			sample_time = time.time() - t
			t = time.time()
			theta, theta_old, stats = self.train_paths(paths)
			# self.sess.run(set_from_flat(self.actor.var_list, np.mean(theta, axis = 0)))
			train_time = time.time() - t
			s_vector = self.sess.run( s_vector_var_list )
			for k, v in stats.iteritems():
				print("%-20s: %15.5f"%(k,np.mean(v)))

			save_value_list = [stats['average_return'], sample_time, stats['total_time_step'], \
				train_time, stats['surrgate_loss'], stats['kl_divergence'], iter_num, s_vector]

			[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]

			if self.pms.save_model and iter_num % self.pms.save_model_iters == 0:
				self.save_model(self.pms.save_dir + self.pms.env_name + '-iter%i'%(iter_num))

		return saving_result




