from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, pdb, multiprocessing
from utils.krylov import cg
from utils.utils import *
from agent.agent import Agent

class MtlTrpoAgent(Agent):
	def __init__(self, envs, mtl_actor, baselines, session, pms, env_contexts = None, saver = None,**kwargs):
		super(MtlTrpoAgent, self).__init__(envs, session, pms, saver)
		self.actor = mtl_actor
		self.baseline = baselines
		self.task_diff = 'task' if isinstance(envs, list) else 'context'
		self.num_of_tasks = len(envs) if self.task_diff is 'task' else len(env_contexts)
		self.env_contexts = env_contexts

		# self.var_list = self.actor.var_list
		self.shared_var_list = self.actor.shared_var_list
		self.task_var_lists = self.actor.task_var_list
		self.var_list = self.shared_var_list + sum(self.task_var_lists, [])
		for baseline in self.baseline:
			baseline.boost_baseline = True 
		# self.boost_baseline = [True] * self.num_of_tasks


	def init_vars(self):

		with tf.name_scope(self.pms.name_scope):
			self.obs = self.actor.input_ph
			self.advant = tf.placeholder(tf.float32, [None], name = 'advantages')
			self.action = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'action')
			self.old_dist_mean = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_mean')
			self.old_dist_logstd = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_logstd')
			self.task_descriptor = tf.placeholder(tf.bool, [None, self.num_of_tasks], name = 'task_descriptor')
			self.new_dist_mean = self.actor.output_net

			self.new_dist_logstd = [tf.tile(lsd, [ tf.shape(self.obs)[0], 1 ] ) for lsd in self.actor.action_logstd]
			# self.dist_mean = tf.case()
			self.dist_mean = tf.reduce_sum(tf.boolean_mask(self.new_dist_mean, self.task_descriptor), axis = 0)

			# self.new_dist_logstd 

			self.dist_logstd = tf.reduce_sum(tf.boolean_mask(self.new_dist_logstd, self.task_descriptor), axis = 0)

			self.flat_tangent = tf.placeholder(tf.float32, shape = [None], name = 'flat_tangent')
			self.weights_to_set = tf.placeholder(tf.float32, shape = [None], name = 'weights_to_set')

			batchsize = tf.cast(tf.shape(self.obs)[0], tf.float32)
			logli_new = log_likelihood(self.action, self.dist_mean, self.dist_logstd)
			logli_old = log_likelihood(self.action, self.old_dist_mean, self.old_dist_logstd)

			self.ratio = tf.exp(logli_new - logli_old)
			surr_loss1 = -self.ratio * self.advant
			surr_loss2 = -tf.clip_by_value( self.ratio, 1.-self.pms.cliprange, 1.+self.pms.cliprange)
			self.surr_loss = tf.reduce_mean(tf.maximum(surr_loss1, surr_loss2))
			self.kl = tf.reduce_mean( kl_sym(self.old_dist_mean, self.old_dist_logstd, self.dist_mean, self.dist_logstd) )

			self.set_shared_var_from_flat = set_from_flat(self.shared_var_list, self.weights_to_set)
			self.set_task_var_from_flat = [set_from_flat(t_var, self.weights_to_set) for t_var in self.task_var_lists]
			self.set_all_var_from_flat = set_from_flat(self.var_list, self.weights_to_set)

			self.flatten_var = flatten_var(self.var_list)
			self.flatten_shared_var = flatten_var(self.shared_var_list)
			self.flatten_task_var = [flatten_var(t_var) for t_var in self.task_var_lists]

			kl_firstfixed = tf.reduce_sum( [ kl_sym_firstfixed( d_mean, d_logstd )for d_mean, d_logstd in zip(self.new_dist_mean, self.new_dist_logstd)] )
			# kl_firstfixed = kl_sym_firstfixed(self.dist_mean, self.dist_logstd)
			all_surr_grad = tf.gradients( self.surr_loss, self.var_list )
			shared_surr_grad = tf.gradients(self.surr_loss, self.shared_var_list)
			task_surr_grad = [tf.gradients(self.surr_loss, t_var) for t_var in self.task_var_lists]

			self.flat_all_surr_grad = flatten_var( all_surr_grad )
			self.flat_shared_surr_grad = flatten_var(shared_surr_grad)
			self.flat_task_surr_grad = [flatten_var(g) for g in task_surr_grad]

			kl_grads_all = tf.gradients(kl_firstfixed, self.var_list)
			flat_kl_grads_all = flatten_var( kl_grads_all ) / batchsize
			fvp_all = tf.gradients( tf.reduce_sum( flat_kl_grads_all * self.flat_tangent), self.var_list )
			self.flat_fvp_all = flatten_var(fvp_all)

			kl_grads_shared = tf.gradients( kl_firstfixed, self.shared_var_list )
			# kl_grads_shared = [tf.gradients(kl, self.shared_var_list) for kl in kl_firstfixed]
			flat_kl_grads_shared = flatten_var(kl_grads_shared)/batchsize
			fvp_shared = tf.gradients( tf.reduce_sum(flat_kl_grads_shared * self.flat_tangent), self.shared_var_list )
			self.flat_fvp_shared = flatten_var(fvp_shared)

			# kl_grads = [ tf.gradients(kl,var) for kl,var in zip(kl_firstfixed, self.task_var_lists)]
			# flat_kl_grads = [flatten_var(g)/batchsize for g in kl_grads]
			# fvp = [ tf.gradients(tf.reduce_sum(fg * self.flat_tangent), var) for fg, var in zip(flat_kl_grads, self.task_var_lists)]
			# self.flat_fvp = [flatten_var(f) for f in fvp]

			# kl_grads_2 = [ tf.gradients(kl, self.shared_var_list) for kl in kl_firstfixed]
			# flat_kl_grads_2 = [flatten_var(g)/batchsize for g in kl_grads_2]
			# cs = [tf.gradients( tf.reduce_sum( kll * self.flat_tangent), var) for kll,var in zip(flat_kl_grads_2, self.task_var_lists)]
			# self.flat_c = [flatten_var(c) for c in cs]

	def get_single_path(self, task_index):
		observations = []
		actions = []
		rewards = []
		actor_infos = []

		if self.task_diff is 'task':
			env = self.env[task_index]
		elif self.task_diff is 'context':
			env = self.env
			context = self.env_contexts[task_index]
			try:
				env.target_value = context['speed']
			except:
				pass

			try:
				env.model.opt.gravity[0] = context['wind']
			except:
				pass
			
			try:
				env.model.opt.gravity[2] = context['gravity']
			except:
				pass


		# if self.pms.env_name.startswith('walker'):
		# 	self.env.model.opt.gravity[2] = gravity
		state = env.reset()

		if self.pms.render:
			env.render()

		for _ in range(self.pms.max_time_step):
			action, actor_info = self.actor.get_action(state, task_index)
			action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)
			next_state, reward, terminal, _ = env.step(self.pms.max_action * action)

			observations.append(state)
			actions.append(action)
			rewards.append(reward)
			actor_infos.append(actor_info)
			if terminal:
				break
			state = next_state
			if self.pms.render:
				env.render()
		if self.pms.render:
			pass
		path = dict(observations = np.array(observations), actions = np.array(actions), rewards = np.array(rewards), actor_infos = actor_infos)
		return path

	def get_paths(self, task_index, prefix = '', verbose = True):
		if not self.pms.max_total_time_step == 0:
			num_of_paths = num_of_paths_per_task = np.inf
			num_of_steps = self.pms.max_total_time_step

		paths = []
		t = time.time()
		path_count = 0
		step_count = 0
		if verbose:
			print(prefix + 'Gathering Samples')

		while True:
			paths.append(self.get_single_path( task_index ) )
			path_count += 1
			if verbose:
				sys.stdout.write('%i-th path sampled. simulation time: %f \r'%(path_count, time.time()-t))
				sys.stdout.flush()
			step_count += len(paths[-1]['actions'])
			if step_count>=num_of_steps or path_count >= num_of_paths:
				break
		# if verbose:
		print(prefix + 'All paths sampled. Total sampled paths: %i. Total time usesd: %f.'%(path_count, time.time() - t) )
		return paths

	def process_paths(self, paths, baseline, fit = True):

		total_time_step = 0
		for path in paths:
			total_time_step += len(path['rewards'])
			path['baselines'] = baseline.predict(path)
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
		returns = np.concatenate([path['returns'] for path in paths])
		actor_infos = np.concatenate([path['actor_infos'] for path in paths])
		if self.pms.center_adv:
			advantages -= np.mean(advantages)
			advantages /= (np.std(advantages) + 1e-8)

		sample_data = dict(
			observations = observations,
			actions = actions,
			rewards = rewards,
			returns = returns,
			advantages = advantages,
			actor_infos = actor_infos,
			paths = paths,
			total_time_step = total_time_step
		)

		if fit:
			if baseline.boost_baseline:
				baseline.fit(observations, returns, iter_num = 5)
				baseline.boost_baseline = False
			else:
				baseline.fit(observations, returns, iter_num = 5)
		else:
			pass

		return sample_data


	def train_paths(self, paths):
		flat_theta_prev = self.sess.run(self.flatten_var)
		flat_shared_prev = self.sess.run(self.flatten_shared_var)
		flat_task_prev = [self.sess.run(var) for var in self.flatten_task_var]

		sample_data = [None] * len(paths)
		for i, (path, baseline) in enumerate(zip(paths, self.baseline)):
			sample_data[i] = self.process_paths(path, baseline)
			tmp_des = np.zeros(self.num_of_tasks).astype(bool)
			tmp_des[i] = True
			sample_data[i]['descritors'] = np.tile(tmp_des, [sample_data[i]['total_time_step'],1])
		episode_rewards = [np.array([np.sum(p['rewards']) for p in sd['paths']]) for sd in sample_data]
		n_sampless = [sd['total_time_step'] for sd in sample_data]
		surrgate_loss = [None] * len(paths)
		kl_divergence = [None] * len(paths)

		batchsize = int(self.pms.max_total_time_step * self.pms.subsample_factor)

		'''
			Gathering Gradients for Shared Vars
		'''

		shared_grads = [None] * len(paths)
		feed_obs = np.empty(shape = [0, sample_data[0]['observations'].shape[1]])
		feed_adv = np.array([])
		feed_act = np.empty(shape = [0, sample_data[0]['actions'].shape[1]])
		feed_old_mean = np.empty( shape = [0, sample_data[0]['actions'].shape[1]])
		feed_old_logstd = np.empty(shape = [0, sample_data[0]['actions'].shape[1]])
		feed_des = np.empty(shape = [0, sample_data[0]['descritors'].shape[1]])
		# shared_grads = [None] * len(paths)
		indses = [None] * len(paths)
		for i, sd in enumerate(sample_data):

			n_samples = sd['total_time_step']
			inds = np.random.choice(n_samples, int(np.floor(n_samples * self.pms.subsample_factor)), replace = False)
			indses[i] = inds
			feed_obs = np.append( feed_obs, np.array(sd['observations'][inds]), axis = 0)
			feed_adv = np.append( feed_adv, np.array(sd['advantages'][inds]) )
			feed_act = np.append( feed_act, np.array(sd['actions'][inds]), axis = 0 )
			feed_des = np.append( feed_des, np.array(sd['descritors'][inds]) ,axis = 0)
			feed_old_mean = np.append( feed_old_mean, np.array([a_info['mean'] for a_info in sd['actor_infos'][inds]]), axis = 0 )
			feed_old_logstd = np.append( feed_old_logstd, np.array([a_info['logstd'] for a_info in sd['actor_infos'][inds]]), axis = 0 )
			# feed_dict = {self.obs: sd['observations'][inds],
			# 			 self.advant: sd['advantages'][inds],
			# 			 self.action: sd['actions'][inds],
			# 			 self.old_dist_mean: np.array([a_info['mean'] for a_info in sd['actor_infos'][inds]]),
			# 			 self.old_dist_logstd: np.array([a_info['logstd'] for a_info in sd['actor_infos'][inds]])
			# 			}
			# shared_grads[i] = self.sess.run( self.flat_shared_surr_grad[i], feed_dict = feed_dict )
			# self.sess.run(self.train_op[i], feed_dict = feed_dict)
		# shared_grad = self.sess.run(self.flatten_shared_var) - flat_shared_prev

		# shared_grad = - np.nanmean(shared_grads, axis = 0)
		# inv_s = [None] * len(paths)

		# for i, sd in enumerate(sample_data):
			# inds = indses[i]
			# feed_dict = {self.obs: sd['observations'][inds],
			# 			 self.advant: sd['advantages'][inds],
			# 			 self.action: sd['actions'][inds],
			# 			 self.old_dist_mean: np.array([a_info['mean'] for a_info in sd['actor_infos'][inds]]),
			# 			 self.old_dist_logstd: np.array([a_info['logstd'] for a_info in sd['actor_infos'][inds]])
			# 			}
		# print(np.shape(np.array(feed_obs))))
		feed_dict = {self.obs: np.array(feed_obs),
					self.action: np.array(feed_act),
					self.advant: np.array(feed_adv),
					self.old_dist_mean: np.array(feed_old_mean),
					self.old_dist_logstd: np.array(feed_old_logstd),
					self.task_descriptor: np.array(feed_des)
					}

		grads = self.sess.run( self.flat_all_surr_grad, feed_dict = feed_dict)

		def fisher_vector_product(p):
			feed_dict[self.flat_tangent] = p
			return self.sess.run(self.flat_fvp_all[i], feed_dict = feed_dict)

		grads = cg(fisher_vector_product, -grads, cg_iters = self.pms.cg_iters)
		sAs = grads.dot(fisher_vector_product(grads))
		inv_s = np.sqrt(sAs/(2.*self.pms.max_kl))
		step_gradient = grads / (inv_s + 1e-8)

		def loss_function(x):
			self.sess.run(self.set_all_var_from_flat, feed_dict = {self.weights_to_set: x})
			loss, kl = self.sess.run( [self.surr_loss, self.kl], feed_dict = feed_dict )
			return loss, kl

		flat_theta_new = linesearch(loss_function, flat_theta_prev, step_gradient, self.pms.max_backtracks, self.pms.max_kl)
		surrgate_loss, kl_divergence = loss_function(flat_theta_new)

		# task_grads = [None] * len(paths)
		# for i, sd in enumerate(sample_data):

		# 	# self.sess.run(self.set_shared_var_from_flat, feed_dict = {self.weights_to_set: flat_shared_prev})

		# 	n_samples = sd['total_time_step']
		# 	# inds = np.random.choice(n_samples, int(np.floor(n_samples * self.pms.subsample_factor)), replace = False)
		# 	inds = indses[i]
		# 	feed_dict = {self.obs: sd['observations'][inds],
		# 				 self.advant: sd['advantages'][inds],
		# 				 self.action: sd['actions'][inds],
		# 				 self.old_dist_mean: np.array([a_info['mean'] for a_info in sd['actor_infos'][inds]]),
		# 				 self.old_dist_logstd: np.array([a_info['logstd'] for a_info in sd['actor_infos'][inds]])
		# 				}
		# 	task_grads[i] = self.sess.run( self.flat_task_surr_grad[i], feed_dict = feed_dict )
		# 	feed_dict[self.flat_tangent] = shared_grad
		# 	additional_c = self.sess.run(self.flat_c[i], feed_dict = feed_dict)
		# 	# print(self.sess.run(self.kl[i] , feed_dict = feed_dict))
		# 	def fisher_vector_product(p):
		# 		feed_dict[self.flat_tangent] = p
		# 		return self.sess.run(self.flat_fvp[i], feed_dict = feed_dict)

		# 	# g = self.sess.run(self.flat_task_surr_grad[i], feed_dict = feed_dict)
		# 	step_gradient = cg(fisher_vector_product, -task_grads[i], cg_iters = self.pms.cg_iters)
		# 	sAs = step_gradient.dot( fisher_vector_product(step_gradient) )
		# 	inv_stepisze = np.sqrt( sAs/(2.*self.pms.max_kl) )
		# 	fullstep_gradient = step_gradient / (inv_stepisze + 1e-8)

		# 	A_1C = cg( fisher_vector_product, additional_c, cg_iters = self.pms.cg_iters)

		# 	def loss_function(x):
		# 		# self.sess.run( self.set_shared_var_from_flat, feed_dict = {self.weights_to_set: flat_shared_prev + shared_grad})
		# 		self.sess.run( self.set_task_var_from_flat[i], feed_dict = {self.weights_to_set: x - A_1C})
		# 		surr_loss, kl = self.sess.run( [self.surr_loss[i], self.kl[i]], feed_dict = feed_dict)
		# 		return surr_loss, kl

		# 	flat_theta_new = linesearch(loss_function, flat_task_prev[i], fullstep_gradient, self.pms.max_backtracks, self.pms.max_kl)

		# 	surrgate_loss[i], kl_divergence[i] = loss_function(flat_theta_new)

		# self.sess.run( self.set_shared_var_from_flat, feed_dict = {self.weights_to_set: flat_shared_prev + shared_grad})
		self.sess.run(self.set_all_var_from_flat, feed_dict = {self.weights_to_set: flat_theta_new})
		stats = dict(
				surrgate_loss = surrgate_loss,
				kl_divergence = kl_divergence,
				average_return = [np.mean(er) for er in episode_rewards],
				total_time_step = n_sampless
				)
		return stats

	def learn(self, iteration = None):
		if iteration is None: iteration = self.pms.max_iter
		dict_keys = ['average_return', 'sample_time', 'total_time_step', \
				'train_time', 'surrgate_loss', 'kl_divergence', 'iteration_number']
		saving_result = dict([(v, []) for v in dict_keys])

		np.set_printoptions(precision = 3)
		# def assign_path(i):
		# 	return self.get_paths(i, prefix = 'Task %i '%i, verbose = False)
		# pool = multiprocessing.Pool(processes = min(self.num_of_tasks, 4))
		# all_paths = [None] * self.num_of_tasks
		
		for iter_num in range(iteration):
			print('\n******************* Iteration %i *******************'%iter_num)
			t = time.time()
			all_paths = [self.get_paths(i, prefix = 'Task %i '%i) for i in range(self.num_of_tasks)]
			# all_paths = pool.map(lambda i: self.get_paths(i, prefix = 'Task %i '%i, verbose = False), range(self.num_of_tasks))
			# pool.join()
			sample_time = time.time() - t
			t = time.time()
			stats = self.train_paths(all_paths)
			# self.sess.run(self.set_var_from_flat, feed_dict = {self.weights_to_set: theta})
			train_time = time.time() - t


			for k, v in stats.items():
					print("%-20s: "%(k) + str(np.array(v)) )

			# logstds = self.sess.run(self.actor.action_logstd)
			# print(logstds)

			save_value_list = [stats['average_return'], sample_time, stats['total_time_step'], \
					train_time, stats['surrgate_loss'], stats['kl_divergence'], iter_num]

			[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]

			if self.pms.save_model and iter_num % self.pms.save_model_iters == 0:
					self.save_model(self.pms.save_dir + self.pms.env_name + '-iter%i'%(iter_num + self.pms.pre_iter))

		return saving_result

# def get_paths(agent, i):
# 	return agent.get_paths()


