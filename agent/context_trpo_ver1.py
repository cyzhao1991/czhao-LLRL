from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.krylov import cg
from utils.utils import *
from agent.agent import Agent
# from dm_control.rl import control

class Context_TRPO_Agent(Agent):

	def __init__(self, env, actor, baseline, session, flags, saver = None, **kwargs):#context_range = None):
		super(Context_TRPO_Agent, self).__init__(env, session, flags, saver)
		self.actor = actor
		self.baseline = baseline
		# self.var_list = self.actor.var_list
		# self.num_of_tasks = len(env)
		# self.num_of_tasks = np.array(env_contexts).shape[0]
		self.context_range = kwargs.get('context_range', None)
		self.env_contexts = kwargs.get('env_contexts', None)
		self.num_of_tasks = np.array(self.env_contexts).shape[0] if self.env_contexts is not None else 0

		self.var_list = self.actor.var_list
		self.shared_var_list = self.actor.shared_var_list
		self.task_var_list = self.actor.task_var_list
		self.boost_baseline = True
		self.shared_var_mask = actor.shared_var_mask
		self.task_var_mask = actor.task_var_mask

		# self.shared_var_num = np.sum( self.sess.run([tf.size(v) for v in self.shared_var_list]) )
		self.default_context = np.array(self.env.model.opt.gravity)
		# self.default_context = np.array(self.env.physics.model.opt.gravity)

		self.init_vars()

		print('Building Network')

	def init_vars(self):

		with tf.name_scope(self.pms.name_scope):
			self.obs = self.actor.input_ph
			self.context = self.actor.context_ph
			self.advant = tf.placeholder(tf.float32, [None], name = 'advantages')
			self.action = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'action')
			self.old_dist_mean = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_mean')
			self.old_dist_logstd = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_logstd')
			self.new_dist_mean = self.actor.output_net
			self.new_dist_logstd = self.actor.action_logstd

			self.flat_tangent = tf.placeholder(tf.float32, shape = [None], name = 'flat_tangent')

			batchsize = tf.cast(tf.shape(self.obs)[0], tf.float32)
			logli_new = log_likelihood(self.action, self.new_dist_mean, self.new_dist_logstd)
			logli_old = log_likelihood(self.action, self.old_dist_mean, self.old_dist_logstd)

			'''
				REINFORCE Policy Gradient
			'''

			self.l1_norm = self.pms.l1_regularizer * tf.add_n( [tf.reduce_sum( tf.nn.leaky_relu(v, alpha = .01) ) for v in self.task_var_list] )
			self.l0_norm = tf.add_n( [tf.count_nonzero( v > 0.001) for v in self.task_var_list] )
			self.column_loss = self.pms.l1_column_reg * tf.add_n( [tf.reduce_sum( tf.reduce_max( tf.nn.leaky_relu(v, alpha = .01), axis=0 )) for v in self.task_var_list] ) 

			self.pg_loss = logli_new * self.advant
			self.pg_grad = tf.gradients( -self.pg_loss , self.task_var_list) 
			self.sparse_grad = tf.gradients( self.l1_norm + self.column_loss, self.task_var_list )
			self.flat_pg_grad = flatten_var( [pg + sg for pg,sg in zip(self.pg_grad, self.sparse_grad) ] )

			kl_firstfixed = kl_sym_firstfixed(self.new_dist_mean, self.new_dist_logstd)
			grads = tf.gradients(kl_firstfixed, self.task_var_list) 
			flat_grads = flatten_var(grads) / batchsize
			# flat_grads = tf.where(self.flat_mask, tmp_flat_grad, np.zeros(self.flatten_var.shape))
			fvp = tf.gradients(tf.reduce_sum(flat_grads * self.flat_tangent), self.task_var_list)
			self.task_flat_fvp = flatten_var(fvp)

			'''
				Trust Region Policy Optimisation
			'''
			self.ratio = tf.exp(logli_new - logli_old)
			surr_loss1 = - self.ratio * self.advant
			surr_loss2 = - self.advant * tf.clip_by_value(self.ratio, 1.0-self.pms.cliprange, 1.0+self.pms.cliprange)
			self.surr_loss = tf.reduce_mean(tf.maximum(surr_loss1, surr_loss2))
			self.kl = tf.reduce_mean(kl_sym(self.old_dist_mean, self.old_dist_logstd, self.new_dist_mean, self.new_dist_logstd))

			# relu_task_var = [tf.nn.relu(v) for v in self.task_var_list]
			# if self.pms.contextual_method is 'meta_s_network':
			# 	self.l1_norm = tf.add_n([tf.reduce_sum(v) for v in relu_task_var])
			# 	self.l0_norm = tf.add_n([tf.count_nonzero(v) for v in relu_task_var])
			# elif self.pms.contextual_method is 'concatenate':
			

			# sig_task_var = [tf.sigmoid(v) for v in self.task_var_list]
			# self.l1_norm = tf.add_n([tf.reduce_sum(v) for v in sig_task_var])
			# self.l0_norm = tf.add_n([tf.count_nonzero(v) for v in sig_task_var])

			if self.pms.contextual_method is 'meta_s_network':
				self.total_loss = self.surr_loss + self.l1_norm + self.column_loss#/ self.old_l1_norm
			elif self.pms.contextual_method is 'concatenate':
				self.total_loss = self.surr_loss

			# self.total_loss = self.surr_loss 
			# if self.shared_var_mask is not None and self.task_var_mask is not None:
			# 	self.flat_mask = flatten_var( self.shared_var_mask + self.task_var_mask )
			# else:
			# 	self.flat_mask = flatten_var( [np.full(s.shape, True, dtype = bool) for s in self.var_list] )

			if self.shared_var_mask is None:
				self.shared_var_mask = [np.full(s.shape, True, dtype = bool) for s in self.shared_var_list]
			self.shared_flat_mask = flatten_var(self.shared_var_mask)
			if self.task_var_mask is None:
				self.task_var_mask = [np.full(s.shape, True, dtype = bool) for s in self.task_var_list]
			self.task_flat_mask = flatten_var(self.task_var_mask)

			self.weights_to_set = tf.placeholder(tf.float32, [None], name = 'weights_to_set')
			self.set_var_from_flat = set_from_flat(self.var_list, self.weights_to_set)
			self.set_shared_var_from_flat = set_from_flat(self.shared_var_list, self.weights_to_set)
			self.set_task_var_from_flat = set_from_flat(self.task_var_list, self.weights_to_set)

			self.flatten_var = flatten_var(self.var_list)
			self.flatten_shared_var = flatten_var(self.shared_var_list)
			self.flatten_task_var = flatten_var(self.task_var_list)

			# surr_grad_task = tf.gradients(self.total_loss, self.task_var_list) 
			# self.flat_surr_grad_task = tf.where( self.task_flat_mask, flatten_var(surr_grad_task), np.zeros(self.flatten_task_var.shape) )
			# self.flat_surr_grad_task = flatten_var(surr_grad_task)
			surr_grad = tf.gradients(self.surr_loss, self.shared_var_list)
			flat_surr_grad = flatten_var( surr_grad )
			# self.flat_surr_grad = tf.where( self.shared_flat_mask, flat_surr_grad, np.zeros(self.flatten_shared_var.shape))
			# self.flat_shared_grad = flatten_var( tf.gradients(self.total_loss, self.shared_var_list) )
			# self.flat_task_grad = flatten_var( tf.gradients(self.total_loss, self.task_var_list) )
			self.flat_surr_grad = flat_surr_grad

			kl_firstfixed = kl_sym_firstfixed(self.new_dist_mean, self.new_dist_logstd)
			grads = tf.gradients(kl_firstfixed, self.shared_var_list) 
			flat_grads = flatten_var(grads) / batchsize
			fvp = tf.gradients(tf.reduce_sum(flat_grads * self.flat_tangent), self.shared_var_list)
			flat_fvp = flatten_var(fvp)
			self.flat_fvp = flat_fvp

			grads_theta2 = tf.gradients(kl_firstfixed, self.task_var_list)
			flat_grads_theta2 = flatten_var(grads_theta2) / batchsize
			c = tf.gradients( tf.reduce_sum( flat_grads_theta2 * self.flat_tangent), self.shared_var_list )
			self.flat_c = flatten_var(c)


	def get_single_path(self, task_context, target_speed = 1., wind = 0., gravity = -9.8):
		observations = []
		contexts = []
		actions = []
		rewards = []
		actor_infos = []
		if self.pms.env_name.startswith('walker'):

			self.env.target_value = target_speed
			# self.env.model.opt.gravity[0] = wind
			self.env.model.opt.gravity[2] = gravity
			state = self.env.reset()

		context = task_context

		if self.pms.render:
			self.env.render()

		for _ in range(self.pms.max_time_step):
			action, actor_info = self.actor.get_action(state, context)
			action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)
			next_state, reward, terminal, _ = self.env.step(self.pms.max_action * action)

			observations.append(state)
			actions.append(action)
			rewards.append(reward)
			contexts.append(context)
			actor_infos.append(actor_info)
			if terminal:
				break
			state = next_state
			if self.pms.render:
				self.env.render()
		if self.pms.render:
			pass
		path = dict(observations = np.array(observations), contexts = np.array(contexts), actions = np.array(actions), rewards = np.array(rewards), actor_infos = actor_infos)
		return path


	def get_paths(self, prefix = '', task_context = None, verbose = True):

		if not self.pms.max_total_time_step == 0: 
			num_of_paths = num_of_paths_per_task = np.inf
			num_of_steps = self.pms.max_total_time_step

		else:
			num_of_paths = self.pms.num_of_paths
			num_of_steps = num_of_steps_per_task = np.inf

		paths = []
		t = time.time()
		path_count = 0
		step_count = 0		
		
		context_size = self.pms.context_shape - 1
		if verbose:
			print(prefix + 'Gathering Samples')


		tmp_idx = 0
		while True:
			if task_context is not None:
				pass
			elif self.context_range is not None:
				task_context = np.random.rand(context_size) * (self.context_range) - self.context_range/2 # + self.default_context
			elif self.env_contexts is not None:
				task_context = self.env_contexts[tmp_idx]
				tmp_idx += 1
				tmp_idx = np.mod(tmp_idx, self.num_of_tasks)
			paths.append(self.get_single_path( task_context ))
			path_count += 1

			if verbose:
				sys.stdout.write('%i-th path sampled. simulation time: %f \r'%(path_count, time.time()-t))
				sys.stdout.flush()

			step_count += len(paths[-1]['actions'])

			if step_count >= num_of_steps or path_count >= num_of_paths:
				break
		if verbose:
			print('All paths sampled. Total sampled paths: %i. Total time usesd: %f.'%(path_count, time.time() - t) )
		return paths

	def process_paths(self, paths, fit = True):
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
		contexts = np.concatenate([path['contexts'] for path in paths])
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
			contexts = contexts,
			actions = actions,
			rewards = rewards,
			returns = returns,
			advantages = advantages,
			actor_infos = actor_infos,
			paths = paths,
			total_time_step = total_time_step
		)

		#self.baseline.fit(observations, contexts, returns)
		if fit:
			if self.boost_baseline:
				self.baseline.fit(observations, returns, iter_num = 500)
				self.boost_baseline = False
			else:
				self.baseline.fit(observations, returns, iter_num = 5)
		else:
			pass

		return sample_data


	def train_paths(self, paths):

		flat_theta_prev = self.sess.run(self.flatten_var)
		flat_shared_prev = self.sess.run(self.flatten_shared_var)
		flat_task_prev = self.sess.run(self.flatten_task_var)

		sample_data = self.process_paths(paths)
		obs_source = sample_data['observations']
		act_source = sample_data['actions']
		adv_source = sample_data['advantages']
		con_source = sample_data['contexts']
		n_samples = sample_data['total_time_step']
		actor_info_source = sample_data['actor_infos']

		episode_rewards = np.array([np.sum(path['rewards']) for path in paths])
		train_number = int(1./self.pms.subsample_factor)
		'''
			Update task variables with pg
		'''

		pg_gradients = []
		# tmp_g1 = []
		# tmp_g2 = []
		for path in paths:
			tmp_sample_data = self.process_paths([path], fit = False)
			feed_dict = {self.obs: tmp_sample_data['observations'],
						 self.advant: tmp_sample_data['advantages'],
						 self.action: tmp_sample_data['actions'],
						 self.context: tmp_sample_data['contexts']
						 }
			pg_gradients.append( self.sess.run( self.flat_pg_grad, feed_dict = feed_dict) )
			# tmp_g1.append(self.sess.run( flatten_var(self.pg_grad), feed_dict = feed_dict))
			# tmp_g2.append(self.sess.run( flatten_var(self.sparse_grad), feed_dict = feed_dict))
		task_pg_gradients = np.nanmean( pg_gradients, axis = 0)
		# print( np.nanmean( tmp_g1, axis = 0) )
		# print( np.nanmean( tmp_g2, axis = 0) )
		# print(self.sess.run(self.task_var_list))

		inds = np.random.choice(n_samples, int(np.floor(n_samples*self.pms.subsample_factor)), replace = False)
		feed_dict = {self.obs: obs_source[inds],
					 self.advant: adv_source[inds],
					 self.action: act_source[inds],
					 self.context: con_source[inds]
					 }

		def fisher_vector_product(p):
			feed_dict[self.flat_tangent] = p
			return self.sess.run(self.task_flat_fvp, feed_dict = feed_dict) + self.pms.cg_damping * p

		step_gradient = cg(fisher_vector_product, -task_pg_gradients, cg_iters = self.pms.cg_iters)
		sAs = step_gradient.dot( fisher_vector_product(step_gradient) )
		inv_stepsize = np.sqrt( sAs/(2.*self.pms.max_kl) )
		task_pg_gradients = step_gradient / (inv_stepsize + 1e-8)

		task_pg_gradients = np.where( self.sess.run(self.task_flat_mask), task_pg_gradients, 0)


		'''
			Update shared variables with trpo
		'''
		
		step_shared_gradients = []

		for iteration in range(train_number):
			inds = np.random.choice(n_samples, int(np.floor(n_samples*self.pms.subsample_factor)), replace = False)
			obs_n = obs_source[inds]
			act_n = act_source[inds]
			adv_n = adv_source[inds]
			con_n = con_source[inds]
			act_dis_mean_n = np.array([a_info['mean'] for a_info in actor_info_source[inds]])
			act_dis_logstd_n = np.array([a_info['logstd'] for a_info in actor_info_source[inds]])

			feed_dict = {self.obs: obs_n,
						self.advant: adv_n,
						self.action: act_n,
						self.context: con_n,
						self.old_dist_mean: act_dis_mean_n,#[:,np.newaxis],
						self.old_dist_logstd: act_dis_logstd_n#[:,np.newaxis]
						}

			def fisher_vector_product(p):
				feed_dict[self.flat_tangent] = p
				return self.sess.run(self.flat_fvp, feed_dict = feed_dict) + self.pms.cg_damping * p

			step_shared_gradients.append( self.sess.run(self.flat_surr_grad, feed_dict = feed_dict) )

		g = np.nanmean(step_shared_gradients, axis = 0)
		
		step_gradient = cg(fisher_vector_product, -g, cg_iters = self.pms.cg_iters)
		sAs = step_gradient.dot( fisher_vector_product(step_gradient) )
		inv_stepsize = np.sqrt( sAs/(2.*self.pms.max_kl) )
		fullstep_gradient = step_gradient / (inv_stepsize + 1e-8)

		feed_dict[self.flat_tangent] = task_pg_gradients
		additional_c = self.sess.run(self.flat_c, feed_dict = feed_dict)

		A_1C = cg(fisher_vector_product, additional_c, cg_iters = self.pms.cg_iters)

		def loss_function(x):
			self.sess.run( self.set_shared_var_from_flat, feed_dict = {self.weights_to_set: x - A_1C})
			surr_loss, kl = self.sess.run( [self.surr_loss, self.kl], feed_dict = feed_dict	)
			return surr_loss, kl

		if self.pms.linesearch:
			flat_theta_new = linesearch(loss_function, flat_shared_prev, fullstep_gradient, self.pms.max_backtracks, self.pms.max_kl)
		else:
			flat_theta_new = flat_theta_prev + fullstep_gradient
		# update_gradient = flat_theta_new - flat_theta_prev

		self.sess.run(self.set_task_var_from_flat, feed_dict = {self.weights_to_set: flat_task_prev + task_pg_gradients})
		self.sess.run(self.set_shared_var_from_flat, feed_dict = {self.weights_to_set: flat_theta_prev})
		# self.sess.run( self.set_shared_var_from_flat, feed_dict = {self.weights_to_set: flat_theta_new - A_1C})
		# surrgate_loss, kl_divergence = self.sess.run( [self.surr_loss, self.kl], feed_dict = feed_dict	)
		surrgate_loss, kl_divergence = loss_function(flat_theta_new)

		stats = dict(
				surrgate_loss = surrgate_loss,
				kl_divergence = kl_divergence,
				average_return = np.mean(episode_rewards),
				total_time_step = n_samples,
				l1_norm = self.sess.run(self.l1_norm),
				column_norm = self.sess.run(self.column_loss)
				)
		return flat_theta_new, flat_theta_prev, stats


	def learn(self, iteration = None):
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
		if iteration is None:	iteration = self.pms.max_iter
		dict_keys = ['average_return', 'sample_time', 'total_time_step', \
				'train_time', 'surrgate_loss', 'kl_divergence', 'iteration_number', 'l1_norm', 'column_norm']
		saving_result = dict([(v, []) for v in dict_keys])

		for iter_num in range(iteration):
			print('\n******************* Iteration %i *******************'%iter_num)
			t = time.time()
			paths = self.get_paths()
			sample_time = time.time() - t
			t = time.time()
			theta, theta_old, stats = self.train_paths(paths)
			# self.sess.run(self.set_var_from_flat, feed_dict = {self.weights_to_set: theta})
			train_time = time.time() - t

			for k, v in stats.items():
					print("%-20s: %15.5f"%(k,v))

			logstds = self.sess.run(self.actor.action_logstd)
			print(logstds)



			save_value_list = [stats['average_return'], sample_time, stats['total_time_step'], \
					train_time, stats['surrgate_loss'], stats['kl_divergence'], iter_num, stats['l1_norm'], stats['column_norm']]

			[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]

			if self.pms.save_model and iter_num % self.pms.save_model_iters == 0:
					self.save_model(self.pms.save_dir + self.pms.env_name + '-iter%i'%(iter_num + self.pms.pre_iter))

		return saving_result
