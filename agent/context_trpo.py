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

			logli_new = log_likelihood(self.action, self.new_dist_mean, self.new_dist_logstd)
			logli_old = log_likelihood(self.action, self.old_dist_mean, self.old_dist_logstd)
			self.ratio = tf.exp(logli_new - logli_old)

			surr_loss1 = - self.ratio * self.advant
			surr_loss2 = - self.advant * tf.clip_by_value(self.ratio, 1.0-self.pms.cliprange, 1.0+self.pms.cliprange)
			self.surr_loss = tf.reduce_mean(tf.maximum(surr_loss1, surr_loss2))
			self.kl = tf.reduce_mean(kl_sym(self.old_dist_mean, self.old_dist_logstd, self.new_dist_mean, self.new_dist_logstd))

			self.old_l1_norm = tf.placeholder(tf.float32, [None], name = 'old_l1_norm')

			# relu_task_var = [tf.nn.relu(v) for v in self.task_var_list]
			# if self.pms.contextual_method is 'meta_s_network':
			# 	self.l1_norm = tf.add_n([tf.reduce_sum(v) for v in relu_task_var])
			# 	self.l0_norm = tf.add_n([tf.count_nonzero(v) for v in relu_task_var])
			# elif self.pms.contextual_method is 'concatenate':
			# 	self.l1_norm = self.l0_norm = tf.constant([0], tf.float32)
			self.l1_norm = self.pms.l1_regularizer * tf.add_n( [tf.reduce_sum( tf.abs(v) ) for v in self.task_var_list] )
			self.l0_norm = tf.add_n( [tf.count_nonzero( tf.abs(v) > 0.001) for v in self.task_var_list] )
			self.column_loss = self.pms.l1_column_reg * tf.add_n( [tf.reduce_sum( tf.reduce_max( tf.abs(v), axis=0 )) for v in self.task_var_list] ) 

			# sig_task_var = [tf.sigmoid(v) for v in self.task_var_list]
			# self.l1_norm = tf.add_n([tf.reduce_sum(v) for v in sig_task_var])
			# self.l0_norm = tf.add_n([tf.count_nonzero(v) for v in sig_task_var])

			if self.pms.contextual_method is 'meta_s_network':
				self.total_loss = self.surr_loss + self.l1_norm + self.column_loss#/ self.old_l1_norm
			elif self.pms.contextual_method is 'concatenate':
				self.total_loss = self.surr_loss

			# self.total_loss = self.surr_loss 
			if self.shared_var_mask is not None and self.task_var_mask is not None:
				self.flat_mask = flatten_var( self.shared_var_mask + self.task_var_mask )
			else:
				self.flat_mask = flatten_var( [np.full(s.shape, True, dtype = bool) for s in self.var_list] )

			self.weights_to_set = tf.placeholder(tf.float32, [None], name = 'weights_to_set')
			self.set_var_from_flat = set_from_flat(self.var_list, self.weights_to_set)
			self.flatten_var = flatten_var(self.var_list)

			surr_grad = tf.gradients(self.total_loss, self.var_list)
			flat_surr_grad = flatten_var( tf.gradients(self.total_loss, self.var_list) )
			self.flat_surr_grad = tf.where( self.flat_mask, flat_surr_grad, np.zeros(self.flatten_var.shape))
			# self.flat_shared_grad = flatten_var( tf.gradients(self.total_loss, self.shared_var_list) )
			# self.flat_task_grad = flatten_var( tf.gradients(self.total_loss, self.task_var_list) )

			batchsize = tf.cast(tf.shape(self.obs)[0], tf.float32)
			self.flat_tangent = tf.placeholder(tf.float32, shape = [None], name = 'flat_tangent')
			kl_firstfixed = kl_sym_firstfixed(self.new_dist_mean, self.new_dist_logstd)
			grads = tf.gradients(kl_firstfixed, self.var_list) 
			flat_grads = flatten_var(grads) / batchsize
			# flat_grads = tf.where(self.flat_mask, tmp_flat_grad, np.zeros(self.flatten_var.shape))

			fvp = tf.gradients(tf.reduce_sum(flat_grads * self.flat_tangent), self.var_list)
			flat_fvp = flatten_var(fvp)
			self.flat_fvp = tf.where(self.flat_mask, flat_fvp, np.zeros(self.flatten_var.shape))

			# shared_grads = tf.gradients(kl_firstfixed, self.shared_var_list)
			# task_grads = tf.gradients(kl_firstfixed, self.task_var_list)
			# # self.shared_flat_fvp = flatten_var( tf.gradients(tf.reduce_sum( flatten_var(shared_grads)/batchsize*self.flat_tangent) , self.shared_var_list) )
			# # self.task_flat_fvp = flatten_var( tf.gradients(tf.reduce_sum( flatten_var(task_grads)/batchsize*self.flat_tangent) , self.task_var_list) )

			# self.set_shared_from_flat = set_from_flat(self.shared_var_list, self.weights_to_set)
			# self.set_task_from_flat = set_from_flat(self.task_var_list, self.weights_to_set)
			# self.flatten_shared_var = flatten_var(self.shared_var_list)
			# self.flatten_task_var = flatten_var(self.task_var_list )

	# def update_vars(self, shared_var_list, task_var_list):
	# 	# self.shared_var_list = shared_var_list
	# 	# self.task_var_list = task_var_list
	# 	var_list = shared_var_list + task_var_list

	# 	with tf.name_scope(self.pms.name_scope):
	# 		# self.l1_norm = self.pms.l1_regularizer * tf.add_n( [tf.reduce_sum( tf.abs(v) ) for v in task_var_list] )
	# 		# self.l0_norm = tf.add_n( [tf.count_nonzero( tf.abs(v) > 0.001) for v in task_var_list] )
			
	# 		# if self.pms.contextual_method is 'meta_s_network':
	# 		# 	self.total_loss = self.surr_loss + self.l1_norm
	# 		# elif self.pms.contextual_method is 'concatenate':
	# 		# 	self.total_loss = self.surr_loss

	# 		self.flat_surr_grad = flatten_var( tf.gradients(self.total_loss, shared_var_list + task_var_list) )
	# 		self.flat_shared_surr_grad = flatten_var( tf.gradients(self.total_loss, shared_var_list) )
	# 		self.flat_task_surr_grad = flatten_var( tf.gradients(self.total_loss, task_var_list) )

	# 		batchsize = tf.cast(tf.shape(self.obs)[0], tf.float32)
	# 		self.flat_tangent = tf.placeholder(tf.float32, shape = [None], name = 'flat_tangent')
	# 		kl_firstfixed = kl_sym_firstfixed(self.new_dist_mean, self.new_dist_logstd)
			
	# 		grads = tf.gradients(kl_firstfixed, var_list) 
	# 		flat_grads = flatten_var(grads) / batchsize
	# 		fvp = tf.gradients(tf.reduce_sum(flat_grads * self.flat_tangent), var_list)
	# 		self.flat_fvp = flatten_var(fvp)

	# 		shared_grads = tf.gradients(kl_firstfixed, shared_var_list)
	# 		task_grads = tf.gradients(kl_firstfixed, task_var_list)
	# 		self.shared_flat_fvp = flatten_var( tf.gradients(tf.reduce_sum( flatten_var(shared_grads)/batchsize*self.flat_tangent) , shared_var_list) )
	# 		self.task_flat_fvp = flatten_var( tf.gradients(tf.reduce_sum( flatten_var(task_grads)/batchsize*self.flat_tangent) , task_var_list) )

	# 		self.weights_to_set = tf.placeholder(tf.float32, [None], name = 'weights_to_set')
	# 		self.set_var_from_flat = set_from_flat(var_list, self.weights_to_set)
	# 		self.flatten_var = flatten_var(var_list)

	# 		self.set_shared_from_flat = set_from_flat(shared_var_list, self.weights_to_set)
	# 		self.set_task_from_flat = set_from_flat(task_var_list, self.weights_to_set)
	# 		self.flatten_shared_var = flatten_var(shared_var_list)
	# 		self.flatten_task_var = flatten_var(task_var_list )



	def get_single_path(self, task_context, target_speed = 1., wind = 0., gravity = -9.8):
		observations = []
		contexts = []
		actions = []
		rewards = []
		actor_infos = []
		# grav_context = 5. * task_context[0] + self.default_context[2]
		if self.pms.env_name.startswith('walker'):
			# print('gravity', grav_context)
			self.env.target_value = target_speed
			self.env.model.opt.gravity[0] = wind
			self.env.model.opt.gravity[2] = gravity
			# self.env.model.opt.gravity[0] = self.default_context[0] + task_context[-1] #grav_context[0]
			# self.env.physics.model.opt.gravity[1] = 0. #grav_context[1]
			# self.env.physics.model.opt.gravity[2] = grav_context #[2]

			state = self.env.reset()
			# state = control.flatten_observation(state[-1])
			# state = state.values()[0]

		context = task_context
		# context = np.append(1., context)
		# print(self.sess.run(self.actor.net.s_vector, feed_dict = {self.context: context[np.newaxis, :]}))
		# print(context)
		if self.pms.render:
			self.env.render()
			# frame = self.env.physics.render(height = 480, width = 480, camera_id = 0)
			# cv2.imshow('simulation', frame)
			# cv2.waitKey(10)

		for _ in range(self.pms.max_time_step):
			action, actor_info = self.actor.get_action(state, context)
			action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)
			next_state, reward, terminal, _ = self.env.step(self.pms.max_action * action)
			# time_step, reward, _, next_state = self.env.step(action)

			# height = next_state['height']
			# next_state = control.flatten_observation(next_state)
			# next_state = next_state.values()[0]
			# terminal = time_step.last()

			observations.append(state)
			actions.append(action)
			rewards.append(reward)
			contexts.append(context)
			actor_infos.append(actor_info)
			if terminal :#or height < 0.3:
				break
			state = next_state
			if self.pms.render:
				# frame = self.env.physics.render(height = 480, width = 480, camera_id = 0)
				# cv2.imshow('simulation',frame)
				# cv2.waitKey(10)
				self.env.render()
		if self.pms.render:
			pass
			# cv2.destroyAllWindows()
		path = dict(observations = np.array(observations), contexts = np.array(contexts), actions = np.array(actions), rewards = np.array(rewards), actor_infos = actor_infos)
		return path


	def get_paths(self, prefix = '', task_context = None, verbose = True):

		if not self.pms.max_total_time_step == 0: 
			num_of_paths = num_of_paths_per_task = np.inf
			num_of_steps = self.pms.max_total_time_step
			# num_of_steps_per_task = np.ceil(num_of_steps * 1./ self.num_of_tasks)

		else:
			num_of_paths = self.pms.num_of_paths
			# num_of_paths_per_task = np.ceil(num_of_paths * 1./ self.num_of_tasks)
			num_of_steps = num_of_steps_per_task = np.inf

		paths = []
		t = time.time()
		path_count = 0
		step_count = 0		
		
		# context_size = self.context_range.shape[0]
		context_size = self.pms.context_shape - 1
		if verbose:
			print(prefix + 'Gathering Samples')

		# for task_index in range(self.num_of_tasks):
		# 	paths.append([])
		# 	task_path_count = 0
		# 	task_step_count = 0
		tmp_idx = 0
		while True:
		# for i in range(num_of_paths):
			if task_context is not None:
				pass
			elif self.context_range is not None:
				task_context = np.random.rand(context_size) * (self.context_range) - self.context_range/2 # + self.default_context
			elif self.env_contexts is not None:
				# tmp_idx = np.random.randint(self.num_of_tasks)
				task_context = self.env_contexts[tmp_idx]
				tmp_idx += 1
				tmp_idx = np.mod(tmp_idx, self.num_of_tasks)
			paths.append(self.get_single_path( task_context ))
			# task_path_count += 1
			path_count += 1

			if verbose:
				sys.stdout.write('%i-th path sampled. simulation time: %f \r'%(path_count, time.time()-t))
				sys.stdout.flush()

			# task_step_count += len(paths[-1]['actions'])
			step_count += len(paths[-1]['actions'])

			if step_count >= num_of_steps or path_count >= num_of_paths:
				break
		if verbose:
			print('All paths sampled. Total sampled paths: %i. Total time usesd: %f.'%(path_count, time.time() - t) )
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
		if self.boost_baseline:
			self.baseline.fit(observations, returns, iter_num = 500)
			self.boost_baseline = False
		else:
			self.baseline.fit(observations, returns, iter_num = 5)

		return sample_data

	# def warm_up(self, paths, var_list = None):
	# 	# if isinstance(var_list, tuple): self.update_vars(var_list[0], var_list[1])

	# 	sample_data = self.process_paths(paths)
	# 	obs_source = sample_data['observations']
	# 	con_source = sample_data['contexts']
	# 	act_source = sample_data['actions']
	# 	adv_source = sample_data['advantages']
	# 	n_samples = sample_data['total_time_step']
	# 	actor_info_source = sample_data['actor_infos']
		
	# 	episode_rewards = np.array([np.sum(path['rewards']) for path in paths])
	# 	# batchsize = int(self.pms.num_of_paths * self.pms.max_time_step * self.num_of_tasks * self.pms.subsample_factor)
	# 	batchsize = int(self.pms.max_total_time_step * self.pms.subsample_factor)
	# 	train_number = int(np.ceil( 1.0 * n_samples / batchsize))
	# 	# train_number = int(1./self.pms.subsample_factor)
	# 	step_gradients = []

	# 	flat_theta_prev = flat_shared_prev = self.sess.run(self.flatten_shared_var)
	# 	old_l1_norm = np.array([self.sess.run(self.l1_norm)])

	# 	for iteration in range(train_number):
	# 		if n_samples > batchsize:
	# 			inds = np.random.choice(n_samples, batchsize, replace = False)
	# 		else:
	# 			inds = np.arange(n_samples)
	# 		obs_n = obs_source[inds]
	# 		con_n = con_source[inds]
	# 		act_n = act_source[inds]
	# 		adv_n = adv_source[inds]
	# 		act_dis_mean_n = np.array([a_info['mean'] for a_info in actor_info_source[inds]])
	# 		act_dis_logstd_n = np.array([a_info['logstd'] for a_info in actor_info_source[inds]])

	# 		feed_dict = {self.obs: obs_n,
	# 					 self.context: con_n,
	# 					 self.advant: adv_n,
	# 					 self.action: act_n,
	# 					 self.old_dist_mean: act_dis_mean_n,#[:,np.newaxis],
	# 					 self.old_dist_logstd: act_dis_logstd_n,#[:,np.newaxis]
	# 					 self.old_l1_norm: old_l1_norm
	# 					 }

	# 		def fisher_vector_product(p):
	# 			feed_dict[self.flat_tangent] = p
	# 			return self.sess.run(self.shared_flat_fvp, feed_dict = feed_dict) + self.pms.cg_damping * p

	# 		g = self.sess.run(self.flat_shared_surr_grad, feed_dict = feed_dict)
	# 		# step_gradients.append(self.sess.run(self.flat_surr_shared_grad, feed_dict = feed_dict))
	# 	# g = np.nanmean(step_gradients, axis = 0)
	# 		step_gradient = cg(fisher_vector_product, -g, cg_iters = self.pms.cg_iters)
	# 		sAs = step_gradient.dot( fisher_vector_product(step_gradient) )
	# 		inv_stepsize = np.sqrt( sAs/(2.*self.pms.max_kl) )
	# 		fullstep_gradient = step_gradient / (inv_stepsize + 1e-8)
	# 		step_gradients.append(fullstep_gradient)
		
	# 	fullstep_gradient = np.nanmean(step_gradients ,axis = 0)

	# 	def loss_function(x):
	# 		self.sess.run(self.set_shared_from_flat, feed_dict = {self.weights_to_set: x})
	# 		return self.sess.run([self.surr_loss, self.kl], feed_dict = feed_dict)

	# 	if self.pms.linesearch:
	# 		flat_theta_new = linesearch(loss_function, flat_shared_prev, fullstep_gradient, self.pms.max_backtracks, self.pms.max_kl)
	# 	else:
	# 		flat_theta_new = flat_shared_prev + fullstep_gradient
	# 	self.sess.run(self.set_shared_from_flat, feed_dict = {self.weights_to_set: flat_shared_prev})
	# 	update_gradient = flat_theta_new - flat_shared_prev
	# 	surrgate_loss, kl_divergence = loss_function(flat_theta_new)
	# 	l1_norm, l0_norm = self.sess.run([self.l1_norm, self.l0_norm])
	# 	stats = dict(
	# 		surrgate_loss = surrgate_loss,
	# 		kl_divergence = kl_divergence,
	# 		average_return = np.mean(episode_rewards),
	# 		total_time_step = n_samples,
	# 		l1_norm = l1_norm,
	# 		l0_norm = l0_norm
	# 		)
	# 	return flat_theta_new, flat_theta_prev, stats



	# def joint_learn(self, paths, var_list = None):
	# 	# if isinstance(var_list, tuple): self.update_vars(var_list[0], var_list[1])

	# 	sample_data = self.process_paths(paths)
	# 	obs_source = sample_data['observations']
	# 	con_source = sample_data['contexts']
	# 	act_source = sample_data['actions']
	# 	adv_source = sample_data['advantages']
	# 	n_samples = sample_data['total_time_step']
	# 	actor_info_source = sample_data['actor_infos']
		
	# 	episode_rewards = np.array([np.sum(path['rewards']) for path in paths])
	# 	# batchsize = int(self.pms.num_of_paths * self.pms.max_time_step * self.num_of_tasks * self.pms.subsample_factor)
	# 	batchsize = int(self.pms.max_total_time_step * self.pms.subsample_factor)
	# 	train_number = int(np.ceil( 1.0 * n_samples / batchsize))
	# 	# train_number = int(1./self.pms.subsample_factor)
	# 	step_gradients_shared = []
	# 	step_gradients_task = []
	# 	flat_theta_prev = self.sess.run(self.flatten_var)
	# 	flat_shared_prev = self.sess.run(self.flatten_shared_var)
	# 	flat_task_prev = self.sess.run(self.flatten_task_var)
	# 	old_l1_norm = np.array([self.sess.run(self.l1_norm)])


	# 	for iteration in range(train_number):
	# 		if n_samples > batchsize:
	# 			inds = np.random.choice(n_samples, batchsize, replace = False)
	# 		else:
	# 			inds = np.arange(n_samples)
	# 		obs_n = obs_source[inds]
	# 		con_n = con_source[inds]
	# 		act_n = act_source[inds]
	# 		adv_n = adv_source[inds]
	# 		act_dis_mean_n = np.array([a_info['mean'] for a_info in actor_info_source[inds]])
	# 		act_dis_logstd_n = np.array([a_info['logstd'] for a_info in actor_info_source[inds]])

	# 		feed_dict = {self.obs: obs_n,
	# 					 self.context: con_n,
	# 					 self.advant: adv_n,
	# 					 self.action: act_n,
	# 					 self.old_dist_mean: act_dis_mean_n,#[:,np.newaxis],
	# 					 self.old_dist_logstd: act_dis_logstd_n,#[:,np.newaxis]
	# 					 self.old_l1_norm: old_l1_norm
	# 					 }

	# 		def fisher_vector_product(p, grad_vars = None):
	# 			feed_dict[self.flat_tangent] = p
	# 			if grad_vars is None:
	# 				return self.sess.run(self.flat_fvp, feed_dict = feed_dict) + self.pms.cg_damping * p
	# 			elif grad_vars is 'shared':
	# 				return self.sess.run(self.shared_flat_fvp, feed_dict = feed_dict) + self.pms.cg_damping * p
	# 			elif grad_vars is 'task':
	# 				return self.sess.run(self.task_flat_fvp, feed_dict = feed_dict) + self.pms.cg_damping * p

	# 		g1 = self.sess.run(self.flat_shared_surr_grad, feed_dict = feed_dict)
	# 		g2 = self.sess.run(self.flat_task_surr_grad, feed_dict = feed_dict)

	# 		shared_step_gradient = cg(lambda x: fisher_vector_product(x, 'shared'), -g1, cg_iters = self.pms.cg_iters)
	# 		shared_sAs = shared_step_gradient.dot( fisher_vector_product(shared_step_gradient, grad_vars = 'shared') )
	# 		shared_inv_stepsize = np.sqrt( shared_sAs/(2.*self.pms.max_kl) )
	# 		shared_fullstep_gradient = shared_step_gradient / (shared_inv_stepsize + 1e-8)
	# 		step_gradients_shared.append(shared_fullstep_gradient)

	# 		task_step_gradient = cg(lambda x: fisher_vector_product(x, 'task'), -g2, cg_iters = self.pms.cg_iters)
	# 		task_sAs = task_step_gradient.dot( fisher_vector_product(task_step_gradient, grad_vars = 'task') )
	# 		task_inv_stepsize = np.sqrt( task_sAs/(2.*self.pms.max_kl) )
	# 		task_fullstep_gradient = task_step_gradient / (task_inv_stepsize + 1e-8)
	# 		step_gradients_task.append(task_fullstep_gradient)

	# 	shared_fullstep_gradient = np.nanmean(step_gradients_shared)
	# 	task_fullstep_gradient = np.nanmean(step_gradients_task)

	# 	def loss_function(x, grad_vars = None):
	# 		if grad_vars is None:
	# 			self.sess.run( self.set_var_from_flat, feed_dict = {self.weights_to_set: x})
	# 			return self.sess.run([self.surr_loss, self.kl], feed_dict = feed_dict)
	# 		elif grad_vars is 'shared':
	# 			self.sess.run(self.set_shared_from_flat, feed_dict = {self.weights_to_set: x})
	# 			return self.sess.run([self.surr_loss, self.kl], feed_dict = feed_dict)
	# 		elif grad_vars is 'task':
	# 			self.sess.run(self.set_task_from_flat, feed_dict = {self.weights_to_set: x})
	# 			return self.sess.run([self.surr_loss, self.kl ], feed_dict = feed_dict)

	# 	if self.pms.linesearch:
	# 		# flat_theta_new = linesearch(loss_function, flat_theta_prev, fullstep_gradient, self.pms.max_backtracks, self.pms.max_kl)
	# 		# flat_shared_new = linesearch(lambda x:loss_function(x, flat_task_prev), flat_shared_prev, fullstep_gradient[:self.shared_var_num], self.pms.max_backtracks, self.pms.max_kl)
	# 		# flat_task_new = linesearch(lambda x:loss_function(flat_shared_prev, x), flat_task_prev, fullstep_gradient[self.shared_var_num:], self.pms.max_backtracks, self.pms.max_kl)
	# 		# flat_theta_new = np.concatenate([flat_shared_new, flat_task_new], axis = 0)
	# 		flat_shared_new = linesearch( lambda x: loss_function(x, grad_vars = 'shared'), flat_shared_prev, shared_fullstep_gradient, self.pms.max_backtracks, self.pms.max_kl)
	# 		flat_task_new = linesearch( lambda x: loss_function(x, grad_vars = 'task'), flat_task_prev, task_fullstep_gradient, self.pms.max_backtracks, self.pms.max_kl)
	# 		flat_theta_new = np.concatenate([flat_shared_new, flat_task_new], axis = 0)
	# 	else:
	# 		flat_theta_new = flat_theta_prev + fullstep_gradient
	# 	update_gradient = flat_theta_new - flat_theta_prev
	# 	surrgate_loss, kl_divergence = loss_function(flat_theta_new)
	# 	l1_norm, l0_norm = self.sess.run([self.l1_norm, self.l0_norm])
	# 	stats = dict(
	# 		surrgate_loss = surrgate_loss,
	# 		kl_divergence = kl_divergence,
	# 		average_return = np.mean(episode_rewards),
	# 		total_time_step = n_samples,
	# 		l1_norm = l1_norm,
	# 		l0_norm = l0_norm
	# 		)
	# 	return flat_theta_new, flat_theta_prev, stats


	def train_paths(self, paths):
		sample_data = self.process_paths(paths)
		obs_source = sample_data['observations']
		act_source = sample_data['actions']
		adv_source = sample_data['advantages']
		con_source = sample_data['contexts']
		n_samples = sample_data['total_time_step']
		actor_info_source = sample_data['actor_infos']

		episode_rewards = np.array([np.sum(path['rewards']) for path in paths])
		train_number = int(1./self.pms.subsample_factor)
		step_gradients = []

		flat_theta_prev = self.sess.run(self.flatten_var)

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

			step_gradients.append( self.sess.run(self.flat_surr_grad, feed_dict = feed_dict) )

		g = np.nanmean(step_gradients, axis = 0)
		step_gradient = cg(fisher_vector_product, -g, cg_iters = self.pms.cg_iters)
		sAs = step_gradient.dot( fisher_vector_product(step_gradient) )
		inv_stepsize = np.sqrt( sAs/(2.*self.pms.max_kl) )
		fullstep_gradient = step_gradient / (inv_stepsize + 1e-8)

		def loss_function(x):
			self.sess.run( self.set_var_from_flat, feed_dict = {self.weights_to_set: x})
			surr_loss, kl = self.sess.run([self.surr_loss, self.kl], feed_dict = feed_dict	)
			# self.sess.run(set_from_flat(self.actor.var_list, flat_theta_prev))
			return surr_loss, kl
		if self.pms.linesearch:
				flat_theta_new = linesearch(loss_function, flat_theta_prev, fullstep_gradient, self.pms.max_backtracks, self.pms.max_kl)
		else:
				flat_theta_new = flat_theta_prev + fullstep_gradient
		self.sess.run(self.set_var_from_flat, feed_dict = {self.weights_to_set: flat_theta_prev})
		update_gradient = flat_theta_new - flat_theta_prev
		# step_gradients.append(flat_theta_new - flat_theta_prev)
		# flat_theta_new = flat_theta_prev + np.nanmean(step_gradients, axis = 0)
		surrgate_loss, kl_divergence = loss_function(flat_theta_new)
		stats = dict(
				surrgate_loss = surrgate_loss,
				kl_divergence = kl_divergence,
				average_return = np.mean(episode_rewards),
				total_time_step = n_samples
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
				'train_time', 'surrgate_loss', 'kl_divergence', 'iteration_number']
		saving_result = dict([(v, []) for v in dict_keys])

		for iter_num in range(self.pms.max_iter):
			print('\n******************* Iteration %i *******************'%iter_num)
			t = time.time()
			paths = self.get_paths()
			sample_time = time.time() - t
			t = time.time()
			theta, theta_old, stats = self.train_paths(paths)
			self.sess.run(self.set_var_from_flat, feed_dict = {self.weights_to_set: theta})
			train_time = time.time() - t

			for k, v in stats.items():
					print("%-20s: %15.5f"%(k,v))

			logstds = self.sess.run(self.actor.action_logstd)
			print(logstds)
			# test_variable = tf.get_variable('test_variable')
			# test_variable = [v for v in tf.trainable_variables() if 'test_variable' in v.name][0]

			# self.sess.run( tf.assign(test_variable, test_variable + 1.) )
			# print(self.sess.run(test_variable))


			save_value_list = [stats['average_return'], sample_time, stats['total_time_step'], \
					train_time, stats['surrgate_loss'], stats['kl_divergence'], iter_num]

			[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]
			# s_vector = self.sess.run(self.task_var_list)
			# print([s[3] for s in s_vector])
			if self.pms.save_model and iter_num % self.pms.save_model_iters == 0:
					self.save_model(self.pms.save_dir + self.pms.env_name + '-iter%i'%(iter_num + self.pms.pre_iter))

		return saving_result	


	'''
	def learn(self):
		dict_keys = ['average_return', 'sample_time', 'total_time_step', \
			'train_time', 'surrgate_loss', 'kl_divergence', 'iteration_number', 's_vector', 'l1_norm', 'l0_norm']
		saving_result = dict([(v, []) for v in dict_keys])

		
			Warm Up Stage
		
		context_size = self.pms.context_shape - 1
		if self.context_range is not None:
			task_context = np.random.rand(context_size) * (self.context_range) - self.context_range/2 # + self.default_context
		elif self.env_contexts is not None:
			tmp_idx = np.random.randint(self.num_of_tasks)
			tmp_idx = 4
			task_context = self.env_contexts[tmp_idx]
		shared_var_list = [v for v in self.shared_var_list if '_m0' in v.name]
		task_var_list = self.task_var_list
		var_list = (shared_var_list, task_var_list)
		self.update_vars(shared_var_list, task_var_list)
		for iter_num in range(self.pms.warm_up_iter):
			print('\n*****Warm up***** Iteration %i *******************'%iter_num)
			t = time.time()
			paths = self.get_paths(task_context = task_context)
			sample_time = time.time() - t
			t = time.time()
			theta, theta_old, stats = self.warm_up(paths)
			train_time = time.time() - t
			s_vector = self.sess.run( self.task_var_list )

			for k, v in stats.iteritems():
				print("%-20s: %15.5f"%(k,np.mean(v)))

			save_value_list = [stats['average_return'], sample_time, stats['total_time_step'], \
				train_time, stats['surrgate_loss'], stats['kl_divergence'], iter_num, s_vector, stats['l1_norm'], stats['l0_norm']]

			[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]

			if self.pms.save_model and iter_num % self.pms.save_model_iters == 0:
				self.save_model(self.pms.save_dir + self.pms.env_name + '-iter%i'%(iter_num))


		
			Joint Learn Stage
		
		shared_var_list = [v for v in self.shared_var_list if not '_m0' in v.name]
		# shared_var_list = self.shared_var_list
		task_var_list = self.task_var_list
		self.update_vars(shared_var_list, task_var_list)
		for i in range(self.pms.joint_learn_iter):
			iter_num = i + self.pms.warm_up_iter
			print('\n*****Joint Learn* Iteration %i *******************'%iter_num)
			t = time.time()
			paths = self.get_paths()
			sample_time = time.time() - t
			t = time.time()
			theta, theta_old, stats = self.joint_learn(paths)
			train_time = time.time() - t
			s_vector = self.sess.run( self.task_var_list )

			for k, v in stats.iteritems():
				print("%-20s: %15.5f"%(k,np.mean(v)))

			save_value_list = [stats['average_return'], sample_time, stats['total_time_step'], \
				train_time, stats['surrgate_loss'], stats['kl_divergence'], iter_num, s_vector, stats['l1_norm'], stats['l0_norm']]

			[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]

			if self.pms.save_model and iter_num % self.pms.save_model_iters == 0:
				self.save_model(self.pms.save_dir + self.pms.env_name + '-iter%i'%(iter_num))

		# for iter_num in range(self.pms.max_iter):
		# 	print('\n******************* Iteration %i *******************'%iter_num)
		# 	t = time.time()
		# 	paths = self.get_paths()
		# 	sample_time = time.time() - t
		# 	t = time.time()
		# 	theta, theta_old, stats = self.train_paths(paths)
		# 	# self.sess.run(set_from_flat(self.actor.var_list, np.mean(theta, axis = 0)))
		# 	train_time = time.time() - t
		# 	s_vector = self.sess.run( self.task_var_list )
		# 	# s_vector = [np.reshape( np.array(v), -1) for v in s_vector]

		# 	for k, v in stats.iteritems():
		# 		print("%-20s: %15.5f"%(k,np.mean(v)))

		# 	save_value_list = [stats['average_return'], sample_time, stats['total_time_step'], \
		# 		train_time, stats['surrgate_loss'], stats['kl_divergence'], iter_num, s_vector, stats['l1_norm'], stats['l0_norm']]

		# 	[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]

		# 	# if stats['surrgate_loss'] / stats['l1_norm'] > 0.2:
		# 	# 	self.pms.l1

		# 	if self.pms.save_model and iter_num % self.pms.save_model_iters == 0:
		# 		self.save_model(self.pms.save_dir + self.pms.env_name + '-iter%i'%(iter_num))

		return saving_result
	'''
