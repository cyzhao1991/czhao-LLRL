from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, pdb, multiprocessing
from utils.krylov import cg
from utils.utils import *
from agent.agent import Agent

class PpoMtl(Agent):
	def __init__(self, envs, mtl_actor, baselines, session, pms, env_contexts = None, saver = None,**kwargs):
		super(PpoMtl, self).__init__(envs, session, pms, saver)
		self.actor = mtl_actor
		self.baseline = baselines
		self.task_diff = 'task' if isinstance(envs, list) else 'context'
		self.num_of_tasks = len(envs) if self.task_diff is 'task' else len(env_contexts)
		self.env_contexts = env_contexts

		self.var_list = self.actor.var_list
		self.shared_var_list = self.actor.shared_var_list
		self.task_var_lists = self.actor.task_var_list

		# for baseline in self.baseline:
		# 	baseline.boost_baseline = True 
		# self.boost_baseline = [True] * self.num_of_tasks


	def init_vars(self):

		with tf.name_scope(self.pms.name_scope):
			self.obs = self.actor.input_ph
			self.advant = tf.placeholder(tf.float32, [None], name = 'advantages')
			self.action = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'action')
			self.old_dist_mean = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_mean')
			self.old_dist_logstd = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_logstd')
			self.task_descriptor = tf.placeholder(tf.float32, [None, self.num_of_tasks], name = 'task_descriptor')
			self.dist_mean = self.actor.output_net
			self.new_dist_mean = tf.reduce_sum( tf.stack( self.dist_mean, axis = 1) * tf.expand_dims( self.task_descriptor, axis = 2), axis = 1)
			self.dist_logstd = [tf.tile(lsd, [ tf.shape(self.obs)[0], 1 ] ) for lsd in self.actor.action_logstd]
			self.new_dist_logstd = tf.reduce_sum( tf.stack( self.dist_logstd, axis = 1) * tf.expand_dims(self.task_descriptor, axis = 2), axis = 1)

			logli_new = log_likelihood(self.action, self.new_dist_mean, self.new_dist_logstd)
			logli_old = log_likelihood(self.action, self.old_dist_mean, self.old_dist_logstd)

			self.ratio = tf.exp(logli_new - logli_old)
			surr_loss1 = -self.ratio * self.advant
			surr_loss2 = -tf.clip_by_value( self.ratio, 1.-self.pms.cliprange, 1.+self.pms.cliprange)
			self.surr_loss = tf.reduce_mean(tf.maximum(surr_loss1, surr_loss2))

			self.optimizer = tf.train.AdamOptimizer( learning_rate = self.pms.ppo_lr, epsilon = 1e-5 )
			self.train_op = self.optimizer.minimize(  self.surr_loss, var_list = self.var_list )
			self.kl = tf.reduce_mean( kl_sym(self.old_dist_mean, self.old_dist_logstd, self.new_dist_mean, self.new_dist_logstd) )

	def init_align_vars(self):

		with tf.name_scope(self.pms.name_scope):
			self.obs = self.actor.input_ph
			self.advant = tf.placeholder(tf.float32, [None], name = 'advantages')
			self.action = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'action')
			self.old_dist_mean = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_mean')
			self.old_dist_logstd = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_logstd')
			self.task_descriptor = tf.placeholder(tf.float32, [None, self.num_of_tasks], name = 'task_descriptor')
			self.dist_mean = self.actor.output_net
			self.new_dist_mean = tf.reduce_sum( tf.stack( self.dist_mean, axis = 1) * tf.expand_dims( self.task_descriptor, axis = 2), axis = 1)
			self.dist_logstd = [tf.tile(lsd, [ tf.shape(self.obs)[0], 1 ] ) for lsd in self.actor.action_logstd]
			self.new_dist_logstd = tf.reduce_sum( tf.stack( self.dist_logstd, axis = 1) * tf.expand_dims(self.task_descriptor, axis = 2), axis = 1)

			logli_new = log_likelihood(self.action, self.new_dist_mean, self.new_dist_logstd)
			logli_old = log_likelihood(self.action, self.old_dist_mean, self.old_dist_logstd)

			self.ratio = tf.exp(logli_new - logli_old)
			surr_loss1 = -self.ratio * self.advant
			surr_loss2 = -tf.clip_by_value( self.ratio, 1.-self.pms.cliprange, 1.+self.pms.cliprange)
			surr_loss = tf.maximum(surr_loss1, surr_loss2)
			task_surr_loss = tf.reduce_mean(tf.expand_dims(surr_loss, axis = 1) * self.task_descriptor, axis = 0)
			# print ( flatten_var(tf.gradients(task_surr_loss[0] )
			tmp_gradients = [tf.gradients( task_surr_loss[i], self.shared_var_list) for i in range(self.num_of_tasks) ]
			task_gradients = [flatten_var(g) for g in tmp_gradients]
			# task_gradients = [flatten_var( tf.gradients(task_surr_loss[i], self.shared_var_list) ) for i in range(self.num_of_tasks)]
			self.alignment_penalty = tf.add_n( [ tf.reduce_sum( task_gradients[0] * task_gradients[i+1] ) for i in range(self.num_of_tasks-1)] )
			self.surr_loss = tf.reduce_mean(surr_loss)

			# self.gradients = [flatten_var(tf.gradients(self.surr_loss, var_list = self.var_list)) for ]

			self.optimizer = tf.train.AdamOptimizer( learning_rate = self.pms.ppo_lr, epsilon = 1e-5 )
			self.train_op = self.optimizer.minimize(  self.surr_loss + 1. * self.alignment_penalty, var_list = self.var_list )
			self.kl = tf.reduce_mean( kl_sym(self.old_dist_mean, self.old_dist_logstd, self.new_dist_mean, self.new_dist_logstd) )


	def get_single_path(self, task_index):
		observations = []
		actions = []
		rewards = []
		descriptor = []
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
		task_context = np.zeros(self.num_of_tasks).astype(np.float32)
		task_context[task_index] = 1.
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
			descriptor.append(task_context)
			if terminal:
				break
			state = next_state
			if self.pms.render:
				env.render()
		if self.pms.render:
			pass
		path = dict(observations = np.array(observations), actions = np.array(actions), rewards = np.array(rewards), descriptor = np.array(descriptor), actor_infos = actor_infos)
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

	def process_paths(self, paths, fit = True):


		total_time_step = 0
		for pp, baseline in zip(paths, self.baseline):
			for path in pp:
				total_time_step += len(path['rewards'])
				path['baselines'] = baseline.predict(path)
				path['returns'] = discount(path['rewards'], self.pms.discount)
				if not self.pms.gae_flag:
					path['advantages'] = path['returns'] - path['baselines']
				else:
					b = np.append(path['baselines'], path['baselines'][-1])
					deltas = path['rewards'] + self.pms.discount * b[1:] - b[:-1]
					path['advantages'] = discount(deltas, self.pms.discount * self.pms.gae_lambda)
			tmp_obs = np.concatenate([path['observations'] for path in pp])
			tmp_rns = np.concatenate([path['returns'] for path in pp])
			baseline.fit(tmp_obs, tmp_rns, iter_num = 5)

		paths = sum(paths, [])

		observations = np.concatenate([path['observations'] for path in paths])
		actions = np.concatenate([path['actions'] for path in paths])
		rewards = np.concatenate([path['rewards'] for path in paths])
		advantages = np.concatenate([path['advantages'] for path in paths])
		returns = np.concatenate([path['returns'] for path in paths])
		actor_infos = np.concatenate([path['actor_infos'] for path in paths])
		descriptor = np.concatenate([path['descriptor'] for path in paths])
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
			descriptor = descriptor,
			total_time_step = total_time_step
		)

		# if fit:
		# 	if baseline.boost_baseline:
		# 		baseline.fit(observations, returns, iter_num = 500)
		# 		baseline.boost_baseline = False
		# 	else:
		# 		baseline.fit(observations, returns, iter_num = 5)
		# else:
		# 	pass

		return sample_data

	def train_paths(self, paths):

		episode_rewards = [np.mean(np.array([np.sum(path['rewards']) for path in pp])) for pp in paths]

		sample_data = self.process_paths(paths)
		obs_source = sample_data['observations']
		act_source = sample_data['actions']
		adv_source = sample_data['advantages']
		n_samples = sample_data['total_time_step']
		actor_info_source = sample_data['actor_infos']
		rtn_source = sample_data['returns']
		des_source = sample_data['descriptor']
		# val_source = np.squeeze(self.sess.run(self.vpred, feed_dict = {self.vf_input: obs_source}))

		batchsize = n_samples//self.pms.nbatch

		inds = np.arange(n_samples)
		for iteration in range(self.pms.nepochs):
			datalogger = []
			np.random.shuffle(inds)
			# for start in range(0, n_samples, batchsize):
			# end = start + batchsize
			# mb_inds = inds[start:end]
			mb_inds = inds[:self.pms.nbatch]

			obs_n = obs_source[mb_inds]
			act_n = act_source[mb_inds]
			adv_n = adv_source[mb_inds]
			des_n = des_source[mb_inds]
			act_dis_mean_n = np.array([a_info['mean'] for a_info in actor_info_source[mb_inds]])
			act_dis_logstd_n = np.array([a_info['logstd'] for a_info in actor_info_source[mb_inds]])
			# val_n = val_source[mb_inds]
			# rtn_n = rtn_source[mb_inds]
			feed_dict = {self.obs: obs_n,
						 self.advant: adv_n,
						 self.action: act_n,
						 self.old_dist_mean: act_dis_mean_n,
						 self.old_dist_logstd: act_dis_logstd_n,
						 self.task_descriptor: des_n
						 # self.vf_input: obs_n,
						 # self.oldvpred: val_n,
						 # self.v: rtn_n
						}

			# datalogger.append(self.sess.run([self.surr_loss, self.vf_loss, self.entropy, self.kl, self.clipfrac, self.train_op], feed_dict = feed_dict)[:-1])
			datalogger.append(self.sess.run([self.surr_loss, self.kl, self.train_op], feed_dict = feed_dict))
		stats = dict(
			surrgate_loss = np.mean([d[0] for d in datalogger], axis = -1),
			# vf_loss = np.mean([d[1] for d in datalogger], axis = -1),
			# entropy = np.mean([d[2] for d in datalogger], axis = -1),
			kl_divergence = np.mean([d[1] for d in datalogger], axis = -1),
			average_return = np.array(episode_rewards),
			total_time_step = n_samples
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


