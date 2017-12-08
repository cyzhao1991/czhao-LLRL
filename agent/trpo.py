from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.krylov import cg
from utils.utils import *
from agent.agent import Agent

class TRPOagent(Agent):

	def __init__(self, env, actor, baseline, session, flags):
		super(TRPOagent, self).__init__(self, env, actor, baseline, session, flags)

		self.init_vars()

		print('Building Network')

	def init_vars(self):

		with tf.name_scope(pms.name_scope):
			self.obs = self.actor.input_ph
			self.advant = tf.placeholder(tf.float32, [None], name = 'advantages')
			self.action = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'action')
			self.old_dist_mean = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_mean')
			self.old_dist_logstd = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_std')
			self.new_dist_mean = self.actor.output_net
			self.new_dist_logstd = self.actor.action_logstd

			logli_new = log_likelihood(self.action, self.new_dist_mean, self.new_dist_logstd)
			logli_old = log_likelihood(self.action, self.old_dist_mean, self.old_dist_logstd)
			self.ratio = tf.exp(logli_new - logli_old)

			self.surr_loss = - tf.reduce_mean(self.ratio * self.advant)
			self.kl = kl_sym(self.old_dist_mean, self.old_dist_logstd, self.new_dist_mean, self.new_dist_logstd)

			surr_grad = tf.gradients(self.surr_loss, self.actor.var_list)
			self.flat_surr_grad = flatten_var(surr_grad)

			batchsize = tf.cast(tf.shape(self.obs)[0], tf.float32)
			self.flat_tangent = tf.placeholder(tf.float32, shape = [None], name = 'flat_tangent')
			kl_firstfixed = kl_firstfixed(self.new_dist_mean, self.new_dist_logstd)
			grads = tf.gradients(kl_firstfixed, self.actor.var_list) / batchsize
			flat_grads = flatten_var(flat_grads)
			self.fvp = tf.gradients(tf.reduce_sum(flat_grads * self.flat_tangent), self.actor.var_list)
			self.flat_fvp = flatten_var(self.fvp)


		# self.actor_var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.actor.name)]
		# self.baseline_var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.baseline.name)]

	def get_single_path(self):

		observations = []
		actions = []
		rewards = []
		actor_infos = []
		state = self.env.reset()

		if self.pms.render:
			self.env.render()

		for _ in range(self.pms.max_time_step):
			action, actor_info = actor.get_action(state)
			next_state, reward, terminal, _ = self.env.step(action)
			observations.append(state)
			actions.append(action)
			rewards.append(reward)
			actor_infos.append(actor_info)
			if terminal:
				break
			state = next_state
			if self.pms.render:
				self.env.render()
		path = dict(observations = np.array(observations), actions = np.array(actions), rewards = np.array(rewards), actor_infos = actor_infos)
		return path

	def get_paths(self, num_of_paths = None):
		if num_of_paths is None:
			num_of_paths = self.pms.num_of_paths
		paths = []
		t = time.time()
		print('Gathering Samples')
		for i in range(num_of_paths):
			paths.append(self.get_single_path())
			sys.stdout.write('%i-th path sampled. simulation time: %f \r'%(i, time.time()-t))
			sys.stdout.flush()
		print('%i paths sampled. Total time used: %f.'%(num_of_paths, time.time()-t))


	def process_paths(self, paths):
		total_time_step = 0
		for path in paths:
			total_time_step += len(path['rewards'])
			path['baselines'] = self.baseline.predict(path)
			path['returns'] = discount(path['rewards'], self.pms.discount)d
			path['advantages'] = path['returns'] - path['baselines']

		observations = np.concatenate([path['observations'] for path in paths])
		actions = np.concatenate([path['actions'] for path in paths])
		rewards = np.concatenate([path['rewards'] for path in paths])
		advantages = np.concatenate([path['advantages'] for path in paths])
		actor_infos = np.concatenate(path['actor_infos'] for path in paths])
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

	def train_paths(self, paths):
		sample_data = self.process_paths(paths)
		obs_source = sample_data['observations']
		act_source = sample_data['actions']
		adv_source = sample_data['advantages']
		n_samples = sample_data['total_time_step']
		actor_info_source = sample_data['actor_infos']
		
		episode_rewards = np.array([np.sum(path['rewards']) for path in paths])
		train_number = int(1./self.pms.subsample_factor)
		step_gradients = []

		flat_theta_prev = self.sess.run(flatten_var(self.actor.var_list))

		for iteration in range(train_number):
			inds = np.random.choice(n_samples, int(np.floor(n_samples*self.pms.subsample_factor)), replace = False)
			obs_n = obs_source[inds]
			act_n = act_source[inds]
			adv_n = adv_source[inds]
			act_dis_mean_n = np.array([a_info['mean'] for a_info in actor_info_source[inds]])
			act_dis_std_n = np.array([a_info['std'] for a_info in actor_info_source[inds]])

			feed_dict = {self.obs: obs_n,
						 self.advant: adv_n, 
						 self.action: act_n, 
						 self.old_dist_mean: act_dis_mean_n,
						 self.old_dist_logstd: act_dis_std_n
						 }

			def fisher_vector_product(p):
				feed_dict[self.flat_tangent] = p
				return self.sess.run(self.flat_fvp, feed_dict = feed_dict) + self.pms.cg_damping * p

			g = self.sess.run(self.flat_surr_grad, feed_dict = feed_dict)
			step_gradient = cg(fisher_vector_product, -g, cg_iters = self.pms.cg_iters)
			sAs = step_gradient.dot( fisher_vector_product(step_gradient) )
			inv_stepsize = np.sqrt( sAs/(2.*self.pms.max_kl) )
			fullstep_gradient = step_gradient / inv_stepsize

			def loss_function(x):
				self.sess.run(set_from_flat(self.actor.var_list, x))
				surr_loss, kl = self.sess.run([self.surr_loss, self.kl])
				# self.sess.run(set_from_flat(self.actor.var_list, flat_theta_prev))
				return surr_loss

			if self.pms.linesearch:
				flat_theta_new = linesearch(loss_function, flat_theta_prev, fullstep_gradient, self.pms.maxbacktracks, self.pms.max_kl)
			else:
				flat_theta_new = flat_theta_prev + fullstep_gradient

			step_gradients.append(flat_theta_new - flat_theta_prev)
			flat_theta_new = flat_theta_prev + np.mean(step_gradients, axis = 0)
			stats = dict(
				surrgate_loss = loss_function(flat_theta_new),
				average_return = np.mean(episode_rewards),
				total_time_step = n_samples
				)
			return flat_theta_new, flat_theta_prev, stats

	def learn(self):
		for iter_num in range(self.pms.max_iter):



