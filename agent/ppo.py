from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.krylov import cg
from utils.utils import *
from agent import Agent

class PPOagent(Agent):

	def __init__(self, env, actor, baseline, session, flags, saver = None):
		super(PPOagent, self).__init__(env, session, flags, saver)
		self.actor = actor
		self.baseline = baseline
		self.var_list = self.actor.var_list + self.baseline.var_list
		self.actor_var = self.actor.var_list
		self.baseline_var = self.baseline.var_list
		self.init_vars()

		print('Building Network')


	def init_vars(self):
		with tf.name_scope(self.pms.name_scope):
			self.obs = self.actor.input_ph
			self.advant = tf.placeholder(tf.float32, [None], name = 'advantages')
			self.action = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'action')
			self.old_dist_mean = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_mean')
			self.old_dist_logstd = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_logstd')
			self.new_dist_mean = self.actor.output_net
			self.new_dist_logstd = self.actor.action_logstd
			self.oldvpred = tf.placeholder(tf.float32, [None], name = 'oldvpred')

			logli_new = log_likelihood(self.action, self.new_dist_mean, self.new_dist_logstd)
			logli_old = log_likelihood(self.action, self.old_dist_mean, self.old_dist_logstd)
			self.ratio = tf.exp(logli_new - logli_old)

			self.vf_input = self.baseline.input
			self.vpred = self.baseline.output
			self.v = self.baseline.value
			vpredclipped = self.oldvpred + tf.clip_by_value(self.vpred - self.oldvpred, -self.pms.cliprange, self.pms.cliprange)
			vf_losses1 = tf.square(self.vpred - self.v)
			vf_losses2 = tf.square(vpredclipped - self.v)
			self.vf_loss = tf.reduce_mean( tf.maximum(vf_losses1, vf_losses2) )

			surr_loss1 = - self.ratio * self.advant
			surr_loss2 = - self.advant * tf.clip_by_value(self.ratio, 1.0-self.pms.cliprange, 1.0+self.pms.cliprange)
			self.surr_loss = tf.reduce_mean( tf.maximum(surr_loss1, surr_loss2) )
			self.kl = tf.reduce_mean(kl_sym(self.old_dist_mean, self.old_dist_logstd, self.new_dist_mean, self.new_dist_logstd))

			self.entropy = tf.reduce_mean(entropy(self.new_dist_logstd) )

			self.clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(self.ratio - 1.0), self.pms.cliprange)))
			self.total_loss = self.surr_loss - self.entropy * self.pms.entropy_coeff + self.vf_loss * self.pms.vf_coeff

			grads = tf.gradients(self.total_loss, self.var_list)
			grads = list(zip(grads, self.var_list))
			self.optimizer = tf.train.AdamOptimizer(learning_rate = self.pms.ppo_lr, epsilon = 1e-5)
			self.train_op = self.optimizer.apply_gradients(grads)

	def get_single_path(self):

		observations = []
		actions = []
		rewards = []
		actor_infos = []
		state = self.env.reset()

		if self.pms.render:
			self.env.render()

		for _ in range(self.pms.max_time_step):
			action, actor_info = self.actor.get_action(state)
			action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)
			next_state, reward, terminal, _ = self.env.step(self.pms.max_action * action)
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

	def get_paths(self, prefix = '', verbose = True):
		if not self.pms.max_total_time_step == 0: 
			num_of_paths = np.inf
			num_of_steps = self.pms.max_total_time_step
		else:
			num_of_paths = self.pms.num_of_paths
			num_of_steps = np.inf

		paths = []
		t = time.time()
		path_count = 0
		step_count = 0

		if verbose:
			print(prefix+'Gathering Samples')
		while True:
		# for i in range(num_of_paths):
			paths.append(self.get_single_path())
			path_count += 1
			if verbose:
				sys.stdout.write('%i-th path sampled. simulation time: %f \r'%(path_count, time.time()-t))
				sys.stdout.flush()
			
			step_count += len(paths[-1]['actions'])
			if path_count >= num_of_paths or step_count >= num_of_steps:
				break
		
		if verbose:
			print('%i paths sampled. Total sampling time used: %f'%(path_count, time.time()-t))
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
		returns = np.concatenate([path['returns'] for path in paths])
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

		return sample_data

	def train_paths(self, paths):
		sample_data = self.process_paths(paths)
		obs_source = sample_data['observations']
		act_source = sample_data['actions']
		adv_source = sample_data['advantages']
		n_samples = sample_data['total_time_step']
		actor_info_source = sample_data['actor_infos']
		rtn_source = sample_data['returns']

		val_source = np.squeeze(self.sess.run(self.vpred, feed_dict = {self.vf_input: obs_source}))
		episode_rewards = np.array([np.sum(path['rewards']) for path in paths])

		batchsize = n_samples//self.pms.nbatch

		inds = np.arange(n_samples)
		for iteration in range(self.pms.nepochs):
			datalogger = []
			np.random.shuffle(inds)
			for start in range(0, n_samples, batchsize):
				end = start + batchsize
				mb_inds = inds[start:end]

				obs_n = obs_source[mb_inds]
				act_n = act_source[mb_inds]
				adv_n = adv_source[mb_inds]
				act_dis_mean_n = np.array([a_info['mean'] for a_info in actor_info_source[mb_inds]])
				act_dis_logstd_n = np.array([a_info['logstd'] for a_info in actor_info_source[mb_inds]])
				val_n = val_source[mb_inds]
				rtn_n = rtn_source[mb_inds]
				feed_dict = {self.obs: obs_n,
							 self.advant: adv_n,
							 self.action: act_n,
							 self.old_dist_mean: act_dis_mean_n,
							 self.old_dist_logstd: act_dis_logstd_n,
							 self.vf_input: obs_n,
							 self.oldvpred: val_n,
							 self.v: rtn_n
							}

				datalogger.append(self.sess.run([self.surr_loss, self.vf_loss, self.entropy, self.kl, self.clipfrac, self.train_op], feed_dict = feed_dict)[:-1])

		stats = dict(
			surrgate_loss = np.mean([d[0] for d in datalogger], axis = -1),
			vf_loss = np.mean([d[1] for d in datalogger], axis = -1),
			entropy = np.mean([d[2] for d in datalogger], axis = -1),
			kl_divergence = np.mean([d[3] for d in datalogger], axis = -1),
			average_return = np.mean(episode_rewards),
			total_time_step = n_samples
			)

		return stats


	def learn(self):
		dict_keys = ['average_return', 'sample_time', 'total_time_step', 'vf_loss', 'entropy',\
			'train_time', 'surrgate_loss', 'kl_divergence', 'iteration_number']
		saving_result = dict([(v, []) for v in dict_keys])

		for iter_num in range(self.pms.max_iter):
			print('\n******************* Iteration %i *******************'%iter_num)
			t = time.time()
			paths = self.get_paths()
			sample_time = time.time() - t
			t = time.time()
			stats = self.train_paths(paths)
			# self.sess.run(self.set_var_from_flat, feed_dict = {self.weights_to_set: theta})
			train_time = time.time() - t

			for k, v in stats.iteritems():
				print("%-20s: %15.5f"%(k,v))

			save_value_list = [stats['average_return'], sample_time, stats['total_time_step'], stats['vf_loss'], stats['entropy'], \
				train_time, stats['surrgate_loss'], stats['kl_divergence'], iter_num]

			[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]

			if self.pms.save_model and iter_num % self.pms.save_model_iters == 0:
				self.save_model(self.pms.save_dir + self.pms.env_name + '-iter%i'%(iter_num))

		return saving_result

