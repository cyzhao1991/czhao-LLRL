from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.krylov import cg
from utils.utils import *
from agent.agent import Agent
# from dm_control.rl import control
import matplotlib.pyplot as plt
from matplotlib import animation
class TRPOagent(Agent):

	def __init__(self, env, actor, baseline, session, flags, saver = None, goal = None):
		super(TRPOagent, self).__init__(env, session, flags, saver)
		self.actor = actor
		self.baseline = baseline
		self.var_list = self.actor.var_list
		self.goal = goal

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

			logli_new = log_likelihood(self.action, self.new_dist_mean, self.new_dist_logstd)
			logli_old = log_likelihood(self.action, self.old_dist_mean, self.old_dist_logstd)
			self.ratio = tf.exp(logli_new - logli_old)

			surr_loss1 = - self.ratio * self.advant
			surr_loss2 = - self.advant * tf.clip_by_value(self.ratio, 1.0-self.pms.cliprange, 1.0+self.pms.cliprange)
			self.surr_loss = tf.reduce_mean( tf.maximum(surr_loss1, surr_loss2) )

			self.kl = tf.reduce_mean(kl_sym(self.old_dist_mean, self.old_dist_logstd, self.new_dist_mean, self.new_dist_logstd))

			surr_grad = tf.gradients(self.surr_loss, self.actor.var_list)
			self.flat_surr_grad = flatten_var(surr_grad)

			batchsize = tf.cast(tf.shape(self.obs)[0], tf.float32)
			self.flat_tangent = tf.placeholder(tf.float32, shape = [None], name = 'flat_tangent')
			kl_firstfixed = kl_sym_firstfixed(self.new_dist_mean, self.new_dist_logstd)
			grads = tf.gradients(kl_firstfixed, self.actor.var_list) 
			
			flat_grads = flatten_var(grads) / batchsize
			self.fvp = tf.gradients(tf.reduce_sum(flat_grads * self.flat_tangent), self.actor.var_list)
			self.flat_fvp = flatten_var(self.fvp)

			self.weights_to_set = tf.placeholder(tf.float32, [None], name = 'weights_to_set')
			self.set_var_from_flat = set_from_flat(self.var_list, self.weights_to_set)
			self.flatten_var = flatten_var(self.var_list)

		# self.actor_var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.actor.name)]
		# self.baseline_var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.baseline.name)]

	def get_single_path(self, task_context = None):

		#img = []
		#fig = plt.figure(1)
		observations = []
		actions = []
		rewards = []
		actor_infos = []


		if task_context is not None:
			self.env.physics.model.opt.gravity[0] = 5. * task_context[1] #task_context[0]
			self.env.physics.model.opt.gravity[1] = 0. #task_context[1]
			self.env.physics.model.opt.gravity[2] = 5. * task_context[0] - 9.8

		state = self.env.reset()
		# state = control.flatten_observation(state[-1])
		# state = state.values()[0]



		if self.pms.render:
			self.env.render()
			# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
			# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			# out = cv2.VideoWriter('output_walk.mp4', fourcc, 50.0, (480,480))

			# frame = self.env.physics.render(height = 480, width = 480, camera_id = 0)
			# #cv2.imshow('simulation',frame)
			# #cv2.waitKey(10)
			# out.write(frame)

		for _ in range(self.pms.max_time_step):
			action, actor_info = self.actor.get_action(state)
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
			actor_infos.append(actor_info)
			# print('state', state)
			# print('action',action)
			if terminal:
				break
			state = next_state
			if self.pms.render:
				self.env.render()
				# frame = self.env.physics.render(height = 480, width = 480, camera_id = 0)
				# img.append([plt.imshow(frame, animated = True)])
				#cv2.imshow('simulation',frame)
				#cv2.waitKey(10)
				pass
				# out.write(frame)

		if self.pms.render:
			# out.release()
			#cv2.destroyAllWindows()
				# plt.show(img)
			pass
		# if self.pms.render:
		# 	animation.ArtistAnimation(fig, img, interval = 50, repeat = False, blit = True)
		# 	plt.show()
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

		self.baseline.fit(observations, returns)

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

		flat_theta_prev = self.sess.run(self.flatten_var)

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
						 self.old_dist_mean: act_dis_mean_n,#[:,np.newaxis],
						 self.old_dist_logstd: act_dis_logstd_n#[:,np.newaxis]
						 }

			def fisher_vector_product(p):
				feed_dict[self.flat_tangent] = p
				return self.sess.run(self.flat_fvp, feed_dict = feed_dict) + self.pms.cg_damping * p

			step_gradients.append( self.sess.run(self.flat_surr_grad, feed_dict = feed_dict) )
			# print([g, np.amax(g), np.amin(g)])
		g = np.nanmean(step_gradients, axis = 0)
		step_gradient = cg(fisher_vector_product, -g, cg_iters = self.pms.cg_iters)
		sAs = step_gradient.dot( fisher_vector_product(step_gradient) )
		inv_stepsize = np.sqrt( sAs/(2.*self.pms.max_kl) )
		fullstep_gradient = step_gradient / (inv_stepsize + 1e-8)

		def loss_function(x):
			self.sess.run( self.set_var_from_flat, feed_dict = {self.weights_to_set: x})
			surr_loss, kl = self.sess.run([self.surr_loss, self.kl], feed_dict = feed_dict)
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

			save_value_list = [stats['average_return'], sample_time, stats['total_time_step'], \
				train_time, stats['surrgate_loss'], stats['kl_divergence'], iter_num]

			[saving_result[k].append(v) for (k,v) in zip(dict_keys, save_value_list)]

			if self.pms.save_model and iter_num % self.pms.save_model_iters == 0:
				self.save_model(self.pms.save_dir + self.pms.env_name + '-iter%i'%(iter_num))

		return saving_result





