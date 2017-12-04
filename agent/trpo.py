import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os

def discount(x, gamma):

	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis = 0)[::-1]

class TRPOagent(object):

	def __init__(self, env, actor, baseline, session, flags):
		self.env = env
		self.actor = actor
		self.baseline = baseline
		self.sess = session
		self.pms = flags

	def init_vars(self):

		with tf.name_scope(pms.name_scope):
			self.advant = tf.placeholder(tf.float32, [None], name = 'advantages')
			self.

		self.actor_var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.actor.name)]
		self.baseline_var_list = [v for v in tf.trainable_variables() if v.name.startswith(self.baseline.name)]
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

		for iteration in range(train_number):
			inds = np.random.choice(n_samples, int(np.floor(n_samples*self.pms.subsample_factor)), replace = False)
			obs_n = obs_source[inds]
			act_n = act_source[inds]
			adv_n = adv_source[inds]
			act_dis_mean_n = np.array([a_info['mean'] for a_info in actor_info_source[inds]])
			act_dis_std_n = np.array([a_info['std'] for a_info in actor_info_source[inds]])



