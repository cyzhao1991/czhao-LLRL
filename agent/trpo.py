import tensorflow as tf
import numpy as np
import scipy.signal

def discount(x, gamma):

	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis = 0)[::-1]

class TRPOagent(object):

	def __init__(self, env, actor, baseline, session, flags):
		self.env = env
		self.actor = actor
		self.baseline = baseline
		self.sess = session
		self.pms = flags

	def get_single_path(self):

		observations = []
		actions = []
		rewards = []

		state = self.env.reset()

		if self.pms.render:
			self.env.render()

		for _ in range(self.pms.max_time_step):
			action, _ = actor.get_action(state)
			next_state, reward, terminal, _ = self.env.step(action)
			observations.append(state)
			actions.append(action)
			rewards.append(reward)

			if terminal:
				break
			state = next_state
			if self.pms.render:
				self.env.render()
		path = dict(observations = np.array(observations), actions = np.array(actions), rewards = np.array(rewards))
		return path

	def get_paths(self, num_of_paths = None):
		if num_of_paths is None:
			num_of_paths = self.pms.num_of_paths
		paths = []
		for _ in range(num_of_paths):
			paths.append(self.get_single_path())


	def process_paths(self, paths):
		total_time_step = 0
		for path in paths:
			total_time_step += len(path['rewards'])
			path['baselines'] = self.baseline.predict(path)
			path['returns'] = discount(path['rewards'], self.pms.discount)
			path['advantages'] = path['returns'] - path['baselines']

		observations = np.concatenate([path['observations'] for path in paths])
		actions = np.concatenate([path['actions'] for path in paths])
		rewards = np.concatenate([path['rewards'] for path in paths])
		advantages = np.concatenate([path['advantages'] for path in paths])

		if self.pms.center_adv:
			advantages -= np.mean(advantages)
			advantages /= (np.std(advantages) + 1e-8)

		sample_data = dict(
			observations = observations,
			actions = actions,
			rewards = rewards,
			advantages = advantages
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
		

