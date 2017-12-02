import tensorflow as tf
import numpy as np

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
			


