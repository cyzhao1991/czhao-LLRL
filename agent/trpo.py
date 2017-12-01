import tensorflow as tf
import numpy as np

class TRPOagent(object):

	def __init__(self, env, actor, session, flags):
		self.env = env
		self.actor = actor
		self.sess = session
		self.pms = flags

	def get_single_path(self):

		observations = []
		actions = []
		reward = []

		state = self.env.reset()

		if self.pms.render:
			self.env.render()

		for _ in range(self.pms.max_time_step)