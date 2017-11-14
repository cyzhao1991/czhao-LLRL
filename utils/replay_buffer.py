import numpy as np
from scipy.stats import rankdata
import random

class ReplayBuffer(object):

	def __init__(self, buffer_size = 100000, seed = None):
		
		self.buffer_size = buffer_size
		self.seed = seed
		self.buffer = []
		self.count = 0
		if self.seed is not None:
			random.seed(self.seed)

	def add_sample(self, state, action, reward, next_state, context):#, time, error):
		self.buffer.append( (state, action, reward, next_state, context) )#, time, error) )
		self.count += 1

	def rand_sample(self, batch_size = 64, seed = None, method = 'rank'):
		
		if seed is not None:
			self.set_seed(seed)

		# p = None
		# if method == 'rank':
		# 	p = np.array([_[-1] for _ in self.buffer])
		# 	p = (1 / (self.count - rankdata(p))) ** 0.7
		# 	p = p / np.sum(p)
		# 	if batch_size < self.count:
		# 		indices = np.random.choice( self.count, batch_size, p = p)
		# 	else:
		# 		indices = np.arange(self.count)



		# elif method == 'prop':
		# 	p = np.array([_[-1] for _ in self.buffer])

		

		if batch_size < self.count:
			sample_batch = random.sample(self.buffer, batch_size)
		else:
			sample_batch = self.buffer

		s_batch = np.array([_[0] for _ in sample_batch])
		a_batch = np.array([_[1] for _ in sample_batch])
		r_batch = np.reshape( np.array([_[2] for _ in sample_batch]), (-1, 1))
		s2_batch = np.array([_[3] for _ in sample_batch])
		# c_batch = np.reshape( np.array([_[4] for _ in sample_batch]), (-1, 1))
		c_batch = np.array([_[4] for _ in sample_batch])
		return s_batch, a_batch, r_batch, s2_batch, c_batch#, w_batch, indices

	def update_batch(self, batch_indices, new_error):
		for index, error in zip(batch_indices, new_error):
			self.buffer[index][-1] = error

	def reset(self):
		self.count = 0
		del self.buffer[:]

	def set_seed(self, seed = None):
		self.seed = seed
		if self.seed is not None:
			random.seed(self.seed)