from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.krylov import cg
from utils.utils import *

class Agent(object):

	def __init__(self, env, session, flags, saver = None):
		self.env = env
		self.sess = session
		self.pms = flags

		if self.pms.save_model:
			self.saver = tf.train.Saver(max_to_keep = 2) if saver is None else saver
	def learn(self):
		raise NotImplementedError

	def save_model(self, filename):
		if not filename.endswith('.ckpt'):
			filename += '.ckpt'
		# filename = self.pms.save_dir + filename
		self.saver.save(self.sess, filename, write_meta_graph = False)

	def load_model(self, filename = None):
		if filename is None:
			filename = tf.train.lastest_checkpoint(self.pms.save_dir)
		try:
			self.save.restore(self.session, filename)
			print('load model %s success'%(filename))
		except:
			print('load model %s failed'%(filename))
