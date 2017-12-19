from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os
from utils.krylov import cg
from utils.utils import *
from agent import Agent


class TRPO_MTLagent(TRPOagent):

	def __init__(self, env_list, actor_list, baseline_list, session, flags):
		super(super(TRPO_MTLagent, self),self).__init__(env, actor, baseline, session, flags)

		self.init_vars()

		print('Building Network') 

	def init_vars(self):
		with tf.name_scope(self.pms.name_scope):
			self.obs = self.actor.input_ph
			self.advant = tf.placeholder(tf.float32, [None], name = 'advantages')