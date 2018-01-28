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
		self.var_list = self.actor.var_list

		self.init_vars()

		print('Building Network')


	def init_vars(self):
		with tf.name_scope(self.pms.name_scope):
			self.obs = self.actor.input_ph
			if self.pms.with_context:
				self.contexts = self.actor.context_ph
			self.advant = tf.placeholder(tf.float32, [None], name = 'advantages')
			self.action = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'action')
			self.old_dist_mean = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_mean')
			self.old_dist_logstd = tf.placeholder(tf.float32, [None, self.pms.action_shape], name = 'old_dist_logstd')
			self.new_dist_mean = self.actor.output_net
			self.new_dist_logstd = self.actor.action_logstd

			logli_new = log_likelihood(self.action, self.new_dist_mean, self.new_dist_logstd)
			logli_old = log_likelihood(self.action, self.old_dist_mean, self.old_dist_logstd)
			self.ratio = tf.exp(logli_new - logli_old)