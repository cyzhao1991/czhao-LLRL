from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import sys, time, os, gym, shelve
# from utils.krylov import cg
# from utils.utils import *
# from agent.agent import Agent

from actor.actor import GaussianActor
from agent.trpo import TRPOagent
from baseline.baseline import BaselineZeros
from env.cartpole import CartPoleEnv
from model.fcnn import Fcnn
from utils.paras import Paras_base



pms = Paras_base().pms
# print(pms.max_iter)
pms.save_dir = '/disk/scratch/chenyang/Data/checkpoint/'

env = CartPoleEnv()
action_size = env.action_space.shape[0]
observation_size = env.observation_space.shape[0]
max_action = env.action_space.high[0]
pms.obs_shape = observation_size
pms.action_shape = action_size
pms.max_action = max_action
pms.num_of_paths = 1000
sess = tf.Session()

actor_net = Fcnn(sess, pms.obs_shape, pms.action_shape, 3, [100,50,25], name = pms.name_scope)
actor = GaussianActor(actor_net, sess, pms)
baseline = BaselineZeros(sess, pms)

learn_agent = TRPOagent(env, actor, baseline, sess, pms)

sess.run(tf.global_variables_initializer())

saving_result = learn_agent.learn()

filename = '/disk/scratch/chenyang/Data/checkpoint/shelve_result'
my_shelf = shelve.open(filename, 'n')
my_shelf['saving_result'] = saving_result
my_shelf.close()
