from __future__ import print_function
import numpy as np
import tensorflow as tf
from model.mtl_net import Mtl_Fcnn_Net
from actor.mtl_actor import Mtl_Gaussian_Actor
from utils.paras import Paras_base
from agent.trpo_mtl import TRPO_MTLagent
from baseline.baseline import BaselineZeros
from env.cartpole import CartPoleEnv

sess = tf.Session()
input_dim = 5
output_dim = 10
layer_dim = [100,75,25]
module_num = [10,10,10]
num_of_tasks = 20

pms = Paras_base().pms

net = Mtl_Fcnn_Net(sess, input_dim, output_dim, layer_dim, module_num, num_of_tasks, \
	name = pms.name_scope, if_bias = [None], activation = ['tanh','tanh','tanh','None'])

actor = Mtl_Gaussian_Actor(net, sess, pms, num_of_tasks)
# for task_var in actor.task_var_list:
# 	[print(v.name) for v in task_var]
# 	print('-------------------')
gravity_list = [0., 4.9, 9.8]
mass_cart = [0.1, 0.5, 1.0]
mass_pole = [0.1, 0.5, 1.0]
env_paras_list = [(g, mc, mp) for g in gravity_list for mc in mass_cart for mp in mass_pole]
env_list = []
[env_list.append(CartPoleEnv(g, mc, mp)) for g,mc,mp in env_paras_list]
env_list = env_list[:20]
num_of_envs = len(env_list)

baseline = BaselineZeros(sess, pms)

learn_agent = TRPO_MTLagent( env_list, actor, baseline, sess, pms)