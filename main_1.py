from __future__ import print_function
from trpo_main import main
import numpy as np
import tensorflow as tf
from env.cartpole import CartPoleEnv
import sys, time

tmp_num = int(sys.argv[1])
task_num = int(sys.argv[2])
# some_list = [range(7), range(7,14), range(14,21), range(21,27)]
# gravity_list = [0., 4.9, 9.8]
# mass_cart = [0.1, 0.5, 1.0]
# mass_pole = [0.1, 0.5, 1.0]

gravity_list = np.arange(0.2, 2.1, .2) * 9.8

env_paras_list = [(g, 1.0, 0.1) for g in gravity_list]
env_list = []
[env_list.append(CartPoleEnv(g, mc, mp)) for g,mc,mp in env_paras_list]

for i in range(10):
	tf.reset_default_graph()
	main(tmp_num, i, env_list[task_num], task_num)