from __future__ import print_function
from trpo_main import main
import numpy as np
import tensorflow as tf
from env.cartpole import CartPoleEnv
import sys, time

tmp_num = int(sys.argv[1])
some_list = [range(7), range(7,14), range(14,21), range(21,27)]
gravity_list = [0., 4.9, 9.8]
mass_cart = [0.1, 0.5, 1.0]
mass_pole = [0.1, 0.5, 1.0]
env_paras_list = [(g, mc, mp) for g in gravity_list for mc in mass_cart for mp in mass_pole]
env_list = []
[env_list.append(CartPoleEnv(g, mc, mp)) for g,mc,mp in env_paras_list]

for i in some_list[tmp_num]:
	tf.reset_default_graph()
	main(tmp_num, i, env_list[i])

