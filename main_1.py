from __future__ import print_function
from trpo_main import main
import numpy as np
import tensorflow as tf
# from env.cartpole import CartPoleEnv
from gym.envs.mujoco.reacher import ReacherEnv

import sys, time, argparse

'''
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default = 0, type = int)
parser.add_argument('--task', default = 0, type = int)
parser.add_argument('--num_of_exps', default = 5, type = int)
parser.add_argument('--num_of_paths', default = 10, type = int)

args = parser.parse_args()
gpu_num = args.gpu
task_num = args.task
num_of_paths = args.num_of_paths
num_of_exps = args.num_of_exps
'''
gpu_num = 0
# task_num = 2
# tmp_num = int(sys.argv[1])
# task_num = int(sys.argv[2])
# some_list = [range(7), range(7,14), range(14,21), range(21,27)]
# gravity_list = [0., 4.9, 9.8]
# mass_cart = [0.1, 0.5, 1.0]
# mass_pole = [0.1, 0.5, 1.0]

# gravity_list = np.arange(0.2, 2.1, .2) * 9.8

# env_paras_list = [(g, 1.0, 0.1) for g in gravity_list]
# env_list = []
# [env_list.append(CartPoleEnv(g, mc, mp)) for g,mc,mp in env_paras_list]

# goal_list = np.zeros([0,2])
# while len(goal_list) < 10:
# 	goal = np.random.rand(1,2) * 0.4 - 0.2
# 	if np.linalg.norm(goal) < 0.2:
# 		goal_list = np.append(goal_list, goal, axis = 0)

# for i in range(num_of_exps):
# 	tf.reset_default_graph()
# 	main(gpu_num, i, env_list[task_num], task_num = task_num, num_of_paths = 10)

gravity_list = [-5., -2.5, 0., 2.5, 5.]
wind_list = [-3., -1.5, 0., 1.5, 3.]
target_speed = [-4., -1., 0., 1., 4.]
# for g in gravity_list:
for i in range(5):
	tf.reset_default_graph()
	main(gpu_num, i, gravity = gravity_list[2], wind = wind_list[0], speed = target_speed[0])
