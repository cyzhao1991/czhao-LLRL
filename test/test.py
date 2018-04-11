from __future__ import print_function
import tensorflow as tf
import numpy as np
# import tensorflow as tf
# from model.fcnnMTL import FcnnMTL
import time,sys, argparse, os
# os.chdir('../')
# print(os.getcwd())
# from model.net import *
from dm_control.suite import walker

env = walker.stand()
state = env.reset()



# def main(argv):
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--gpu', default = 0, type = int)
# 	parser.add_argument('--exp', default = 0, type = int)
# 	args = vars(parser.parse_args())
# 	gpu_num = args['gpu']
# 	exp_num = args['exp']
# 	print(gpu_num, exp_num)

# if __name__ == "__main__":
# 	print(sys.argv)
# 	sys.argv[-1] = '4'
# 	main(sys.argv)
# sess = tf.Session()
# test_fcnn = Fcnn(sess, 10, 10,[100,50,25], name = 'fcnn')

# test_mod_fcnn = Modular_Fcnn(sess, 10, 10, [100, 50, 25], [10,10,10], name = 'mod_fcnn')

# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', default = 0, type = int)
# parser.add_argument('--exp', default = 0, type = int)
# args = vars(parser.parse_args())
# gpu_num = args['gpu']
# exp_num = args['exp']

# # with open('log.txt', 'a') as text_file:
# # 	text_file.write('gpu %i exp %i finished.\n'%(gpu_num, exp_num))

# def test_fun(a, **kwargs):
# 	testarg = kwargs['args'] if 'args' in kwargs.keys() else False
# 	print(testarg)
# 	# print(kwargs['testarg'])
# 	# print(kwargs.keys())
# 	# print([key, kwargs[key]] for key in kwargs)

# test_fun(10, arg1 = 10, arg2 = 20, args = 3)
# assert(10 == 10)