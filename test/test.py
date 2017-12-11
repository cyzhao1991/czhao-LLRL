from __future__ import print_function
# import tensorflow as tf
# from model.fcnnMTL import FcnnMTL
import time,sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default = 0, type = int)
parser.add_argument('--exp', default = 0, type = int)
print(parser.parse_args())
args = vars(parser.parse_args())
print(args['exp'])