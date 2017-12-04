from __future__ import print_function
# import tensorflow as tf
# from model.fcnnMTL import FcnnMTL
import time,sys

print('whatever sth')
t = time.time()
for i in range(10):
	sys.stdout.write('num: %i'%i+'\r')
	sys.stdout.flush()
	time.sleep(0.1)
print('finished print. time = %f'%(time.time()-t))
print('sth new')