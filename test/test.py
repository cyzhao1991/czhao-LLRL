import tensorflow as tf
from model.fcnnMTL import FcnnMTL

sess = tf.Session()
a = FcnnMTL(sess, 10, 10, 2, [40,40], 10)