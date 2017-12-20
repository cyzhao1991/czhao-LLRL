from __future__ import print_function
import numpy as np
import tensorflow as tf
from model.net import * 

xs = np.random.normal(0., 1., [15000, 20])
thetas = np.random.normal(0., 1., [3, 21]) + 1.
ys = [np.dot( np.append(xs, np.ones([15000,1]), axis = 1), th ) for th in thetas]
ys = [y[:, np.newaxis] for y in ys]
def data_split(xs, ys, training_size, testing_size):
	index = np.random.permutation( len(ys) )
	train_ind = index[:training_size]
	test_ind = index[training_size:training_size+testing_size]
	training_set = {'xs': xs[train_ind], 'ys': ys[train_ind]}
	testing_set = {'xs': xs[test_ind], 'ys': ys[test_ind]}
	return training_set, testing_set

train_set, test_set = data_split( xs, ys[0], 150, 350)

sess = tf.Session()
net = Fcnn(sess, 20, 1, [100, 50])

input_ph = net.input
predict = net.output
label = tf.placeholder(tf.float32, [None,1])
var_list = [v for v in tf.trainable_variables()]
l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in var_list])
loss = tf.losses.mean_squared_error(predict, label)
total_loss = loss + .01* l2_loss

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(total_loss)

sess.run(tf.global_variables_initializer())

train_losses = []
test_losses = []

for i in range(10000):

	_, train_loss= sess.run([train_op, total_loss], feed_dict = {input_ph: train_set['xs'], label:train_set['ys']})
	test_loss = sess.run(loss, feed_dict = {input_ph: test_set['xs'], label: test_set['ys']})
	# print(train_loss)
	# print(test_loss)
	train_losses.append(train_loss)
	test_losses.append(test_loss)
	print('iteration: %i, train_loss: %f, test_loss: %f'%(i, train_loss, test_loss))


