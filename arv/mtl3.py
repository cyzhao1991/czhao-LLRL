from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.fcnnMTL import FcnnMTL
import shelve, pdb
from tensorflow.examples.tutorials.mnist import input_data

MINI_BATCH_SIZE = 100
LEARNING_RATE = 0.0001
MAX_ITER = 4000
LAMBDA = 0.0001
LAMBDA2 = 0.1
TASKS_NUM = 10
SPLIT_NUM = 5
NUM_OF_LATENT = 10

sess = tf.Session()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
imgs = mnist.train.images
# 55000 * 784
labs = mnist.train.labels
# 55000 * 10
input_dim = imgs.shape[1]
output_dim = labs.shape[1]

output_dim = 2

dim_of_units = [400,400,400,400]

model = FcnnMTL(sess, input_dim, output_dim, len(dim_of_units), dim_of_units, NUM_OF_LATENT,bias = True)
# train_sample_x = tf.placeholder(tf.float32, [MINI_BATCH_SIZE, input_dim])
train_sample_y = tf.placeholder(tf.float32, [None, output_dim])
train_sample_x, train_predict_y = model.place_holder()

# weights_s_list = {}
# l1_loss_list = {}
# for t in range(TASKS_NUM):
# 	train_sample_x, train_predict_y, weights_s_list[t] = model.place_holder()
# 	l1_loss_list[t] = tf.reduce_sum( tf.abs(weights_s_list[t]) )

cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = train_sample_y, logits = train_predict_y))
correct_prediction = tf.equal( tf.argmax(train_predict_y, axis = 1), tf.argmax(train_sample_y, axis = 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

paras = [v for v in tf.trainable_variables() if 'fcnn' in v.name]
l2_loss = LAMBDA * tf.add_n([tf.nn.l2_loss(v) for v in paras if 'weights' in v.name])
l1_loss = LAMBDA2 * tf.add_n([tf.reduce_sum(tf.abs(v)) for v in paras if 'weight_s' in v.name])

total_loss = l1_loss + l2_loss + cross_entropy



optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE, name = 'adam')
train_op = optimizer.minimize(total_loss)
loss_list = np.zeros([MAX_ITER, TASKS_NUM])
ent_list = np.zeros([MAX_ITER, TASKS_NUM])
l2_list = np.zeros([MAX_ITER, TASKS_NUM])
l1_list = np.zeros([MAX_ITER, TASKS_NUM])
train_acc_list = np.zeros([MAX_ITER, TASKS_NUM])

weight_s_list = np.zeros([TASKS_NUM,len(dim_of_units) + 1, NUM_OF_LATENT ])
#######

Total_num_of_sample = imgs.shape[0]
# perm1 = np.arange(Total_num_of_sample)
# np.random.shuffle(perm1)
# imgs = imgs[perm1]
# labs = labs[perm1]

task_list = np.arange(10)
split_list = np.array([0,0,1,1,2,2,3,3,4,4])
for task, split in zip(task_list, split_list):

	training_x = imgs[ split*Total_num_of_sample/SPLIT_NUM : (split+1)*Total_num_of_sample/SPLIT_NUM ]
	training_y = labs[ split*Total_num_of_sample/SPLIT_NUM : (split+1)*Total_num_of_sample/SPLIT_NUM ]

	training_y = training_y[:,task]
	training_y = np.array([training_y, 1. - training_y]).transpose()


	# sess.run(tf.global_variables_initializer())
	sess.run(tf.variables_initializer([v for v in paras if 'weight_s' in v.name]))

	start = 0
	num_of_sample = training_x.shape[0]

	for i in range(MAX_ITER):

		if start + MINI_BATCH_SIZE > num_of_sample:
			perm = np.arange(num_of_sample)
			np.random.shuffle(perm)
			training_x = training_x[perm]
			training_y = training_y[perm]
			start = 0
		
		x_batch = training_x[start:start+MINI_BATCH_SIZE]
		y_batch = training_y[start:start+MINI_BATCH_SIZE]
		start = start + MINI_BATCH_SIZE
		y_pre = model.predict(x_batch)
		# _ = [print(v.name) for v in tf.trainable_variables()]
		loss, cross_ent, l2loss, l1loss, train_acc, _ = sess.run([total_loss, cross_entropy, l2_loss, l1_loss , accuracy, train_op], \
			feed_dict = {train_sample_x: x_batch, train_sample_y: y_batch, train_predict_y: y_pre})
		

		loss_list[i,task] = loss
		ent_list[i,task] = cross_ent
		l2_list[i,task] = l2loss
		l1_list[i,task] = l1loss
		train_acc_list[i,task] = train_acc

		

		if i%10 == 0:
			print('Task: %d, Step: %d, Training_Acc: %g, Loss: %g, Entropy: %g, L2_loss: %g, L1_loss: %g' % (task, i, train_acc, loss, cross_ent, l2loss, l1loss))
	weight_s = np.squeeze([v.eval(session = sess) for v in paras if 'weight_s' in v.name])
	weight_s_list[task] = weight_s

filename = 'Data/exp5.out'
my_shelf = shelve.open(filename,'n') 
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()