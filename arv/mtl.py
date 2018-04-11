from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.fcnnMTL import FcnnMTL
import shelve, pdb, sys
from tensorflow.examples.tutorials.mnist import input_data

MINI_BATCH_SIZE = 100
LEARNING_RATE = 0.0001
LEARNING_RATE_2 = 0.005
MAX_ITER = 4100
LAMBDA = 0.0001
TASKS_NUM = 10
SPLIT_NUM = 5
NUM_OF_LATENT = 10
LAMBDA2_list = np.logspace(-2,4,100)

exp_num_index = sys.argv.index('--exp_num')
exp_num = int(sys.argv[exp_num_index + 1])
gpu_num_index = sys.argv.index('--gpu_num')
gpu_num = int(sys.argv[gpu_num_index + 1])
LAMBDA2 = LAMBDA2_list[exp_num]

s_ITER = range(100,300) + range(1100,1300) + range(2100,2300) + range(3100, 3300)
L_ITER = range(300,1100)+ range(1300,2100) + range(2300,3100) + range(3300, 4100)
#s_ITER = range(MAX_ITER/9, MAX_ITER*2/9) + range(MAX_ITER*3/9, MAX_ITER*4/9) + range(MAX_ITER*5/9, MAX_ITER*6/9) + range(MAX_ITER*7/9, MAX_ITER*8/9)
#L_ITER = range(MAX_ITER/9) + range(MAX_ITER*2/9, MAX_ITER*3/9) + range(MAX_ITER*4/9, MAX_ITER*5/9) + range(MAX_ITER*6/9, MAX_ITER*7/9) + range(MAX_ITER*8/9, MAX_ITER)

with tf.device('/gpu:%i'%(gpu_num)):
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.23
	sess = tf.Session(config = config)
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	imgs = mnist.train.images
	# 55000 * 784
	labs = mnist.train.labels
	# 55000 * 10
	input_dim = imgs.shape[1]
	output_dim = labs.shape[1]

	output_dim = 2

	dim_of_units = [400,400,400]

	model = FcnnMTL(sess, input_dim, output_dim, len(dim_of_units), dim_of_units, NUM_OF_LATENT,bias = True)
	train_sample_y = tf.placeholder(tf.float32, [None, output_dim])
	train_sample_x, train_predict_y = model.place_holder()

	paras = [v for v in tf.trainable_variables() if 'fcnn' in v.name]
	paras_L = [v for v in paras if 'weights_L' in v.name]
	paras_bias = [v for v in paras if 'weights_bias' in v.name]
	paras_s = [v for v in paras if 'weights_s' in v.name]


	cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = train_sample_y, logits = train_predict_y))
	correct_prediction = tf.equal( tf.argmax(train_predict_y, axis = 1), tf.argmax(train_sample_y, axis = 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	l2_loss = LAMBDA * tf.add_n([tf.nn.l2_loss(v) for v in paras_L])
	optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE, name = 'adam')
	optimizer_2 = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE_2, name = 'adam2')	
	l1_loss = LAMBDA2 * tf.add_n([tf.reduce_sum(tf.abs(v)) for v in paras_s])

	if 'total_loss' not in globals().keys():
		total_loss = l1_loss + l2_loss + cross_entropy

	if 'train_op' not in globals().keys():
		train_op = optimizer.minimize(total_loss)
		train_L = optimizer.minimize(total_loss, var_list = paras_L + paras_bias)
		train_S = optimizer_2.minimize(total_loss, var_list = paras_s)

	loss_list = np.zeros([MAX_ITER, TASKS_NUM])
	ent_list = np.zeros([MAX_ITER, TASKS_NUM])
	l2_list = np.zeros([MAX_ITER, TASKS_NUM])
	l1_list = np.zeros([MAX_ITER, TASKS_NUM])
	train_acc_list = np.zeros([MAX_ITER, TASKS_NUM])

	weight_s_list = np.zeros([TASKS_NUM,len(dim_of_units) + 1, NUM_OF_LATENT ])
	#######

	Total_num_of_sample = imgs.shape[0]


	task_list = np.arange(10)
	split_list = np.array([0,0,1,1,2,2,3,3,4,4])
	for task, split in zip(task_list, split_list):

		training_x = imgs[ split*Total_num_of_sample/SPLIT_NUM : (split+1)*Total_num_of_sample/SPLIT_NUM ]
		training_y = labs[ split*Total_num_of_sample/SPLIT_NUM : (split+1)*Total_num_of_sample/SPLIT_NUM ]

		training_y = training_y[:,task]
		training_y = np.array([training_y, 1. - training_y]).transpose()

		testing_x = imgs
		testing_y = labs[:, task]
		testing_y = np.array([testing_y, 1. - testing_y]).transpose()

		sess.run(tf.global_variables_initializer())

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
			if i in range(100):
				random_s = np.zeros((len(dim_of_units)+1, NUM_OF_LATENT))
				for j in range(len(dim_of_units)+1):
					index = np.random.choice(NUM_OF_LATENT,size = 3, replace = False)
					random_s[j, index] = 1.
				#random_s = np.random.binomial(1, 0.3, (len(dim_of_units) + 1, NUM_OF_LATENT))
				sess.run([tf.assign(t, r_s) for t, r_s in zip(paras_s, random_s)])
				loss, cross_ent, l2loss, l1loss, _ = sess.run([total_loss, cross_entropy, l2_loss, l1_loss,train_L], \
					feed_dict = {train_sample_x: x_batch, train_sample_y: y_batch})
			elif i in s_ITER:
				loss, cross_ent, l2loss, l1loss, _ = sess.run([total_loss, cross_entropy, l2_loss, l1_loss , train_S], \
					feed_dict = {train_sample_x: x_batch, train_sample_y: y_batch})
			elif i in L_ITER:
				loss, cross_ent, l2loss, l1loss, _ = sess.run([total_loss, cross_entropy, l2_loss, l1_loss, train_L], \
					feed_dict = {train_sample_x: x_batch, train_sample_y: y_batch})
			#else:
			#	loss, cross_ent, l2loss, l1loss, _ = sess.run([total_loss, cross_entropy, l2_loss, l1_loss , train_S], \
                        #                feed_dict = {train_sample_x: x_batch, train_sample_y: y_batch})
			train_acc = sess.run(accuracy, feed_dict = {train_sample_x: testing_x, train_sample_y: testing_y})


			loss_list[i,task] = loss
			ent_list[i,task] = cross_ent
			l2_list[i,task] = l2loss
			l1_list[i,task] = l1loss
			train_acc_list[i,task] = train_acc


			if i%10 == 0:
				print('Task: %d, Step: %d, Training_Acc: %.2f, Loss: %.2f, Entropy: %.2f, L2_loss: %.2f, L1_loss: %.2f' % (task, i, train_acc, loss, cross_ent, l2loss, l1loss))
		weight_s = np.array([v.eval(session = sess) for v in paras_s])
		weight_s = np.squeeze(weight_s)
		weight_s_list[task] = weight_s

	filename = '/disk/scratch/chenyang/transfer/Data/mtl_dropout/exp_l1_%3.2f.out' % (LAMBDA2)
	my_shelf = shelve.open(filename,'n')
	saving_var_list = ['loss_list', 'ent_list', 'l2_list', 'l1_list', 'train_acc_list', 'weight_s_list', 'LAMBDA2_list']
	for key in saving_var_list:
	    try:
	        my_shelf[key] = globals()[key]
	    except TypeError:
	        print('ERROR shelving: {0}'.format(key))
	my_shelf.close()
