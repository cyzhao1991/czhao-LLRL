from __future__ import print_function
import numpy as np
import tensorflow as tf
from model.net import * 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import shelve

np.random.seed(2626)
xs = np.random.normal(0., 1., [15000, 20])
thetas = np.random.normal(0., 1., [3, 21]) + 1.
ys = [np.dot( np.append(xs, np.ones([15000,1]), axis = 1), th ) + np.random.normal(0., 0.2, [15000]) for th in thetas]
ys = [y[:, np.newaxis] for y in ys]
def data_split(xs, ys, training_size, testing_size):
	index = np.random.permutation( len(ys) )
	train_ind = index[:training_size]
	test_ind = index[training_size:training_size+testing_size]
	training_set = {'xs': xs[train_ind], 'ys': ys[train_ind]}
	testing_set = {'xs': xs[test_ind], 'ys': ys[test_ind]}
	return (training_set, testing_set)
'''
	stl setting
'''
# train_losses_across_task = []
# test_losses_across_task = []
# learned_models = []

# for _ in range(100):
# 	temp_int = np.random.randint(3)
# 	train_set, test_set = data_split( xs, ys[temp_int], 150, 350 )

# 	sess = tf.Session()
# 	net = Fcnn(sess, 20, 1, [100, 50])

# 	input_ph = net.input
# 	predict = net.output
# 	label = tf.placeholder(tf.float32, [None,1])
# 	var_list = [v for v in tf.trainable_variables()]
# 	l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in var_list])
# 	loss = tf.losses.mean_squared_error(predict, label)
# 	total_loss = loss + .01* l2_loss

# 	optimizer = tf.train.AdamOptimizer()
# 	train_op = optimizer.minimize(total_loss)

# 	sess.run(tf.global_variables_initializer())

# 	train_losses = []
# 	test_losses = []
# 	for i in range(10000):

# 		_, train_loss= sess.run([train_op, total_loss], feed_dict = {input_ph: train_set['xs'], label:train_set['ys']})
# 		test_loss = sess.run(loss, feed_dict = {input_ph: test_set['xs'], label: test_set['ys']}) / len(test_set['xs'])
# 		# print(train_loss)
# 		# print(test_loss)
# 		train_losses.append(train_loss)
# 		test_losses.append(test_loss)
# 		print('iteration: %i, train_loss: %f, test_loss: %f'%(i, train_loss, test_loss))

# 	learned_model = sess.run(var_list)
	
# 	sess.close()

# 	train_losses_across_task.append(train_losses)
# 	test_losses_across_task.append(test_losses)
# 	learned_models.append(learned_model)

# filename = '/afs/inf.ed.ac.uk/user/s16/s1686875/Documents/transfer/czhao-LLRL/Data/'
# my_shelf = shelve.open(filename + 'stl_synthetic_regression.out','n')
# saving_var_list = ['train_losses_across_task', 'test_losses_across_task', 'learned_models']
# for key in saving_var_list:
#     try:
#         my_shelf[key] = globals()[key]
#     except TypeError:
#         print('ERROR shelving: {0}'.format(key))
# my_shelf.close()

# x_axis = np.arange(100)
# train_mu = np.mean(train_losses_across_task, axis = 0)
# train_std = np.std(train_losses_across_task, axis = 0)
# test_mu  = np.mean(test_losses_across_task,  axis = 0)
# test_std  = np.std(test_losses_across_task,  axis = 0)
# plt.figure('train_losses')
# plt.plot(x_axis, train_mu, 'r')
# plt.fill_between(x_axis, train_mu - train_std, train_mu + train_std, facecolor = '#ff6666', rasterized = True)
# plt.savefig(filename + 'stl_train_losses.eps', rasterized = True)
# plt.figure('test_losses')
# plt.plot(x_axis, test_mu, 'r')
# plt.fill_between(x_axis, test_mu - test_std, test_mu + test_std, facecolor = '#ff6666', rasterized = True)
# plt.savefig(filename + 'stl_test_losses.eps', rasterized = True)



'''
	mtl setting
'''
train_losses_across_task = []
test_losses_across_task = []
sparse_losses_across_task = []
num_of_valid_modules_across_task = []
num_of_invalid_modules_across_task = []
s_weights_across_task = []

for _ in range(30):
	data_sets = []
	yss = np.concatenate( [ys[0][:5000], ys[1][5000:10000], ys[2][10000:] ], axis = 0)
	start = 0
	end = 5000
	for _ in range(3):
		
		for i in range(10):
			data_sets.append(data_split(xs[start:end], yss[start:end], 150, 350))
			# data_sets.append(data_split(xs[5000:10000], ys[5000:10000], 150, 350))
			# data_sets.append(data_split(xs[10000:], ys[10000:], 150, 350))
		start += 5000
		end += 5000

	task_num = 30
	task_des = np.eye(30)
	all_training_set_x = []
	all_training_set_y = []
	all_training_set_t = []

	all_testing_set_x = []
	all_testing_set_y = []
	all_testing_set_t = []

	for data_set, task_d in zip(data_sets, task_des):
		all_training_set_x += data_set[0]['xs'].tolist()
		all_training_set_y += data_set[0]['ys'].tolist()
		all_training_set_t += np.tile(task_d, (150,1)).tolist()

		all_testing_set_x += data_set[1]['xs'].tolist()
		all_testing_set_y += data_set[1]['ys'].tolist()
		all_testing_set_t +=np.tile(task_d, (350,1)).tolist()

	all_training_set_x = np.array(all_training_set_x)
	all_training_set_y = np.array(all_training_set_y)
	all_training_set_t = np.array(all_training_set_t)
	all_testing_set_x = np.array(all_testing_set_x)
	all_testing_set_y = np.array(all_testing_set_y)
	all_testing_set_t = np.array(all_testing_set_t)

	# print(all_training_set_x.shape, all_training_set_y.shape, all_training_set_t.shape)

	sess = tf.Session()
	weight_net = Fcnn(sess, 30, 30, [], name = 'weight_net', if_bias = [False], activation = ['None'])
	w_out = weight_net.output
	s_weights = [tf.slice(w_out, [0,0], [-1, 10]), tf.slice(w_out, [0, 10], [-1,10]), tf.slice(w_out, [0, 20], [-1,10]) ]
	module_net = Modular_Fcnn(sess, 20, 1, [100, 50], [10, 10, 10], name = 'module_net', s_weights = s_weights)

	# weight_net = Fcnn(sess, 30, 10, [], name = 'weight_net', if_bias = [False], activation = ['None'])
	# w_out = weight_net.output
	# s_weights = [tf.slice(w_out, [0,0], [-1, 10]), tf.slice(w_out, [0, 10], [-1,10]), tf.slice(w_out, [0, 20], [-1,10]) ]
	# module_net = Modular_Fcnn(sess, 20, 1, [], [10], name = 'module_net', s_weights = s_weights)

	input_ph = module_net.input
	predict = module_net.output
	context_ph = weight_net.input
	label = tf.placeholder(tf.float32, [None,1])

	var_list = [v for v in tf.trainable_variables()]
	l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in var_list if 'module_net' in v.name])
	l1_loss = tf.reduce_sum( [tf.reduce_sum(tf.abs(v)) for v in var_list if 'weight_net' in v.name] )
	loss = tf.losses.mean_squared_error(predict, label)
	total_loss = loss + .01* l2_loss + .01 * l1_loss

	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(total_loss)

	sess.run(tf.global_variables_initializer())
	train_losses = []
	test_losses = []
	sparse_losses = []
	nums_of_valid_modules = []
	nums_of_invalid_modules = []
	for i in range(10000):
		train_feed_dict = {input_ph: all_training_set_x, label:all_training_set_y, context_ph:all_training_set_t}
		test_feed_dict = {input_ph: all_testing_set_x, label: all_testing_set_y, context_ph: all_testing_set_t}
		_, train_loss, sparse_loss = sess.run([train_op, total_loss, l1_loss], feed_dict = train_feed_dict)
		test_loss = sess.run(loss, feed_dict = test_feed_dict) / len(all_testing_set_x)
		weights = sess.run([v for v in var_list if 'weight_net' in v.name])
		invalid_modules = np.count_nonzero( np.abs(weights[0]) < .01)
		valid_modules = np.count_nonzero( np.abs(weights[0]) > .1 )

		train_losses.append(train_loss)
		test_losses.append(test_loss)
		sparse_losses.append(sparse_loss)
		nums_of_valid_modules.append(valid_modules)
		nums_of_invalid_modules.append(invalid_modules)
		print('iteration: %i, l1_loss: %f, valid_modules: %i, invalid_modules: %i, train_loss: %f, test_loss: %f'%(i, sparse_loss, valid_modules,invalid_modules,train_loss, test_loss))

	sess.close()

	train_losses_across_task.append(train_losses)
	test_losses_across_task.append(test_losses)
	sparse_losses_across_task.append(sparse_losses)
	num_of_valid_modules_across_task.append(nums_of_valid_modules)
	num_of_invalid_modules_across_task.append(nums_of_invalid_modules)
	s_weights_across_task.append(weights)


filename = '/afs/inf.ed.ac.uk/user/s16/s1686875/Documents/transfer/czhao-LLRL/Data/'
my_shelf = shelve.open(filename + 'mtl_synthetic_regression.out','n')
saving_var_list = ['train_losses_across_task', 'test_losses_across_task', 'sparse_losses_across_task', 'num_of_valid_modules_across_task', \
	'num_of_invalid_modules_across_task', 's_weights_across_task']
for key in saving_var_list:
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()

x_axis = np.arange(10000)
train_mu = np.mean(train_losses_across_task, axis = 0)
train_std = np.std(train_losses_across_task, axis = 0)
test_mu  = np.mean(test_losses_across_task,  axis = 0)
test_std  = np.std(test_losses_across_task,  axis = 0)
plt.figure('train_losses')
plt.plot(x_axis, train_mu, 'r')
plt.fill_between(x_axis, train_mu - train_std, train_mu + train_std, facecolor = '#ff6666', rasterized = True)
plt.savefig(filename + 'mtl_train_losses.eps', rasterized = True)
plt.figure('test_losses')
plt.plot(x_axis, test_mu, 'r')
plt.fill_between(x_axis, test_mu - test_std, test_mu + test_std, facecolor = '#ff6666', rasterized = True)
plt.savefig(filename + 'mtl_test_losses.eps', rasterized = True)

plt.figure('nums_of_valid_modules')
plt.plot(x_axis, np.mean(num_of_valid_modules_across_task, axis = 0), 'r')
plt.savefig(filename + 'mtl_valid_nums.eps', rasterized = True)
plt.figure('nums_of_invalid_modules')
plt.plot(x_axis, np.mean(num_of_invalid_modules_across_task, axis = 0), 'r')
plt.savefig(filename + 'mtl_invalid_nums.eps', rasterized = True)


