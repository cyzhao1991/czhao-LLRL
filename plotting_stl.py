from __future__ import print_function
import numpy as np
import shelve, pdb
import matplotlib.pyplot as plt
from os import listdir

# LAMBDA2_list = np.logspace(-1,3,101)
# LAMBDA2_list = np.concatenate((LAMBDA2_list[:-1], np.logspace(3,6,101), np.logspace(6,9,101)), axis = 0)
# s_list = [-4., -2., -1., 0., 1., 2., 4.]
s_list = [-2., 0. ,2.]
g_list = [-5., -2.5, 0., 2.5, 5.]
# w_list = [-3., -1.5, 0., 1.5, 3.]
w_list = [-4, -2, -1, 0, 1, 2, 4]
mtl_w_list = [-3., -1.5, 0., 1.5, 3.]
filename_list = ['Data/dm_control/stl(con)/walker_s%1.1f/w0.0g0.0/exp%i/shelve_result'%(j,i) for i in [9] for j in s_list]
# filename_list = ['Data/dm_control/stl(con)/walker_s%1.1f/w0.0g0.0/exp1/shelve_result'%(i) for i in mtl_w_list]
# filename_list = ['Data/dm_control/finetune/walker_s%1.1f/w0.0g0.0/exp0/shelve_result'%i for i in mtl_w_list]
# filename_list = ['Data/dm_control/finetune/mtl_walker_s%1.1f/w0.0g0.0/exp0/shelve_result'%i for i in mtl_w_list]
# filename_list = ['Data/dm_control/finetune_ver1/mtl_walker_s%1.1f/w0.0g0.0/exp0/shelve_result'%i for i in mtl_w_list]
# filename_list = ['Data/dm_control/finetune_ver1/mtl_walker_s%1.1f/w0.0g0.0/exp0/shelve_result'%(i) for i in mtl_w_list]
shelf_list = [shelve.open(filename) for filename in filename_list]
# pdb.set_trace()
all_result = []
for filename, shelf in zip(filename_list, shelf_list):
	try:	
		all_result.append(shelf['saving_result'])
		print('good     '+filename)
		# print(shelf['goal'])
		# all_result = [shelf['saving_result'] for shelf in shelf_list]
	except:
		print('bad      '+filename)

key_list = all_result[0].keys()
all_result = dict([ (key, np.array([np.array(result[key][:500]) for result in all_result]) ) for key in key_list])
plt.figure(1)
# x_data = np.arange(500)
# x_err_data = np.arange(0,500,10)


try:
	del all_result['iteration_number']
	del all_result['s_vector']

except:
	pass

for i, (key, data) in enumerate(all_result.items()):
	# print(i)
	plt.subplot(2,4,i + 1)
	plt.title(key)
	mean_data = np.mean(data, axis = 0)#[x_data]
	print(key, mean_data.shape) 
	std_data = np.std(data, axis = 0)#[x_data]
	plt.plot( mean_data, linewidth = .3)
	# plt.errorbar( x_err_data, mean_data[x_err_data], std_data[x_err_data] )
	plt.xlim(-1,mean_data.shape[0])

	plt.grid()

# plt.subplot(2,4,8)
plt.figure(2)
plt.title('single task return')
[plt.plot(y[:], linewidth = .5) for y in all_result['average_return']]
plt.grid()
plt.legend('1234567')
# plt.show()

# plt.figure(3)
# [plt.plot(y[:100], linewidth = .5) for y in all_result['l1_norm']]
# plt.grid()
# plt.legend('1234567')

# plt.figure(4)
# # plt.title('single task return')
# [plt.plot(y[:100], linewidth = .5) for y in all_result['column_norm']]
# plt.grid()
# plt.legend('1234567')
plt.show()
# exp_name = 'trpo_stl'

# filelist = listdir('../Data/'+exp_name+'/')
# filelist = list(set([t[:-4] for t in filelist]))
# LAMBDA2_list = [float(t[-1:]) for t in filelist]
# index = np.argsort(LAMBDA2_list)

# LAMBDA2_list = np.array([LAMBDA2_list[t] for t in index])

# # LAMBDA2_list = LAMBDA2_list[0:50]

# acc_result = np.zeros(len(LAMBDA2_list))

# # l0_loss = np.zeros(len(LAMBDA2_list))
# # l0_loss = []
# l2_loss_list = np.zeros(len(LAMBDA2_list))
# # l1_loss_list = np.zeros(len(LAMBDA2_list))
# loss_list = np.zeros(len(LAMBDA2_list))
# # l0_losss = np.zeros(len(LAMBDA2_list))
# full_acc_result = []
# # weight_s_list = []
# # norm_s_list = []
# for i, lamb2 in zip(range(len(LAMBDA2_list)), LAMBDA2_list):
# 	# filename = './Data/'+exp_name+'/exp_l1_%3.2f.out' % (lamb2)
# 	filename = './Data/'+exp_name+'/exp%i.out'%(i)
# 	my_shelf = shelve.open(filename)
# 	acc = my_shelf['train_acc_list']
# 	# weight_s = my_shelf['weight_s_list']
# 	# l1_loss = my_shelf['l1_list']
# 	l2_loss = my_shelf['l2_list']
# 	loss = my_shelf['loss_list']
# 	my_shelf.close()

# 	# acc = acc.transpose()
# 	acc = np.mean(acc, axis =1)
# 	# l1_loss = np.mean(l1_loss, axis = 1)
# 	l2_loss = np.mean(l2_loss, axis = 1)
# 	loss = np.mean(loss, axis = 1)
# 	acc_result[i] = np.mean(acc[-20:])
# 	l2_loss_list[i] = np.mean(l2_loss[-20:])
# 	# l1_loss_list[i] = np.mean(l1_loss[-20:])
# 	loss_list[i] = np.mean(loss[-20:])
# 	full_acc_result.append(acc)
 

# 	# sparsity_list = []
# 	# sparsity_listt = []
# 	# norm_s = []

# 	# for weight_s_per_task in weight_s:
# 	# 	maxs = np.max(np.abs(weight_s_per_task), axis = 1)
# 	# 	weight_s_per_task = np.transpose(np.abs(weight_s_per_task).T/maxs)
# 	# 	sparsity = 1. * np.count_nonzero( weight_s_per_task <  0.1 ,axis = 1) / (weight_s_per_task.size/4)
# 	# 	sparsity_list.append(sparsity)
# 	# 	sparsityy = 1. * np.count_nonzero( weight_s_per_task > 0.5 ,axis = 1) / (weight_s_per_task.size/4)
# 	# 	sparsity_listt.append(sparsityy)
# 	# 	norm_s.append(weight_s_per_task)

# 	# l0_loss.append(np.mean(sparsity_list, axis = 0))
# 	# # l0_losss[i] = np.mean(sparsity_listt, axis = 0)
# 	# weight_s_list.append(weight_s)
# 	# norm_s_list.append(np.array(norm_s))

# # l0_loss = np.array(l0_loss)
# # l0_losss = np.array(l0_losss)

# # filename = 'Data/exp2.out'
# # my_shelf_2 = shelve.open(filename)
# # acc_2 = my_shelf_2['train_acc_list']
# # acc_2 = acc_2.transpose()
# # weight_s_2 = my_shelf_2['weight_s_list']
# # my_shelf_2.close()

# # filename = 'Data/exp3.out'
# # my_shelf_3 = shelve.open(filename)
# # acc_3 = my_shelf_3['train_acc_list']
# # acc_3 = acc_3.transpose()
# # weight_s_3 = my_shelf_3['weight_s_list']

# # my_shelf_3.close()

# # filename = 'Data/exp4.out'
# # my_shelf_4 = shelve.open(filename)
# # acc_4 = my_shelf_4['train_acc_list']
# # acc_4 = acc_4.transpose()
# # weight_s_4 = my_shelf_4['weight_s_list']

# # my_shelf_4.close()


# # print(weight_s_2[0])
# full_acc_result = np.array(full_acc_result)
# print(full_acc_result.shape)
# np.set_printoptions(precision = 3, suppress = True)
# plt.figure(1)
# # print(l0_loss.shape)
# # print(len(LAMBDA2_list))
# # _ = [plt.plot(LAMBDA2_list, np.squeeze(data), c) for data, c in zip(l0_loss.T, ['r','g','b','y'])]
# # plt.plot(LAMBDA2_list, np.mean(l0_loss, axis = 1), 'black')
# # plt.plot(LAMBDA2_list, l0_losss,'g')
# plt.plot(np.mean(full_acc_result, axis = 0))
# plt.grid()# plt.xscale('log')

# # plt.figure(2)
# # plt.plot(LAMBDA2_list, acc_result,'b')
# # # plt.plot(LAMBDA2_list, l1_loss_list/(loss_list - l2_loss_list), 'r')
# # plt.xscale('log')
# # for accc in acc:
# # 	plt.plot( accc )
# # plt.plot(np.mean(acc, axis = 0), color = 'r')
# # plt.plot(np.mean(acc_2, axis = 0), color = 'g')
# # plt.plot(np.mean(acc_3, axis = 0), color = 'b')
# # plt.plot(np.mean(acc_4, axis = 0), color = 'y')

# # plt.figure(3)
# # plt.imshow(np.abs(weight_s_list[-3][5]), cmap = 'hot', interpolation = 'nearest')
# # print(weight_s_list[-1][9])
# # plt.colorbar()

# # plt.figure(4)
# # plt.imshow(np.abs(norm_s_list[-3][5]), cmap = 'hot', interpolation = 'nearest')
# # plt.colorbar()

# # plt.figure(5)
# # plt.plot(full_acc_result[0])



# # for i in range(5):
# # 	plt.figure(3+i)
# # 	plt.imshow(np.abs(weight_s_3[:,i,:]), cmap = 'hot', interpolation = 'nearest')
# # 	plt.colorbar()

# plt.show()


