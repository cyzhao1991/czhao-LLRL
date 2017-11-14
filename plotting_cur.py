from __future__ import print_function
import numpy as np
import shelve, pdb, os, cv2, time
import matplotlib.pyplot as plt
from os import listdir

# LAMBDA2_list = np.logspace(-1,3,101)
# LAMBDA2_list = np.concatenate((LAMBDA2_list[:-1], np.logspace(3,6,101), np.logspace(6,9,101)), axis = 0)


exp_name = 'curri_l1_10'

filelist = listdir('./Data/'+exp_name+'/')
filelist = list(set([t[:-4] for t in filelist]))
# LAMBDA2_list = [float(t[7:-4]) for t in filelist]
# index = np.argsort(LAMBDA2_list)

# LAMBDA2_list = np.array([LAMBDA2_list[t] for t in index])

# LAMBDA2_list = LAMBDA2_list[0:50]
LAMBDA2_list = filelist
acc_result = np.zeros(len(LAMBDA2_list))

# l0_loss = np.zeros(len(LAMBDA2_list))
l0_loss = []
l2_loss_list = np.zeros(len(LAMBDA2_list))
l1_loss_list = np.zeros(len(LAMBDA2_list))
loss_list = np.zeros(len(LAMBDA2_list))
l0_losss = np.zeros(len(LAMBDA2_list))
full_acc_result = []
weight_s_list = []
norm_s_list = []

for i, filen in zip(range(len(LAMBDA2_list)), LAMBDA2_list):
	# if 'exp_0' in filen:
	# 	continue
	# if 'exp_50' in filen:
	# 	continue
	filename = './Data/'+exp_name+'/'+filen
	my_shelf = shelve.open(filename)
	# print(my_shelf.keys())
	acc = my_shelf['train_acc_list']

	weight_s = my_shelf['weight_s_list']
	l1_loss = my_shelf['l1_list']
	l2_loss = my_shelf['l2_list']
	loss = my_shelf['loss_list']
	task_list = my_shelf['task_list']
	# task_list = [task_list[3], task_list[2], task_list[1], task_list[0]] + list(task_list[4:])
	# task_list.extend()
	# print(task_list)
	weight_s_eval_list = my_shelf['weight_s_eval_list']
	my_shelf.close()
	# print(weight_s_eval_list.shape)
	acc = acc.transpose()
	# print(acc.shape)
	# print(task_list)
	# acc = np.mean(acc, axis =1)
	acc = acc[task_list]
	l1_loss = np.mean(l1_loss, axis = 1) 
	l2_loss = np.mean(l2_loss, axis = 1)
	loss = np.mean(loss, axis = 1)
	acc_result[i] = np.mean(acc[-20:])
	l2_loss_list[i] = np.mean(l2_loss[-20:])
	l1_loss_list[i] = np.mean(l1_loss[-20:])
	loss_list[i] = np.mean(loss[-20:])
	full_acc_result.append(acc)
 

	sparsity_list = []
	sparsity_listt = []
	norm_s = []

	for weight_s_per_task in weight_s:
		maxs = np.max(np.abs(weight_s_per_task), axis = 1)
		weight_s_per_task = np.transpose(np.abs(weight_s_per_task).T/maxs)
		sparsity = 1. * np.count_nonzero( weight_s_per_task <  0.1 ,axis = 1) / (weight_s_per_task.size/4)
		sparsity_list.append(sparsity)
		sparsityy = 1. * np.count_nonzero( weight_s_per_task > 0.5 ,axis = 1) / (weight_s_per_task.size/4)
		sparsity_listt.append(sparsityy)
		norm_s.append(weight_s_per_task)

	l0_loss.append(np.mean(sparsity_list, axis = 0))
	# l0_losss[i] = np.mean(sparsity_listt, axis = 0)
	weight_s_list.append(weight_s)
	norm_s_list.append(np.array(norm_s))

l0_loss = np.array(l0_loss)
l0_losss = np.array(l0_losss)
full_acc_result = np.array(full_acc_result)


# filename = 'Data/exp2.out'
# my_shelf_2 = shelve.open(filename)
# acc_2 = my_shelf_2['train_acc_list']
# acc_2 = acc_2.transpose()
# weight_s_2 = my_shelf_2['weight_s_list']
# my_shelf_2.close()

# filename = 'Data/exp3.out'
# my_shelf_3 = shelve.open(filename)
# acc_3 = my_shelf_3['train_acc_list']
# acc_3 = acc_3.transpose()
# weight_s_3 = my_shelf_3['weight_s_list']

# my_shelf_3.close()

# filename = 'Data/exp4.out'
# my_shelf_4 = shelve.open(filename)
# acc_4 = my_shelf_4['train_acc_list']
# acc_4 = acc_4.transpose()
# weight_s_4 = my_shelf_4['weight_s_list']

# my_shelf_4.close()


# print(weight_s_2[0])
np.set_printoptions(precision = 3, suppress = True)



# some_indexs = np.logical_and(acc_result > 0.95 , np.mean(l0_loss, axis=1) > 0.5)
# print(some_indexs)
# pdb.set_trace()

# plt.figure(1)
# _ = [plt.plot(LAMBDA2_list, np.squeeze(data), c) for data, c in zip(l0_loss.T, ['r','g','b','y'])]
# plt.plot(LAMBDA2_list, np.mean(l0_loss, axis = 1), 'black')
# # plt.plot(LAMBDA2_list, l0_losss,'g')
# plt.xscale('log')

# plt.figure(2)
# plt.plot(LAMBDA2_list, acc_result,'b')
# # plt.plot(LAMBDA2_list, l1_loss_list/(loss_list - l2_loss_list), 'r')
# plt.xscale('log')
# # for accc in acc:
# # 	plt.plot( accc )
# # plt.plot(np.mean(acc, axis = 0), color = 'r')
# # plt.plot(np.mean(acc_2, axis = 0), color = 'g')
# # plt.plot(np.mean(acc_3, axis = 0), color = 'b')
# # plt.plot(np.mean(acc_4, axis = 0), color = 'y')
# index = np.nonzero(some_indexs)
# index = (index[0])
# plt.figure(3)
# plt.imshow(np.abs(weight_s_list[index[0]][5]), cmap = 'hot', interpolation = 'nearest')
# print(weight_s_list[-1][9])
# plt.colorbar()

# plt.figure(4)
# plt.imshow(np.abs(norm_s_list[index[0]][5]), cmap = 'hot', interpolation = 'nearest')
# plt.colorbar()

plt.figure(5)
for i in range(10):
	acc_data = full_acc_result[:,i,:]
	# print(acc_data.shape)
	# acc_data = np.delete(acc_data, np.where(acc_data[:,-1]<0.9), axis = 0)
	# print(acc_data.shape)
	plt.plot(np.mean(acc_data,axis = 0), label = 'task%i'%(i))

# plt.plot(np.mean(full_acc_result[:,1,:],axis = 0), 'b')
plt.grid(which = 'both')
plt.legend()

# for i in range(5):
# 	plt.figure(3+i)
# 	plt.imshow(np.abs(weight_s_3[:,i,:]), cmap = 'hot', interpolation = 'nearest')
# 	plt.colorbar()

plt.show()

plt.figure(2)
plt.ion()

for i in range(10):
	print('task%i'%i)
	time.sleep(1)
	for j in range(0, 4100, 1000):
		print('iter:%i'%j)
		plt.imshow(np.transpose(np.abs(weight_s_eval_list[i,j]).T/np.max(weight_s_eval_list[i,j], axis = 1)), cmap = 'hot', interpolation = 'nearest')
		plt.pause(0.1)

		# time.sleep(.1)
