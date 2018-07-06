from __future__ import print_function
import numpy as np
import shelve, pdb
import matplotlib.pyplot as plt
from os import listdir

filename = "ppo_Data/dm_control/mtl/walker/multiwind/exp0/shelve_result"
tmp_s = shelve.open(filename)
result = tmp_s['saving_result']
# for _,v in result.items():
# 	v = np.array(v)

all_return = np.array(result['average_return'])
plt.figure(1)
# [plt.plot(all_return[:,i]) for i in range( all_return.shape[-1] )]
for y in np.array(all_return).T:
	y = np.array([np.mean(y[np.max([0, i-10]):i+1]) for i in range(500)])
	plt.plot(y[:], linewidth = 1.5)
plt.legend('123')
plt.grid()
plt.show()




# LAMBDA2_list = np.logspace(-1,3,101)
# # LAMBDA2_list = np.concatenate((LAMBDA2_list[:-1], np.logspace(3,6,101), np.logspace(6,9,101)), axis = 0)
# # rng = [0,1,2,3,4,7,8,9]
# rng = range(10)
# # filename_list = ['../Data/context_trpo_path10/mod_5_exp%i/shelve_result'%i for i in rng]
# filename_list = ['Data/dm_control/context_mtl/walker_walk(arv)/exp%i/shelve_result'%i for i in range(5)]
# # shelf_list = [shelve.open(filename) for filename in filename_list]
# # pdb.set_trace()
# all_result = []
# for filename in filename_list:
# 	try:
# 		shelf = shelve.open(filename)
# 		all_result.append(shelf['saving_result'])
# 		print('good     '+filename)
# 		# all_result = [shelf['saving_result'] for shelf in shelf_list]
# 	except:
# 		print('bad      '+filename)
# key_list = all_result[0].keys()
# all_result = dict([ (key, np.array([result[key] for result in all_result]) ) for key in key_list])
# plt.figure(1)
# x_data = np.arange(500)
# x_err_data = np.arange(0,500,10)

# [print(key, value.shape) for key, value in all_result.iteritems()]
# all_s_vector = all_result['s_vector']

# del all_result['s_vector']
# del all_result['iteration_number']
# # pdb.set_trace()

# for i, (key, data) in enumerate(all_result.iteritems()):
# 	# print(i)
# 	plt.subplot(2,4,i + 1)
# 	plt.title(key)
# 	mean_data = np.mean(data, axis = 0)[x_data]
# 	std_data = np.std(data, axis = 0)[x_data]

# 	if len(mean_data.shape) > 1:
# 		mean_data = np.mean(mean_data, axis = -1)
# 		std_data = np.std(std_data, axis = -1)
# 	if key is 's_vector' or len(mean_data.shape) > 1:
# 		continue
# 	print(mean_data.shape, std_data.shape)
# 	plt.plot( mean_data )
# 	plt.errorbar( x_err_data, mean_data[x_err_data], std_data[x_err_data] )
# 	plt.xlim(-1,x_data[-1])
# 	plt.grid()

# # plt.subplot(2,4,8)
# # all_return = np.concatenate( [y for y in all_result['average_return']], axis = 1 )
# all_return = all_result['average_return']
# print('all_return', all_return.shape)
# plt.figure(2)
# plt.title('single task return')
# [plt.plot(y, label = 'task%i'%i) for i,y in enumerate(all_return)]
# plt.legend()
# plt.figure(3)
# all_s_vector = np.swapaxes(all_s_vector, 2, 3)
# for i in range(4):
# 	plt.subplot(2,2,i+1)
# 	print(all_s_vector[i][-1].shape)
# 	vis_data = np.reshape(all_s_vector[i][-1], [4, -1])
# 	# vis_data = np.array([a/np.std(a) for a in vis_data])
# #	data_tmp = np.reshape(all_s_vector[i][-1], [4, -1])
# 	plt.imshow( np.abs(vis_data), interpolation = 'nearest', cmap = 'hot')
# 	plt.colorbar()
# plt.figure(4)
# for i in range(4):
# 	plt.subplot(2,2,i+1)
# 	vis_data = np.reshape(all_s_vector[i][-1], [4, -1])
# 	# vis_data = np.array([a/np.std(a) for a in vis_data])
# 	print(np.count_nonzero(np.abs(vis_data) > 0.01) )
# 	plt.imshow( np.abs(vis_data) > 0.01, interpolation = 'nearest', cmap = 'hot')
# 	# plt.colorbar()
# # print('all_s_vector', all_s_vector[0].shape)
# tmp_l0norm = np.array([np.count_nonzero( np.abs(v) > 0.001) for v in all_s_vector[0]])
# tmp_l1norm = np.array([np.sum(v) for v in all_s_vector[0]])

# # tmp_l0norm = np.count_nonzero(all_s_vector[0],axis = 0)
# # print('tmp_l0norm',tmp_l0norm.shape)
# # plt.figure(5)	
# # plt.plot(tmp_l0norm)
# plt.show()

# # for i in range(10):
# # 	plt.figure(i)
# # 	for j in range(4):
# # 		plt.subplot(2,2,j+1)
# # 		data_tmp = all_s_vector[i][-1][j]
# # 		plt.imshow( np.maximum(data_tmp, 0, data_tmp), interpolation = 'nearest', cmap = 'hot')
# # 		plt.colorbar()
# # 	plt.savefig('../task%i.png'%i)
# # plt.show()