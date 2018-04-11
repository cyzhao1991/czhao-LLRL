from __future__ import print_function
import numpy as np
import shelve, pdb, os
import matplotlib.pyplot as plt
from os import listdir

dir_list = os.listdir('./Data/trpo_stl/')
all_result = []
for dir_name in dir_list:
	try:
		my_shelve = shelve.open('./Data/trpo_stl/%s/shelve_result'%dir_name)
		all_result.append(my_shelve['saving_result'])
		# print(dir_name)
	except:
		pass

key_list = all_result[0].keys()
dict_result = dict([(key,[]) for key in key_list])
for key in key_list:
	[dict_result[key].append(r[key]) for r in all_result]

print(dict_result.keys())

iter_num = dict_result['iteration_number'][0]

for i, key in enumerate(dict_result.keys()):
	plt.figure(key)
# plt.plot(np.mean(dict_result['average_return'], axis = 0))
	plt.errorbar(iter_num, np.mean(dict_result[key], axis = 0), np.std(dict_result[key], axis = 0))

plt.show()