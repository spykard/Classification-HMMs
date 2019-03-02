import numpy as np


trans_matrix_pos = np.ones((5,5))
trans_matrix_neg = np.ones((6,6))
mapping_pos = ['1', '2', '3', '4', '5']
mapping_pos = ['3', '4', '5', '6', '7', '8']

ind_of_pos = 0
ind_of_neg = 0

remap_size = 102

trans_matrix_pos_new = np.zeros((102,102))

for i in range(remap_size):
    
    print(i)
