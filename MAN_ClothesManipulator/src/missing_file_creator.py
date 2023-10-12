import numpy as np
import pickle

# This file creates the missing files "cut_index.obj" and "split_index.obj" for ClothesManipulator Project
# Change to the correct saving path

attr = np.loadtxt("../splits/Shopping100k/attr_num.txt", dtype=int)
# attr = np.array([16, 17, 19, 14, 10, 15, 2, 11, 16, 7, 9, 15])

print(attr)

cut_index = []
start = 0
for i, num_attr in enumerate(attr):
    cut_index.append((start, start + num_attr))
    start += num_attr

print(cut_index)

split_index = [inx[1] for i, inx in enumerate(cut_index)]

print(split_index)


pickle.dump(cut_index, open('../multi_manip/cut_index.obj', 'wb'))
pickle.dump(split_index, open('../multi_manip/split_index.obj', 'wb'))
