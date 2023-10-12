import pickle
import numpy as np

with open('../multi_manip/split_index.obj', 'rb') as fp:
    split_index = pickle.load(fp)
    fp.close()
with open('../multi_manip/cut_index.obj', 'rb') as fp:
    cut_index = pickle.load(fp)
    fp.close()


def get_idx_label(label_vector, attr_num):
    """
    convert the one-hot attribute labels to the label that indicate GT's position for each attribute
    Args:
        label_vector: one-hot labels
        attr_num: 1-D list of numbers of attribute values for each attribute
    """
    start_idx = 0
    labels = []
    for i in attr_num:
        sub_labels = label_vector[start_idx:start_idx + i]
        if sum(sub_labels) > 0:
            assert sum(sub_labels) == 1
            labels.append(np.argmax(sub_labels))
        else:
            labels.append(-1)  # missing label
        start_idx += i
    return np.array(labels)


def listify_manip(multi_manip):
    Nsplit = np.split(multi_manip, split_index[:-1])
    manip_array = np.zeros((12, 151), dtype=int)

    for j in range(manip_array.shape[0]):
        start, end = cut_index[j][0], cut_index[j][1]
        manip_array[j][start:end] = Nsplit[j]

    manip_list = manip_array[~np.all(manip_array == 0, axis=1)]

    manip_list = manip_list[np.random.permutation(len(manip_list))]

    return manip_list
