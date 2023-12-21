"""
 The "get_next_pair_sample" method return a couple of samples at distance less or equal to "max_distance_between_pair"
"""

import numpy as np
import torchvision.transforms as transforms
import MAN_ClothesManipulator.src.constants as C
from MAN_ClothesManipulator.src.dataloader import Data
from MAN_ClothesManipulator.src.utils import cut_index
from utils import listify_manip
from numpy.linalg import norm

from utils import ATTRIBUTES, ATTRIBUTES_VALUES


def normalize(dis_feat, normalization):
    if normalization:
        dis_feat_normalized = (dis_feat - np.min(dis_feat)) / (np.max(dis_feat) - np.min(dis_feat))
        return dis_feat_normalized
    return dis_feat


class DataSupplier:

    def __init__(self, file_root, img_root_path, dis_feat_root, mode='train', normalized=False):
        self.mode = mode  # 'train' or 'test'

        if self.mode == 'train':
            self.data = Data(file_root, img_root_path,
                             transforms.Compose([
                                 transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                             ]), self.mode)
            if normalized:
                dis_feat_root = format(dis_feat_root + "/feat_train_Norm.npy")
            else:
                dis_feat_root = format(dis_feat_root + "/feat_train_senzaNorm.npy")
        elif self.mode == 'test':
            self.data = Data(file_root, img_root_path,
                             transforms.Compose([
                                 transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                             ]), self.mode)
            if normalized:
                dis_feat_root = format(dis_feat_root + "/feat_test_Norm.npy")
            else:
                dis_feat_root = format(dis_feat_root + "/feat_test_senzaNorm.npy")

        self.dis_feat_file = np.load(dis_feat_root)

    def get_next_pair_sample(self, max_distance_between_pair):
        labels = self.data.label_data
        q, t = self.find_couple(labels, max_distance_between_pair)  # query = start image, target = wanted image

        if q != -1 and t != -1:
            return True, q, t

        return False, q, t

    def find_couple(self, labels, max_distance_between_pair):
        n_labels = labels.shape[0]
        cut_index_np = np.array(cut_index)

        found_q = -1
        found_t = -1

        q_indexes = np.arange(n_labels)
        np.random.shuffle(q_indexes)

        for q_id in q_indexes:
            t_indexes = np.arange(n_labels)  # [0, 1, ..., n_labels]
            np.random.shuffle(t_indexes)
            for t_id in t_indexes:
                if not np.array_equal(labels[q_id], labels[t_id]):
                    if not self.too_much_distance(labels[q_id], labels[t_id], cut_index_np, max_distance_between_pair):
                        found_q = q_id
                        found_t = t_id
                        return found_q, found_t  # Contain id of two images with <=N distance

        return found_q, found_t  # return -1, -1

    def too_much_distance(self, q_lbl, t_lbl, cut_index_np, max_distance_between_pair):
        multi_manip = np.subtract(q_lbl, t_lbl)
        distance = 0
        for ci in cut_index_np:
            if np.any(multi_manip[ci[0]:ci[1]]):  # return false if [0, 0, ..., 0], true if [-1, 0, 1, 0 ..., 0]
                distance += 1
        if distance > max_distance_between_pair:
            return True
        else:
            return False

    def get_disentangled_features(self, q_id, t_id):
        q_dis_feat = self.dis_feat_file[q_id]
        t_dis_feat = self.dis_feat_file[t_id]
        return q_dis_feat, t_dis_feat

    def get_one_hot_labels(self, q_id, t_id):
        labels = self.data.label_data
        return labels[q_id], labels[t_id]

    def get_on_hot_label(self, id):
        labels = self.data.label_data
        return labels[id]

    def get_images(self, q_id, t_id):
        return self.get_image(q_id), self.get_image(t_id)

    def get_image(self, id):
        return self.data.__getitem__(id)

    def get_manipulation_vectors(self, q_id, t_id, max_distance_between_pair):
        q_label, t_label = self.get_one_hot_labels(q_id, t_id)
        multi_manip = np.subtract(t_label, q_label)
        multi_manip = listify_manip(multi_manip)
        if len(multi_manip) <= max_distance_between_pair:
            return True, multi_manip
        return False, multi_manip

    def cosine_similarity(self, dis_feat, target_feat):
        return np.dot(target_feat, dis_feat) / (norm(target_feat) * norm(dis_feat))

    def find_x_ids_images_more_similiar(self, dis_feat, x):
        x_ids = []
        for j in range(x):
            max = -2
            for i in range(self.data.__len__()):
                if i not in x_ids:
                    curr_cs = self.cosine_similarity(dis_feat, self.dis_feat_file[i])
                    if curr_cs > max:
                        max = curr_cs
                        best_id = i
            x_ids.append(best_id)
        return x_ids

    def get_manipulation_info(self, manipulation):
        manipulation = manipulation.numpy().reshape(151)
        cut_index_np = cut_index_np = np.array(cut_index)
        index_attributes = 0
        result = ""
        for ci in cut_index_np:
            if np.any(manipulation[ci[0]:ci[1]]):
                current_attribute_manipulation = manipulation[ci[0]:ci[1]]
                result += "Manipulation in " + ATTRIBUTES[index_attributes] + " attribute,"
                if -1 in current_attribute_manipulation:
                    values = ATTRIBUTES_VALUES[index_attributes]
                    result += " - " + values[np.where(current_attribute_manipulation == -1)[0][0]]
                if 1 in current_attribute_manipulation:
                    values = ATTRIBUTES_VALUES[index_attributes]
                    result += " + " + values[np.where(current_attribute_manipulation == 1)[0][0]]
                break
            index_attributes += 1

        return result
