import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from MAN_ClothesManipulator.src.utils import get_idx_label


class Data(data.Dataset):
    def __init__(self, file_root, img_root_path, img_transform=None, mode='train'):
        super(Data, self).__init__()

        self.file_root = file_root
        self.img_root_path = img_root_path
        self.img_transform = img_transform
        self.mode = mode
        if not self.img_transform:
            self.img_transform = transforms.ToTensor()

        self.img_data, self.label_data, self.attr_num = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.file_root, "imgs_%s.txt" % self.mode)) as f:
            img_data = f.read().splitlines()

        label_data = np.loadtxt(os.path.join(self.file_root, "labels_%s.txt" % self.mode), dtype=int)
        assert len(img_data) == label_data.shape[0]

        attr_num = np.loadtxt(os.path.join(self.file_root, "attr_num.txt"), dtype=int)

        return img_data, label_data, attr_num

    def __len__(self):
        return self.label_data.shape[0]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_root_path, self.img_data[index]))
        img = img.convert('RGB')
        if self.img_transform:
            img = self.img_transform(img)

        label_vector = self.label_data[index]  # one-hot

        return img, get_idx_label(label_vector, self.attr_num)


"""
DATA QUERY FOR eval.py
"""

class DataQuery(data.Dataset):
    """
    Load generated queries for evaluation. Each query consists of a reference image and an indicator vector
    The indicator vector consists of -1, 1 and 0, which means remove, add, not modify
    Args:
        file_root: path that stores preprocessed files (e.g. imgs_test.txt, see README.md for more explanation)
        img_root_path: path that stores raw images
        ref_ids: the file name of the generated txt file, which includes the indices of reference images
        query_inds: the file name of the generated txt file, which includes the indicator vector for queries.
        img_transform: transformation functions for img. Default: ToTensor()
        mode: the mode 'train' or 'test' decides to load training set or test set
    """
    def __init__(self, file_root,  img_root_path, ref_ids,  query_inds, img_transform=None,
                 mode='test'):
        super(DataQuery, self).__init__()

        self.file_root = file_root
        self.img_transform = img_transform
        self.img_root_path = img_root_path
        self.mode = mode
        self.ref_ids = ref_ids
        self.query_inds = query_inds

        if not self.img_transform:
            self.img_transform = transforms.ToTensor()

        self.img_data, self.label_data, self.ref_idxs, self.query_inds, self.attr_num = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.file_root, "imgs_%s.txt" % self.mode)) as f:
            img_data = f.read().splitlines()

        label_data = np.loadtxt(os.path.join(self.file_root, "labels_%s.txt" % self.mode), dtype=int)

        query_inds = np.loadtxt(os.path.join(self.file_root, self.query_inds), dtype=int)
        ref_idxs = np.loadtxt(os.path.join(self.file_root, self.ref_ids), dtype=int)

        attr_num = np.loadtxt(os.path.join(self.file_root, "attr_num.txt"), dtype=int)

        assert len(img_data) == label_data.shape[0]

        return img_data, label_data, ref_idxs, query_inds, attr_num

    def __len__(self):
        return self.ref_idxs.shape[0]

    def __getitem__(self, index):

        ref_id = int(self.ref_idxs[index])
        img = Image.open(os.path.join(self.img_root_path, self.img_data[ref_id]))
        img = img.convert('RGB')

        if self.img_transform:
            img = self.img_transform(img)

        indicator = self.query_inds[index]

        return img, indicator

