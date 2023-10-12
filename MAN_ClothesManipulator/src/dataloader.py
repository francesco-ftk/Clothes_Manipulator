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
