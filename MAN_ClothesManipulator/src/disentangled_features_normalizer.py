import numpy as np
import torchvision.transforms as transforms
from dataloader import Data
import constants as C

# max = 14, min = -13

file_root = r"../splits/Shopping100k"
img_root_path = r"../Shopping100k/Images"


def get_max_min(array):
    return np.max(array), np.min(array)


def normalize(dis_feat, max, min):
    return (dis_feat - min) / (max - min)


disentangled_features_train = np.load("../disentangledFeaturesExtractor/feat_train_senzaNorm.npy")
disentangled_features_test = np.load("../disentangledFeaturesExtractor/feat_test_senzaNorm.npy")

max = -100
min = 100

for i in range(disentangled_features_train.shape[0]):
    current_max, current_min = get_max_min(disentangled_features_train[i])
    if current_max > max:
        max = current_max
    if current_min < min:
        min = current_min

for i in range(disentangled_features_test.shape[0]):
    current_max, current_min = get_max_min(disentangled_features_test[i])
    if current_max > max:
        max = current_max
    if current_min < min:
        min = current_min

gallery_data = Data(file_root, img_root_path,
                    transforms.Compose([
                        transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                    ]), mode='train')

gallery_feat = []
for i in range(disentangled_features_train.shape[0]):
    gallery_feat.append(normalize(disentangled_features_train[i], max, min))

dim_chunk = 340  # 340 values for feature
gallery_feat = np.concatenate(gallery_feat, axis=0).reshape(-1, dim_chunk * len(gallery_data.attr_num))

print("shape of gallery_feat_train: {}".format(len(gallery_feat)))
np.save("../disentangledFeaturesExtractor/feat_train_Norm.npy", gallery_feat)
print('Saved indexed features at /feat_train_Norm.npy')

gallery_data = Data(file_root, img_root_path,
                    transforms.Compose([
                        transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                    ]), mode='test')

gallery_feat = []
for i in range(disentangled_features_test.shape[0]):
    gallery_feat.append(normalize(disentangled_features_test[i], max, min))

dim_chunk = 340  # 340 values for feature
gallery_feat = np.concatenate(gallery_feat, axis=0).reshape(-1, dim_chunk * len(gallery_data.attr_num))

print("shape of gallery_feat_test: {}".format(len(gallery_feat)))
np.save("../disentangledFeaturesExtractor/feat_test_Norm.npy", gallery_feat)
print('Saved indexed features at /feat_test_Norm.npy')
