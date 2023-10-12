# Used to extract all features from dataset (ADDE)

import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from dataloader import Data
from model import Extractor
import constants as C


if not torch.cuda.is_available():
    print('Warning: Using CPU')
else:
    torch.cuda.set_device(0)

file_root = r"../splits/Shopping100k"
img_root_path = r"../Shopping100k/Images"

# load dataset
print('Loading gallery...')
gallery_data = Data(file_root, img_root_path,
                    transforms.Compose([
                        transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                    ]), mode='train')  # test

gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=64, shuffle=False,
                                             sampler=torch.utils.data.SequentialSampler(gallery_data),
                                             num_workers=16,
                                             drop_last=False)

model = Extractor(gallery_data.attr_num, backbone="alexnet", dim_chunk=340)
model_pretrained = r"../models/Shopping100k/extractor_best.pkl"
print('load {path} \n'.format(path=model_pretrained))
model.load_state_dict(torch.load(model_pretrained))
model.cuda()
model.eval()  # say not training but evaluation and sets some values

# indexing the gallery
gallery_feat = []
with torch.no_grad():  # say not training but evaluation and sets some values
    for i, (img, _) in enumerate(tqdm(gallery_loader)):
        img = img.cuda()
        dis_feat, _ = model(img)
        gallery_feat.append(torch.cat(dis_feat, 1).squeeze().cpu().numpy())

dim_chunk = 340  # 340 values for feature
gallery_feat = np.concatenate(gallery_feat, axis=0).reshape(-1, dim_chunk * len(gallery_data.attr_num))

print("shape of gallery_feat_train: {}".format(len(gallery_feat)))  # test
np.save("../disentangledFeaturesExtractor/feat_train_senzaNorm.npy", gallery_feat)  # test
print('Saved indexed features at /feat_train_senzaNorm.npy')  # test

# np.load() for reload
