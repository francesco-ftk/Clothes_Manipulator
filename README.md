# Clothes_Manipulator
COMPUTER VISION and INTELLIGENT MULTIMEDIA RECOGNITION Project Exam

## Before Cloning: git-lfs

Install [git-lfs](https://git-lfs.github.com/) before cloning by following the instructions [here](https://github.com/git-lfs/git-lfs/wiki/Installation).

# Installation
1. Clone [amazon's ADDE-M repo](https://github.com/amzn/fashion-attribute-disentanglement)
2. Install requirements with 
```bash
conda create -n adde-m python=3.7
conda activate adde-m
conda install --file requirements.txt
pip install faiss-cpu==1.6.3 torchvision==0.5.0
```
3. Clone [ClothesManipulator repo](https://github.com/sim-pez/ClothesManipulator) in the same folder
3. Install requirements with 
```bash
pip install -r requirements.txt 
```
4. Clone my repo in the same folder of the amazon's ADDE-M repo
5. `run missing_file_creator.py` for create the missing files of previous repo



These are the files that one would need to create:
+ `imgs_train.txt`,`imgs_test.txt`: they store the name of images in relative path for training and testing.
+ `labels_train.txt`,`labels_test.txt`: they store one-hot attribute labels, used to train the attribute-driven disentangled encoder. 
`labels_*.txt` is paired with `imgs_*.txt`, that is, the i-th line in  `labels_*.txt`  is the label vector for the i-th line image in `img_*.txt`.
+ `attr_num.txt`: a list that consists of the number of attribute values for each attribute type.

   For example in Shopping100k, we list is `[16, 17, 19, 14, 10, 15, 2, 11, 16, 7, 9, 15]` since there are 12 attribute types in total, and the first attribute (category) has 16 values, the second has 17 and so on.
