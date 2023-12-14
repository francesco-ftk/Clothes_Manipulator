# Clothes_Manipulator
COMPUTER VISION and INTELLIGENT MULTIMEDIA RECOGNITION Project Exam

# Installation
1. Clone this repo
2. Install requirements with 
```bash
pip install faiss-cpu==1.6.3 torchvision==0.5.0
```
3. Download and put `Shopping100k` dataset in `MAN_ClothesManipulator` folder ([Contact the author](https://sites.google.com/view/kenanemirak/home) of the dataset to get access to the images)
4. Run `disentangled_features_extractor.py` to extract all disentangled features from the dataset in `MAN_ClothesManipulator/disentangledFeaturesExtractor` folder (run with `MODE` constant equal to 'train' and equal to 'test')
5. Run `disentangled_features_normalizer.py` to get all disentangled features normalized
6. Download from [amazon's ADDE-M repo](https://github.com/amzn/fashion-attribute-disentanglement) `models/Shopping100k/extractor_best.pkl` and `models/Shopping100k/memory_best.pkl`, then put them with the same path in `MAN_ClothesManipulator` folder
7. Create `MAN_ClothesManipulator/tensorboard` folder

# Utility class for dataset
+ `data_supplier.py` for getting a couple of images with distance less or equal to N (big N)
+ `optimized_data_supplier.py` for getting a couple of images with distance equal to N (small N)

# Train
+ `train.py`

# Test
+ `test.py` for visual evaluation and other tests
+ `eval.py` for comparing with Amazon
