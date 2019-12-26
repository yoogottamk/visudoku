import os
import random
import pickle

import numpy as np
import cv2 as cv
from tqdm import tqdm

CATEGORIES = list(range(10))
IMG_SIZE = 30

train_data = []

def create_train_data(data_dir):
    for c in CATEGORIES:
        path = os.path.join(data_dir, str(c))

        for img_path in tqdm(os.listdir(path)):
            try:
                img = cv.imread(os.path.join(path, img_path), cv.IMREAD_GRAYSCALE)
                train_data.append([img, c])
            except Exception as e:
                pass

augment_dirs = []

"""
Use this only if you have generated
  the default augmented directories
"""
for scale in range(1, 3):
    augment_dirs.append(f"aug_{scale}/train")
    augment_dirs.append(f"aug_{-scale}/train")

    augment_dirs.append(f"aug_rot{scale * 4}/train")
    augment_dirs.append(f"aug_rot{-scale * 4}/train")

create_train_data('train')

for aug_dir in augment_dirs:
    create_train_data(aug_dir)

random.shuffle(train_data)

X = []
y = []

for features, label in train_data:
    X.append(features)
    y.append(label)

X = np.array(X)
X = np.reshape(X, (-1, IMG_SIZE, IMG_SIZE, 1))

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
