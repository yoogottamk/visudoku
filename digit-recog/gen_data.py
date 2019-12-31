import os
import random
import pickle

import numpy as np
import cv2 as cv

CATEGORIES = list(range(10))
IMG_SIZE = 30
CWD = os.path.dirname(os.path.realpath(__file__))

train_data = []

def create_train_data(data_dir):
    count = 0

    for c in CATEGORIES:
        path = os.path.join(data_dir, str(c))

        for img_path in os.listdir(path):
            try:
                count += 1
                img = cv.imread(os.path.join(path, img_path), cv.IMREAD_GRAYSCALE)
                train_data.append([img, c])
            except Exception as e:
                pass

    print(f"{count} images added")

augment_dirs = []

"""
Use this only if you have generated
  the default augmented directories
"""
for scale in range(1, 3):
    augment_dirs.append(f"{CWD}/aug_{scale}")
    augment_dirs.append(f"{CWD}/aug_{-scale}")

    augment_dirs.append(f"{CWD}/aug_rot{scale * 4}")
    augment_dirs.append(f"{CWD}/aug_rot{-scale * 4}")

create_train_data(f"{CWD}/train")

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

pickle_out = open(f"{CWD}/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open(f"{CWD}/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
