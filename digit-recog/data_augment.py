import os

import cv2 as cv
import numpy as np

def augemnt_by_scaling(imgs, border_size):
    """
    Augments image data by scaling it a little [adds/subtracts border]
    Assumes image is grayscale and the caller is
        intelligent enough to not give an extremely
        large border for cutting off
    """

    # no change reqd. don't waste time, simpy return
    if border_size == 0:
        return

    for img_path in imgs:
        img = cv.imread(img_path)

        h, w = img.shape[:2]
        _h, _w = h + int(2 * border_size), w + int(2 * border_size)

        if border_size > 0:
            # have to add border
            new_img = np.zeros((_w, _h, 3), dtype=np.uint8)
            new_img[border_size:w+border_size, border_size:h+border_size] = img
        else:
            # have to return a smaller image
            new_img = img[-border_size:w + border_size, -border_size:w + border_size]

        new_img = cv.resize(new_img, (30, 30))

        new_img_path = f"./aug_{border_size}/{img_path}"
        new_img_dir = os.path.dirname(new_img_path)
        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir, exist_ok=True)
        cv.imwrite(new_img_path, new_img)

def augemnt_by_rotation(imgs, angle):
    """
    Augments image data by rotating it a little
    """
    for img_path in imgs:
        img = cv.imread(img_path)

        h, w = img.shape[:2]

        M = cv.getRotationMatrix2D((h/2, w/2), angle, 1)
        new_img = cv.warpAffine(img, M, (w, h))

        new_img_path = f"./aug_rot{angle}/{img_path}"
        new_img_dir = os.path.dirname(new_img_path)
        if not os.path.exists(new_img_path):
            os.makedirs(new_img_dir, exist_ok=True)
        cv.imwrite(new_img_path, new_img)

"""
Default augmentation:
 - added black border of {1..4}px around the image to simulate smaller digits
 - removed black border of {1..4}px from the image to simulate larger digits
 - rotated {4, 8, 12, 16} degrees clockwise and anticlockwise to simulate weird orientations

"""
imgs = []
for dig in range(10):
    dirname = f"train/{dig}/"
    for f in os.listdir(f"{dirname}"):
        imgs.append(dirname + f)

for i in range(1, 5):
    augemnt_by_scaling(imgs, i)
    augemnt_by_scaling(imgs, -i)

    augemnt_by_rotation(imgs, i*4)
    augemnt_by_rotation(imgs, -i*4)
