#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
from cv2 import cv2
import argparse
import shutil
import scipy.io as sio
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.contrib import tzip


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin_dir',
                        default='/home/icml007/Nightmare4214/datasets/ShanghaiTech_Crowd_Counting_Dataset/part_B',
                        help='original data directory')
    parser.add_argument('--data_dir',
                        default='/home/icml007/Nightmare4214/datasets/ShanghaiTech_Crowd_Counting_Dataset-Train-Val-Test/part_B',
                        help='processed data directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    origin_dir = args.origin_dir
    save_dir = args.data_dir
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)

    train_dir = os.path.join(origin_dir, 'train_data')
    train_image_dir = os.path.join(train_dir, 'images')
    train_gt_dir = os.path.join(train_dir, 'ground-truth')

    test_dir = os.path.join(origin_dir, 'test_data')
    test_image_dir = os.path.join(test_dir, 'images')
    test_gt_dir = os.path.join(test_dir, 'ground-truth')

    train_image_paths = list(glob(os.path.join(train_image_dir, '*')))
    train_image_paths.sort()
    train_gt_paths = list(glob(os.path.join(train_gt_dir, '*')))
    train_gt_paths.sort()
    train_image_paths, val_image_paths, train_gt_paths, val_gt_paths = train_test_split(train_image_paths, train_gt_paths, test_size=0.2, random_state=42)

    test_image_paths = list(glob(os.path.join(test_image_dir, '*')))
    test_image_paths.sort()
    test_gt_paths = list(glob(os.path.join(test_gt_dir, '*')))
    test_gt_paths.sort()

    for train_image_path, train_gt_path in tzip(train_image_paths, train_gt_paths):
        target_train_image_path = os.path.join(save_dir, 'train', os.path.relpath(train_image_path, train_image_dir))
        shutil.copy(train_image_path, target_train_image_path)
        keypoints = sio.loadmat(train_gt_path)['image_info'][0][0][0][0][0].astype(float)
        np.save(os.path.splitext(target_train_image_path)[0] + '.npy', keypoints)

    for val_image_path, val_gt_path in tzip(val_image_paths, val_gt_paths):
        target_val_image_path = os.path.join(save_dir, 'val', os.path.relpath(val_image_path, train_image_dir))
        shutil.copy(val_image_path, target_val_image_path)
        keypoints = sio.loadmat(val_gt_path)['image_info'][0][0][0][0][0].astype(float)
        np.save(os.path.splitext(target_val_image_path)[0] + '.npy', keypoints)

    for test_image_path, test_gt_path in tzip(test_image_paths, test_gt_paths):
        target_test_image_path = os.path.join(save_dir, 'test', os.path.relpath(test_image_path, test_image_dir))
        shutil.copy(test_image_path, target_test_image_path)
        keypoints = sio.loadmat(test_gt_path)['image_info'][0][0][0][0][0].astype(float)
        np.save(os.path.splitext(target_test_image_path)[0] + '.npy', keypoints)

