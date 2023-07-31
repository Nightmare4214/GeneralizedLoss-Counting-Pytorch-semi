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


def find_dis(point):
    # a = point[:, None, :]
    # b = point[None, ...]
    # dis = np.linalg.norm(a - b, ord=2, axis=-1)  # dis_{i,j} = ||p_i - p_j||
    # dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)

    square = np.sum(point * point, axis=1)
    # dis_{i,j} = ||p_i - p_j||
    dis = np.sqrt(np.maximum(square[:, None] - 2 * np.matmul(point, point.T) + square[None, :], 0.0))
    # mean(4th_min, 2 of the [1th_min, 2nd_min, 3rd_min])
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis


def generate_data(image_path, gt_path):
    im = Image.open(image_path)
    im_w, im_h = im.size
    points = sio.loadmat(gt_path)['image_info'][0][0][0][0][0].astype(float)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    # im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    # im = np.array(im)
    # if rr != 1.0:
    #     im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
    #     points = points * rr
    # return Image.fromarray(im), points
    return im, points



def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin_dir',
                        default='/mnt/data/datasets/ShanghaiTech_Crowd_Counting_Dataset/part_B_final',
                        help='original data directory')
    parser.add_argument('--data_dir',
                        default='/mnt/data/datasets/ShanghaiTech_Crowd_Counting_Dataset-Train-Val-Test/part_B',
                        help='processed data directory')
    parser.add_argument('--part',
                        default='B',
                        help='processed data directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    origin_dir = args.origin_dir
    save_dir = args.data_dir
    part = args.part
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)

    train_dir = os.path.join(origin_dir, 'train_data')
    train_image_dir = os.path.join(train_dir, 'images')
    train_gt_dir = os.path.join(train_dir, 'ground_truth')

    test_dir = os.path.join(origin_dir, 'test_data')
    test_image_dir = os.path.join(test_dir, 'images')
    test_gt_dir = os.path.join(test_dir, 'ground_truth')


    train_image_paths = []
    train_gt_paths = []
    val_image_paths = []
    val_gt_paths = []
    test_image_paths = []
    test_gt_paths = []

    for line in open(f'part_{part}_train.txt', 'r'):
        line = line.strip()
        if not line:
            continue
        train_image_paths.append(os.path.join(train_image_dir, line))
        train_gt_paths.append(os.path.join(train_gt_dir, 'GT_{}.mat'.format(os.path.splitext(line)[0])))
    
    for line in open(f'part_{part}_val.txt', 'r'):
        line = line.strip()
        if not line:
            continue
        val_image_paths.append(os.path.join(train_image_dir, line))
        val_gt_paths.append(os.path.join(train_gt_dir, 'GT_{}.mat'.format(os.path.splitext(line)[0])))
    
    for line in open(f'part_{part}_test.txt', 'r'):
        line = line.strip()
        if not line:
            continue
        test_image_paths.append(os.path.join(test_image_dir, line))
        test_gt_paths.append(os.path.join(test_gt_dir, 'GT_{}.mat'.format(os.path.splitext(line)[0])))

    for train_image_path, train_gt_path in tzip(train_image_paths, train_gt_paths):
        target_train_image_path = os.path.join(save_dir, 'train', os.path.relpath(train_image_path, train_image_dir))
        shutil.copy(train_image_path, target_train_image_path)
        _, keypoints = generate_data(train_image_path, train_gt_path)
        dis = find_dis(keypoints)
        keypoints = np.concatenate((keypoints, dis), axis=1)
        np.save(os.path.splitext(target_train_image_path)[0] + '.npy', keypoints)

    for val_image_path, val_gt_path in tzip(val_image_paths, val_gt_paths):
        target_val_image_path = os.path.join(save_dir, 'val', os.path.relpath(val_image_path, train_image_dir))
        shutil.copy(val_image_path, target_val_image_path)
        _, keypoints = generate_data(train_image_path, train_gt_path)
        dis = find_dis(keypoints)
        keypoints = np.concatenate((keypoints, dis), axis=1)
        np.save(os.path.splitext(target_val_image_path)[0] + '.npy', keypoints)

    for test_image_path, test_gt_path in tzip(test_image_paths, test_gt_paths):
        target_test_image_path = os.path.join(save_dir, 'test', os.path.relpath(test_image_path, test_image_dir))
        shutil.copy(test_image_path, target_test_image_path)
        _, keypoints = generate_data(train_image_path, train_gt_path)
        dis = find_dis(keypoints)
        keypoints = np.concatenate((keypoints, dis), axis=1)
        np.save(os.path.splitext(target_test_image_path)[0] + '.npy', keypoints)

