#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import argparse
import os
import cv2
from glob import glob
import scipy
import pickle
import numpy as np
from PIL import Image
import scipy.io as io
from itertools import islice
from tqdm import tqdm
from matplotlib import pyplot as plt
from sortedcontainers import SortedDict
from scipy.ndimage import gaussian_filter

min_size = 512
max_size = 2048
precomputed_kernels_path = 'gaussian_kernels.pkl'

def generate_gaussian_kernels(out_kernels_path='gaussian_kernels.pkl', round_decimals = 3, sigma_threshold = 4, sigma_min=0, sigma_max=20, num_sigmas=801, normalization=True):
    """
    Computing gaussian filter kernel for sigmas in linspace(sigma_min, sigma_max, num_sigmas) and saving 
    them to dict.     
    """
    kernels_dict = dict()
    sigma_space = np.linspace(sigma_min, sigma_max, num_sigmas)
    for sigma in tqdm(sigma_space):
        sigma = np.round(sigma, decimals=round_decimals) 
        kernel_size = np.ceil(sigma * sigma_threshold).astype(int)

        img_shape  = (kernel_size * 2 + 1, kernel_size * 2 + 1)
        img_center = (img_shape[0] // 2, img_shape[1] // 2)

        arr = np.zeros(img_shape)
        arr[img_center] = 1

        arr = gaussian_filter(arr, sigma, mode='constant')
        if normalization:
            kernel = arr / arr.sum()
        else:
            kernel = arr
        kernels_dict[sigma] = kernel
        
    print(f'Computed {len(sigma_space)} gaussian kernels. Saving them to {out_kernels_path}')

    with open(out_kernels_path, 'wb') as f:
        pickle.dump(kernels_dict, f)


def compute_sigma(gt_count, distance=None, min_sigma=1, method=1, k=3, beta=0.1, fixed_sigma=15):
    """
    Compute sigma for gaussian kernel with different methods :
    * method = 1 : sigma = (sum of distance to k nearest neighbors) / 10
    * method = 2 : sigma = fixed value
    ** if sigma lower than threshold 'min_sigma', then 'min_sigma' will be used
    ** in case of one point on the image sigma = 'fixed_sigma'
    """    
    if gt_count > 1 and distance is not None:
        if method == 1:
            sigma = np.mean(distance[1:k + 1]) * beta
        elif method == 2:
            sigma = fixed_sigma
    else:
        sigma = fixed_sigma
    if sigma < min_sigma:
        sigma = min_sigma
    return sigma


def find_closest_key(sorted_dict, key):
    """
    Find closest key in sorted_dict to 'key'
    """
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))


def gaussian_filter_density(non_zero_points, map_h, map_w, distances=None, kernels_dict=None, min_sigma=2, method=1, k=3, beta=0.1, const_sigma=15):
    """
    Fast gaussian filter implementation : using precomputed distances and kernels
    """
    gt_count = non_zero_points.shape[0]
    non_zero_points = non_zero_points.round().astype(int)
    density_map = np.zeros((map_h, map_w), dtype=np.float32)

    for i in range(gt_count):
        # width, height
        point_y, point_x = non_zero_points[i]
        sigma = compute_sigma(gt_count, distances[i], min_sigma=min_sigma, method=method, k=k, beta=beta, fixed_sigma=const_sigma)
        closest_sigma = find_closest_key(kernels_dict, sigma)
        kernel = kernels_dict[closest_sigma]
        full_kernel_size = kernel.shape[0]
        kernel_size = full_kernel_size // 2

        min_img_x = max(0, point_x - kernel_size)
        min_img_y = max(0, point_y - kernel_size)
        max_img_x = min(point_x + kernel_size, map_h)
        max_img_y = min(point_y + kernel_size, map_w)
        
        kernel_x_min = max(kernel_size - point_x, 0)
        kernel_y_min = max(kernel_size - point_y, 0)
        kernel_x_max = kernel_x_min + max_img_x - min_img_x
        kernel_y_max = kernel_y_min + max_img_y - min_img_y

        density_map[min_img_x:max_img_x, min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max, kernel_y_min:kernel_y_max]
    return density_map


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def generate_data(image_path, flip=False):
    im = Image.open(image_path)
    # pillow: first width then height
    im_w, im_h = im.size
    gt_path = im_path.replace('.jpg', '_ann.mat')
    # (w, h)
    points = io.loadmat(gt_path)['annPoints'].astype(float)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] < im_w) * (points[:, 1] >= 0) * (points[:, 1] < im_h)
    points = points[idx_mask]
    if flip:
        points = points[:, ::-1]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points, im_h, im_w


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin_dir',
                        default='/home/icml016/Nightmare4214/datasets/UCF-QNRF_ECCV18',
                        help='original data directory')
    parser.add_argument('--data_dir',
                        default='/home/icml016/Nightmare4214/datasets/UCF-Train-Val-Test',
                        help='processed data directory')
    parser.add_argument('--method', type=int, default=1,
                        help='1. sigma = beta * (sum of distance to k nearest neighbors), 2. fixed sigma')
    parser.add_argument('--k', type=int, default=3,
                        help='k nearest neighbors')
    parser.add_argument('--fixed_sigma', type=float, default=15,
                        help='fixed sigma')
    parser.add_argument('--min_sigma', type=float, default=2,
                        help='min sigma')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='beta')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    origin_dir = args.origin_dir
    save_dir = args.data_dir
    method = args.method
    k = args.k
    fixed_sigma = args.fixed_sigma
    min_sigma = args.min_sigma
    beta = args.beta
    
    generate_gaussian_kernels(precomputed_kernels_path, round_decimals=3, sigma_threshold=4, sigma_min=0, sigma_max=20, num_sigmas=801)    
    with open(precomputed_kernels_path, 'rb') as f:
        kernels_dict = pickle.load(f)
        kernels_dict = SortedDict(kernels_dict)
    
    
    for phase in ['Train', 'Test']:
        sub_dir = os.path.join(args.origin_dir, phase)
        if phase == 'Train':
            sub_phase_list = ['train', 'val']
            for sub_phase in sub_phase_list:
                sub_save_dir = os.path.join(save_dir, sub_phase)
                os.makedirs(sub_save_dir, exist_ok=True)
                with open('{}.txt'.format(sub_phase)) as f:
                    for i in tqdm(f):
                        i = i.strip()
                        im_path = os.path.join(sub_dir, i)
                        filename = os.path.splitext(i)[0]
                        # name = os.path.basename(im_path)
                        # print(name)
                        im, keypoints, im_h, im_w = generate_data(im_path)
                        tree = scipy.spatial.KDTree(keypoints.copy(), leafsize=1024)  # build kdtree
                        distances, _ = tree.query(keypoints, k=k + 1)  # query kdtree (n, k + 1)
                        density_map = gaussian_filter_density(keypoints, im_h, im_w, distances, kernels_dict, min_sigma, method, k, beta, fixed_sigma)
                        points = np.concatenate((keypoints, distances[:, 1:].mean(axis=1, keepdims=True)), axis=1)
                        im_save_path = os.path.join(sub_save_dir, i)
                        im.save(im_save_path)
                        gd_save_path = os.path.join(sub_save_dir, filename + '.npy')
                        density_map_save_path = os.path.join(sub_save_dir, filename + '_density_map.npy')
                        np.save(gd_save_path, points)
                        np.save(density_map_save_path, density_map)
        else:
            sub_save_dir = os.path.join(save_dir, 'test')
            os.makedirs(sub_save_dir, exist_ok=True)
            im_list = glob(os.path.join(sub_dir, '*jpg'))
            for im_path in tqdm(im_list):
                filename = os.path.splitext(os.path.basename(im_path))[0]
                
                im, keypoints, im_h, im_w = generate_data(im_path)
                tree = scipy.spatial.KDTree(keypoints.copy(), leafsize=1024)  # build kdtree
                distances, _ = tree.query(keypoints, k=k + 1)  # query kdtree (n, k + 1)
                density_map = gaussian_filter_density(keypoints, im_h, im_w, distances, kernels_dict, min_sigma, method, k, beta, fixed_sigma)
                points = np.concatenate((keypoints, distances[:, 1:].mean(axis=1, keepdims=True)), axis=1)
                im_save_path = os.path.join(sub_save_dir, filename + '.jpg')
                im.save(im_save_path)
                gd_save_path = os.path.join(sub_save_dir, filename + '.npy')
                density_map_save_path = os.path.join(sub_save_dir, filename + '_density_map.npy')
                np.save(gd_save_path, points)
                np.save(density_map_save_path, density_map)
