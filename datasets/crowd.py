import json
import os
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(inner_down - inner_up, 0.0)
    return inner_area


def get_im_list(root_path, json_file):
    with open(json_file) as f:
        im_list = json.load(f)
    im_list = [os.path.join(root_path, x.split('/')[-1]) for x in im_list]
    return im_list


def train_val(im_list, ratio=0.9):
    N = int(float(len(im_list)) * ratio)
    idx = torch.randperm(len(im_list))
    train_list = [im_list[i] for i in idx[0:N]]
    val_list = [im_list[i] for i in idx[N + 1:]]
    return train_list, val_list


def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map

    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))
    p_index = torch.from_numpy(p_h * im_width + p_w)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index,
                                                                  src=torch.ones(im_width * im_height)).view(im_height,
                                                                                                             im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train', resize=False, im_list=None, noise=0, extra_aug=True):

        self.noise = noise
        self.root_path = root_path
        self.resize = resize
        self.extra_aug = extra_aug
        if im_list is None:
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        else:
            self.im_list = im_list
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        # self.d_ratio = downsample_ratio
        # assert self.c_size % self.d_ratio == 0
        # self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return 1 * len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item % len(self.im_list)]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        keypoints = np.load(gd_path)

        if self.method == 'train':
            return self.train_transform_with_crop(img, keypoints)
        elif self.method == 'val':
            img = self.trans(img)
            name = os.path.splitext(os.path.basename(img_path))[0]
            if len(keypoints) == 0:
                keypoints = torch.zeros(size=(1, 1))
            return img, keypoints, name

    def train_transform_with_crop(self, img, keypoints):
        """random crop image patch and find people in it"""
        # pillow: first width then height
        wd, ht = img.size
        if self.extra_aug:
            # assert len(keypoints) > 0
            if random.random() > 0.88:
                img = img.convert('L').convert('RGB')
            re_size = random.random() * 0.5 + 0.75
            wdd = (int)(wd * re_size)
            htt = (int)(ht * re_size)
            if min(wdd, htt) >= self.c_size:
                wd = wdd
                ht = htt
                img = img.resize((wd, ht))
                keypoints = keypoints * re_size
        st_size = min(wd, ht)
        if st_size < self.c_size:
            c_size = 512
        else:
            c_size = self.c_size
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, c_size, c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) < 1:
            if random.random() > 0.5:
                img = F.hflip(img)
            return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
                torch.from_numpy(keypoints.copy()).float(), st_size
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] < w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] < h)
        keypoints = keypoints[idx_mask]
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
            torch.from_numpy(target.copy()).float(), st_size


class Crowd_sh(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train', extra_aug=True):
        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        self.extra_aug = extra_aug
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.splitext(os.path.basename(img_path))[0]
        gd_path = os.path.join(self.root_path, name + '.npy')
        img = Image.open(img_path).convert('RGB')
        keypoints = np.load(gd_path)[:, :2]

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            img = self.trans(img)
            if len(keypoints) == 0:
                keypoints = torch.zeros(size=(1, 1))
            return img, keypoints, name

    def train_transform(self, img, keypoints):
        # pillow: first width then height
        wd, ht = img.size
        if self.extra_aug:
            # assert len(keypoints) > 0
            if random.random() > 0.88:
                img = img.convert('L').convert('RGB')
            re_size = random.random() * 0.5 + 0.75
            wdd = (int)(wd * re_size)
            htt = (int)(ht * re_size)
            if min(wdd, htt) >= self.c_size:
                wd = wdd
                ht = htt
                img = img.resize((wd, ht))
                keypoints = keypoints * re_size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] < w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] < h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float(), st_size


if __name__ == '__main__':
    dataset = Crowd_sh(os.path.join('/home/icml007/Nightmare4214/datasets/ShanghaiTech_Crowd_Counting_Dataset-Train-Val-Test/part_A', 'val'), 256, method='val')
    for x in dataset:
        print(x[1])