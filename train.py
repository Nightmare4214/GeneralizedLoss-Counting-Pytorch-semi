import argparse
import os

import torch

from utils.emd_dot_trainer import EMDTrainer

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--save_dir', default='/home/icml007/Nightmare4214/PyTorch_model/GL_semi',
                        help='directory to save models.')
    parser.add_argument('--data_dir', default='/home/icml007/Nightmare4214/datasets/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--dataset', default='qnrf', help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--o_cn', type=int, default=1,
                        help='output channel number')
    parser.add_argument('--cost', type=str, default='p_norm',
                        help='cost distance')
    parser.add_argument('--scale', type=float, default=0.6,
                        help='scale for coordinates')  # perspective-guided cost
    parser.add_argument('--reach', type=float, default=0.5,
                        help='reach')
    parser.add_argument('--blur', type=float, default=0.01,
                        help='blur')  # epsilon
    parser.add_argument('--scaling', type=float, default=0.5,
                        help='scaling')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='blur')
    parser.add_argument('--p', type=float, default=1,
                        help='p')  # ?
    parser.add_argument('--p_norm', type=float, default=2,
                        help='p_norm')
    parser.add_argument('--norm_coord', type=int, default=1,
                        help='normalize coordination to calculate the cost matrix')
    parser.add_argument('--d_point', type=str, default='l1',
                        help='divergence for point loss')
    parser.add_argument('--d_pixel', type=str, default='l2',
                        help='divergence for pixel loss')
    parser.add_argument('--phi', type=str, default='kl',
                        help='divergence')
    parser.add_argument('--rho', type=float, default=1,
                        help='rho')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--lr_lbfgs', type=float, default=1,
                        help='the learning rate for lbfgs')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='the weight decay')
    parser.add_argument('--scheduler', type=str, default='poly',
                        help='scheduler')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max_model_num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max_epoch', type=int, default=500,
                        help='max training epoch')
    parser.add_argument('--val_epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val_start', type=int, default=10,
                        help='the epoch start to val')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--is_gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--downsample_ratio', type=int, default=8,
                        help='downsample ratio')
    parser.add_argument('--extra_aug', default=False, required=False, action='store_true', help='extra_aug')
    parser.add_argument('--randomless', default=False, required=False, action='store_true', help='randomless')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 384
        args.val_epoch = 50
    elif args.dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    else:
        raise NotImplementedError
    return args


if __name__ == '__main__':
    args = parse_args()
    # torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = EMDTrainer(args)
    trainer.setup()
    trainer.train()
    trainer.test()
