import os
import logging
from datetime import datetime
from utils.logger import setlogger


include_keys = ['max_epoch', 'crop_size', 'downsample_ratio', 'lr', 'lr_lbfgs', 'scheduler', 'cost', 'scale', 'reach',
                'blur', 'scaling', 'tau', 'p', 'p_norm', 'norm_coord', 'phi', 'd_point', 'd_pixel', 'batch_size']

def get_run_name_by_args(args, include_keys=None, exclude_keys=None):
    data = args.__dict__
    result = []
    if include_keys:
        for k in include_keys:
            result.append(f'{k}_{data[k]}')
    else:
        for k, v in data.items():
            if exclude_keys and k in exclude_keys:
                continue
            result.append(f'{k}_{v}')
    return '_'.join(result)


def rename_if_exist(path):
    base_path = path
    i = 1
    while os.path.exists(path):
        path = base_path + f'_({i})'
    return path


class Trainer(object):
    def __init__(self, args):
        self.save_dir = os.path.join(args.save_dir, get_run_name_by_args(args, include_keys) + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S'))
        args.save_dir = self.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger
        for k, v in args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))
        self.args = args

    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        pass

    def train(self):
        """training one epoch"""
        pass
