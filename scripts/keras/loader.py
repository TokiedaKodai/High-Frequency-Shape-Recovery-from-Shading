import os
import sys

import numpy as np
import cv2
from tqdm import tqdm
from keras.utils import Sequence

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
sys.path.append('../../')

from utils import tools
import config

class SyntheticGenerator(Sequence):
    ''' Data Loader for Synthetic data

    Crop image into patches and return them as a batch.

    Args:
        num : image num
        start : start image index (default: 0)
        batch_size : batch size
    Returns:
        training batch in [start: start + num]
    '''
    def __init__(self, num, start=0, batch_size=8):
        self.num = num
        self.start = start
        self.batch_size = batch_size

        self.patch_num = (
            int(config.shape_img[0] / config.shape_patch[0]),
            int(config.shape_img[1] / config.shape_patch[1])
        )
        self.patch_size = self.patch_num[0] * self.patch_num[1]
        self.length = self.patch_size * num
        self.batches_per_img = int(self.patch_size / batch_size)
        self.batches_per_epoch = int(self.length / batch_size)

        dir_data = config.dir_root_data + config.dir_synthetic
        self.file_gt = dir_data + config.dir_gt + '{:05d}.bmp'
        self.file_low = dir_data + config.dir_low + '{:05d}.bmp'
        self.file_shade = dir_data + config.dir_shade + '{:05d}.png'
        self.file_proj = dir_data + config.dir_proj + '{:05d}.png'

    def __getitem__(self, idx):
        idx_img = int(idx / self.batches_per_img) + self.start
        idx_batch = idx % self.batches_per_img
        idx_patch = int(idx_batch * self.batch_size / self.batches_per_img)
        ''' Load image'''
        gt_img = cv2.imread(self.file_gt.format(idx_img), -1)
        low_img = cv2.imread(self.file_low.format(idx_img), -1)
        shade = cv2.imread(self.file_shade.format(idx_img), 0)
        proj = cv2.imread(self.file_proj.format(idx_img), 0)
        ''' image to depth'''
        gt = tools.unpack_bmp_bgra_to_float(gt_img)
        low = tools.unpack_bmp_bgra_to_float(low_img)
        ''' Mask '''
        is_gt_valid = gt > config.threshold_depth
        is_low_valid = low > config.threshold_depth
        is_shade_valid = np.array(shade) > config.threshold_shade
        mask = (is_gt_valid * is_low_valid * is_shade_valid) * 1.
        ''' Learn difference between True depth and Low depth'''
        diff = (gt - low) * mask
        ''' Normalize '''
        if config.is_norm_diff:
            len_mask = np.sum(mask)
            mean = np.sum(diff) / len_mask
            std = np.sqrt(np.sum(np.square(diff)) / len_mask)
            diff = ((diff - mean) / std) * mask
        shade = shade / 255.
        proj = proj / 255.
        ''' Batch'''
        batch_x = []
        batch_y = []
        ph, pw = config.shape_patch
        for i in range(self.batch_size):
            i += idx_patch
            h = int(i / self.patch_num[0])
            w = i % self.patch_num[1]
            batch_x.append(np.dstack([
                shade[h*ph:(h+1)*ph, w*pw:(w+1)*pw],
                proj[h*ph:(h+1)*ph, w*pw:(w+1)*pw],
                low[h*ph:(h+1)*ph, w*pw:(w+1)*pw]
            ]))
            batch_y.append(np.dstack([
                diff[h*ph:(h+1)*ph, w*pw:(w+1)*pw],
                mask[h*ph:(h+1)*ph, w*pw:(w+1)*pw]
            ]))

        return np.array(batch_x), np.array(batch_y)

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        pass


def TestLoader(idx, start=config.synthetic_test[0], kind='synthetic'):
    ''' Data Loader for Test phase '''

    idx += start

    if kind == 'synthetic':
        dir_data = config.dir_root_data + config.dir_synthetic
    elif kind == 'real':
        dir_data = config.dir_root_data + config.dir_real
    file_gt = dir_data + config.dir_gt + '{:05d}.bmp'
    file_low = dir_data + config.dir_low + '{:05d}.bmp'
    file_shade = dir_data + config.dir_shade + '{:05d}.png'
    file_proj = dir_data + config.dir_proj + '{:05d}.png'

    ''' Load image'''
    gt_img = cv2.imread(file_gt.format(idx), -1)
    low_img = cv2.imread(file_low.format(idx), -1)
    shade = cv2.imread(file_shade.format(idx), 0)
    proj = cv2.imread(file_proj.format(idx), 0)
    ''' image to depth'''
    gt = tools.unpack_bmp_bgra_to_float(gt_img)
    low = tools.unpack_bmp_bgra_to_float(low_img)
    ''' Mask '''
    is_gt_valid = gt > config.threshold_depth
    is_low_valid = low > config.threshold_depth
    is_shade_valid = np.array(shade) > config.threshold_shade
    mask = (is_gt_valid * is_low_valid * is_shade_valid) * 1.
    ''' Normalize '''
    shade = shade / 255.
    proj = proj / 255.

    inputs = np.dstack([shade, proj, low])
    inputs = inputs[np.newaxis, :, :, :]

    return inputs, gt, low, shade, mask

def LoadData(generator):
    ''' Load data on memory '''

    data_x = []
    data_y = []

    for batch_x, batch_y in tqdm(generator):
        data_x.append(batch_x[0])
        data_y.append(batch_y[0])
    
    return np.array(data_x), np.array(data_y)