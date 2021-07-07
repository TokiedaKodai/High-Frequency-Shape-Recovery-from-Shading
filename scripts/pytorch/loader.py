import os
import sys

import numpy as np
import cv2
from tqdm import tqdm
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
sys.path.append('../../')

from utils import tools
import config

def LoadData(num, start=0):
    ''' Load data on memory '''
    patch_num = (
        int(config.shape_img[0] / config.shape_patch[0]),
        int(config.shape_img[1] / config.shape_patch[1])
    )

    dir_data = config.dir_root_data + config.dir_synthetic
    file_gt = dir_data + config.dir_gt + '{:05d}.bmp'
    file_low = dir_data + config.dir_low + '{:05d}.bmp'
    file_shade = dir_data + config.dir_shade + '{:05d}.png'
    file_proj = dir_data + config.dir_proj + '{:05d}.png'

    data = []

    for idx in tqdm(range(start, start+num)):
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
        ''' Learn difference between True depth and Low depth'''
        diff = (gt - low) * mask
        ''' Normalize '''
        if config.is_norm_diff:
            len_mask = np.sum(mask)
            mean = np.sum(diff) / len_mask
            std = np.sqrt(np.sum(np.square(diff)) / len_mask)
            diff = ((diff - mean) / std) * mask
        if config.is_norm_shade:
            shade = shade / np.max(shade)
        else:
            shade = shade / 255.
        proj = proj / 255.
        ''' To patch '''
        ph, pw = config.shape_patch
        for h in range(patch_num[0]):
            for w in range(patch_num[1]):
                mask_patch = mask[h*ph:(h+1)*ph, w*pw:(w+1)*pw]
                if np.sum(mask_patch) > ph*pw*config.threshold_valid:
                    tmp = np.dstack([
                        shade[h*ph:(h+1)*ph, w*pw:(w+1)*pw],
                        proj[h*ph:(h+1)*ph, w*pw:(w+1)*pw],
                        low[h*ph:(h+1)*ph, w*pw:(w+1)*pw],
                        diff[h*ph:(h+1)*ph, w*pw:(w+1)*pw],
                        mask[h*ph:(h+1)*ph, w*pw:(w+1)*pw]
                    ])
                    tmp = tmp.transpose(2, 0, 1)
                    data.append(tmp)
    
    data = np.array(data)
    data = torch.from_numpy(data)

    return data.float()

# Test Loader
def TestLoader(idx, start=config.synthetic_test[0], kind='synthetic_test'):
    ''' Data Loader for Test phase '''

    idx += start

    if kind == 'synthetic_test':
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
    if config.is_norm_shade:
        shade = shade / np.max(shade)
    else:
        shade = shade / 255.
    proj = proj / 255.

    x = np.dstack([shade, proj, low])
    x = x[np.newaxis, :, :, :]

    x = torch.from_numpy(x)
    x = x.permute(0, 3, 1, 2)

    return x.float(), gt, low, shade, mask
