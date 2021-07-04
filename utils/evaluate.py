import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.append('../')
sys.path.append('../../')

from scripts import config

# Normalize patch size
p = config.size_norm_patch
patch_rate = config.rate_valid_norm_patch

def norm_diff(pred, gt, mask):
    len_mask = np.sum(mask)
    pred *= mask
    gt *= mask

    mean_pred = np.sum(pred) / len_mask
    mean_gt = np.sum(gt) / len_mask

    std_pred = np.sqrt(np.sum(np.square(pred - mean_pred)) / len_mask)
    std_gt = np.sqrt(np.sum(np.square(gt - mean_gt)) / len_mask)

    return ((pred - mean_pred) * (std_gt / std_pred) + mean_gt) * mask

def norm_diff_pix_patch(pred, gt, mask):
    shapes = pred.shape
    normed = np.zeros_like(pred)
    new_mask = mask.copy()

    pred *= mask
    gt *= mask

    cnt = 0
    for i in range(p, shapes[0]-p):
        for j in range(p, shapes[1]-p):
            cnt += 1
            if not mask[i,j]:
                normed[i,j] = 0
                continue

            local_mask = mask[i-p:i+p+1, j-p:j+p+1]
            local_gt = gt[i-p:i+p+1, j-p:j+p+1]
            local_pred = pred[i-p:i+p+1, j-p:j+p+1]
            local_mask_len = np.sum(local_mask)
            patch_len = (p*2 + 1) ** 2
            if local_mask_len < patch_len*patch_rate/100:
                normed[i, j] = 0
                new_mask[i, j] = 0
                continue
            local_mean_gt = np.sum(local_gt) / local_mask_len
            local_mean_pred = np.sum(local_pred) / local_mask_len
            local_sd_gt = np.sqrt(np.sum(np.square(local_gt - local_mean_gt)) / local_mask_len)
            local_sd_pred = np.sqrt(np.sum(np.square(local_pred - local_mean_pred)) / local_mask_len)
            normed[i, j] = (pred[i, j] - local_mean_pred) * (local_sd_gt / local_sd_pred) + local_mean_gt

    new_mask[:p, :] = 0
    new_mask[shapes[0]-p:, :] = 0
    new_mask[:, :p] = 0
    new_mask[:, shapes[0]-p:] = 0

    return normed, new_mask

def calc_rmse(pred, gt, mask):
    length = np.sum(mask)
    rmse = np.sqrt(np.sum(np.square(pred - gt)*mask) / length)
    return rmse
