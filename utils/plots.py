import os
import sys
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.cm import ScalarMappable

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
sys.path.append('../../')

from scripts import config

def plot_graph(dir_current, dir_model, file_log):
    ''' Plot training and validation loss graph'''

    os.chdir(dir_current)

    df = pd.read_csv(file_log)
    df.set_index('epoch', inplace=True)

    mean = df['val_loss'].mean()
    min_train = df['loss'].min()
    min_val = df['val_loss'].min()
    minimum = min(min_train, min_val)

    df.plot(ylim=[minimum/2, mean*2])
    
    plt.savefig(dir_model + 'loss.jpg')
    plt.close('all')

def plot_result(dir_current, dir_save, idx, gt, low, pred, shade, mask):
    ''' Plot result 
    GT depth    | Low-res depth | Prediction depth
    Shading img | Low-res error | Prediction error
    '''

    os.chdir(dir_current)

    ''' Calc error'''
    err_low = np.abs(gt - low) * mask
    err_pred = np.abs(gt - pred) * mask
    ''' Setting '''
    mean_gt = np.sum(gt) / np.sum(mask)
    v_min, v_max = mean_gt - config.depth_range, mean_gt + config.depth_range
    e_max = config.err_range

    fig = plt.figure(figsize=(12, 6))
    plt.rcParams["font.size"] = 18
    gs_master = GridSpec(nrows=2,
                        ncols=2,
                        height_ratios=[1, 1],
                        width_ratios=[5, 0.1],
                        wspace=0.05,
                        hspace=0.05)
    gs_1 = GridSpecFromSubplotSpec(nrows=1,
                                ncols=3,
                                subplot_spec=gs_master[0, 0],
                                wspace=0.05,
                                hspace=0)
    gs_2 = GridSpecFromSubplotSpec(nrows=1,
                                ncols=3,
                                subplot_spec=gs_master[1, 0],
                                wspace=0.05,
                                hspace=0)
    gs_3 = GridSpecFromSubplotSpec(nrows=2,
                                ncols=1,
                                subplot_spec=gs_master[0:1, 1],
                                wspace=0,
                                hspace=0.1)
    ax_enh0 = fig.add_subplot(gs_1[0, 0])
    ax_enh1 = fig.add_subplot(gs_1[0, 1])
    ax_enh2 = fig.add_subplot(gs_1[0, 2])

    ax_misc0 = fig.add_subplot(gs_2[0, 0])
    ax_err_low = fig.add_subplot(gs_2[0, 1])
    ax_err_pred = fig.add_subplot(gs_2[0, 2])

    ax_cb0 = fig.add_subplot(gs_3[0, 0])
    ax_cb1 = fig.add_subplot(gs_3[1, 0])

    for ax in [
            ax_enh0, ax_enh1, ax_enh2,
            ax_misc0, ax_err_low, ax_err_pred
    ]:
        ax.axis('off')

    ''' Depth '''
    ax_enh0.imshow(gt, cmap='jet', vmin=v_min, vmax=v_max)
    ax_enh1.imshow(low, cmap='jet', vmin=v_min, vmax=v_max)
    ax_enh2.imshow(pred, cmap='jet', vmin=v_min, vmax=v_max)

    ax_enh0.set_title('Ground Truth')
    ax_enh1.set_title('Low-res')
    ax_enh2.set_title('Prediction')

    ''' Shading '''
    ax_misc0.imshow(np.dstack([shade,shade,shade]))
    ''' Error map '''
    scale = 1000 # m -> mm
    e_max *= scale
    ax_err_low.imshow(err_low*scale, cmap='jet', vmin=0, vmax=e_max)
    ax_err_pred.imshow(err_pred*scale, cmap='jet', vmin=0, vmax=e_max)

    ''' Colorbar '''
    fig.savefig(io.BytesIO())
    cb_offset = 0

    plt.colorbar(ScalarMappable(colors.Normalize(vmin=v_min, vmax=v_max),
                                cmap='jet'),
                cax=ax_cb0)
    im_pos, cb_pos = ax_enh0.get_position(), ax_cb1.get_position()
    ax_cb0.set_position([
        cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
        im_pos.y1 - im_pos.y0
    ])
    ax_cb0.set_ylabel('Depth [m]')

    plt.colorbar(ScalarMappable(colors.Normalize(vmin=0, vmax=e_max),
                                cmap='jet'),
                cax=ax_cb1)
    im_pos, cb_pos = ax_err_pred.get_position(), ax_cb1.get_position()
    ax_cb1.set_position([
        cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
        im_pos.y1 - im_pos.y0
    ])
    if scale == 1:
        ax_cb1.set_ylabel('Error [m]')
    elif scale == 1000:
        ax_cb1.set_ylabel('Error [mm]')

    ''' Save '''
    plt.savefig(dir_save + 'result-{:03d}.jpg'.format(idx), dpi=300)
    plt.close()