import os
import sys

import cv2
from keras import models
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.engine.training_arrays import predict_loop

dir_current = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_current)

sys.path.append('../')
sys.path.append('../../')

import network, loader
import config
from utils import parser, tools, evaluate, plots

os.chdir(dir_current)

args = parser.parser_train.parse_args()
################################## SETTING ##################################
''' Directory, File '''
dir_model = config.dir_root_model + args.name + '/'
file_log = dir_model + config.file_log
file_model_final = dir_model + config.file_model_keras_final
file_model_best = dir_model + config.file_model_keras_best
''' Output directory'''
dir_output = config.dir_root_output + args.name + '/'
''' Network '''
is_input_low = config.is_input_low
is_input_proj = config.is_input_proj
num_ch = sum([is_input_low, is_input_proj]) + 1
shape_patch = config.shape_patch
shape_img = config.shape_img
''' Data '''
num = args.num
''' Normalization '''
# is_norm_shade = config.is_norm_shade
is_norm_diff = config.is_norm_diff
is_patch_norm = config.is_patch_norm
''' Save '''
is_save_ply = args.ply
is_save_bmp = args.bmp

################################## RUN ##################################
os.makedirs(dir_output, exist_ok=True)
''' Network '''
net = network.BuildUnet(num_ch, shape_img)
print(os.getcwd())
net.load_weights(file_model_best)

for idx in range(num):
    ''' Load test data '''
    inputs, gt, low, shade, mask = loader.TestLoader(idx)
    ''' Predict'''
    output = net.predict(inputs, batch_size=1)
    pred = output[0, :, :, 0]

    ''' Normalize for evaluation '''
    gt *= mask
    low *= mask
    pred *= mask
    diff = (gt - low) * mask
    if is_norm_diff:
        pred = evaluate.norm_diff(pred, diff, mask)
    if is_patch_norm:
        pred, mask = evaluate.norm_diff_(pred, diff, mask)
        gt *= mask
        low *= mask
        pred *= mask

    ''' Prediction = Output + Low-res depth '''
    pred = (pred + low) * mask

    ''' Plot results '''
    plots.plot_result(dir_current, dir_output, idx, gt, low, pred, shade, mask)
    ''' Save PLY file '''
    if is_save_ply:
        xyz_gt = tools.convert_depth_to_coords(gt, config.cam_params)
        xyz_low = tools.convert_depth_to_coords(low, config.cam_params)
        xyz_pred = tools.convert_depth_to_coords(pred, config.cam_params)
        tools.dump_ply(dir_output + 'ply_gt_{:03d}.ply'.format(idx), xyz_gt.reshape(-1, 3).tolist())
        tools.dump_ply(dir_output + 'ply_low-{:03d}.ply'.format(idx), xyz_low.reshape(-1, 3).tolist())
        tools.dump_ply(dir_output + 'ply_pred-{:03d}.ply'.format(idx), xyz_pred.reshape(-1, 3).tolist())
    if is_save_bmp:
        img_pred = tools.pack_float_to_bmp_bgra(pred)
        cv2.imwrite(dir_output + 'pred-{:03d}.bmp'.format(idx),img_pred)
    