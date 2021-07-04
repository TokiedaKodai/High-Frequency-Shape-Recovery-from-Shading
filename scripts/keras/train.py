import os
import sys

from keras.callbacks import CSVLogger, ModelCheckpoint

dir_current = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_current)
sys.path.append('../')
sys.path.append('../../')

import network, loader
import config
from utils import parser, plots

os.chdir(dir_current)
################################## SETTING ##################################

args = parser.parser_train.parse_args()
''' Directory, File '''
dir_model = config.dir_root_model + args.name + '/'
file_log = dir_model + config.file_log
file_model_final = dir_model + config.file_model_keras_final
file_model_best = dir_model + config.file_model_keras_best
''' Training Parameters '''
epoch = args.epoch
num_data = args.num
lr = args.lr
rate_drop = args.drop
size_batch = args.batch
# size_batch = 
rate_val = args.val
verbose = args.verbose
is_aug = args.aug
is_retrain = args.retrain
is_finetune = args.finetune
is_load_memory = args.load
''' Network '''
is_input_low = config.is_input_low
is_input_proj = config.is_input_proj
num_ch = sum([is_input_low, is_input_proj]) + 1
shape_patch = config.shape_patch
''' Data '''
num_train = int(num_data * (1 - rate_val))
num_val = int(num_data * rate_val)
''' Normalization '''
# is_norm_shade = config.is_norm_shade
# is_norm_diff = config.is_norm_diff

################################## RUN ##################################
os.makedirs(dir_model, exist_ok=True)

''' Data Generator '''
if is_load_memory:
    generator = loader.SyntheticGenerator(num_data, batch_size=1)
else:
    generator_train = loader.SyntheticGenerator(num_train, batch_size=size_batch)
    generator_val = loader.SyntheticGenerator(num_val, num_train, batch_size=size_batch)
''' Network '''
net = network.BuildUnet(num_ch, shape_patch, lr, rate_drop)
''' Callbacks '''
save_final = ModelCheckpoint(
    file_model_final,
    save_weights_only=True,
    period=1
)
save_best = ModelCheckpoint(
    file_model_best,
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    period=1
)
csv_logger = CSVLogger(file_log)

''' Load Weight '''
if is_retrain:
    net.load_weights(file_model_final)
elif is_finetune:
    net.load_weights(file_model_best)

''' Training'''
print('Training...')
if is_load_memory:
    data_x, data_y = loader.LoadData(generator)
    net.fit(
        data_x,
        data_y,
        batch_size=size_batch,
        epochs=epoch,
        initial_epoch=0,
        shuffle=True,
        callbacks=[save_final, save_best, csv_logger],
        validation_split=rate_val,
        verbose=verbose
    )
else:
    net.fit_generator(
        generator_train,
        steps_per_epoch=generator_train.batches_per_epoch,
        epochs=epoch,
        initial_epoch=0,
        shuffle=True,
        callbacks=[save_final, save_best, csv_logger],
        validation_data=generator_val,
        validation_steps=generator_val.batches_per_epoch,
        verbose=verbose,
        max_queue_size=10)

plots.plot_graph(dir_current, dir_model, file_log)