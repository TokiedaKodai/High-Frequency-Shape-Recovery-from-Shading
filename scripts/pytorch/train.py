import os
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dir_current = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_current)
sys.path.append('../')
sys.path.append('../../')

import network, loader
import config
from utils import parser, plots

os.chdir(dir_current)

args = parser.parser_train.parse_args()
################################## SETTING ##################################
''' Directory, File '''
dir_model = config.dir_root_model + args.name + '/'
file_log = dir_model + config.file_log
file_model_final = dir_model + config.file_model_torch_final
file_model_best = dir_model + config.file_model_torch_best
''' Training Parameters '''
end_epoch = args.epoch
num_data = args.num
lr = args.lr
rate_drop = args.drop
size_batch = args.batch
rate_val = args.val
verbose = args.verbose
# is_aug = args.aug
is_retrain = args.retrain
is_finetune = args.finetune
# is_use_generator = args.generator
# is_load_memory = args.load
''' Network '''
is_input_low = config.is_input_low
is_input_proj = config.is_input_proj
num_ch = sum([is_input_low, is_input_proj]) + 1
shape_patch = config.shape_patch
''' Data '''
num_train = int(num_data * (1 - rate_val))
num_val = int(num_data * rate_val)

################################## FUNCTION ##################################
''' Training '''
def train(net, device, loader, optimizer, loss):
    #initialise counters
    running_loss = 0.0
    loss = []
    net.train()

    start_time = time.time()
    for inputs in loader:
        x = inputs[:, :3, :, :]
        gt = inputs[:, 3, :, :]
        mask = inputs[:, 4, :, :]

        optimizer.zero_grad()

        x = x.to(device)
        ''' Output '''
        out = net(x)
        out = out.permute(1, 0, 2, 3)
        pred = out[0]

        mask = mask.to(device)
        gt = gt.to(device)

        loss = loss(pred, gt)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    #end training 
    end_time = time.time()
    running_loss /= len(loader)

    print('  Training Loss :', running_loss, '-- Time:', end_time - start_time, 's', end=verbose)
    torch.save(net.state_dict(), file_model_final)

    return running_loss
''' Validation '''
def validate(net, device, loader, optimizer, loss):
    #initialise counters
    running_loss = 0.0
    loss = []
    net.eval()

    start_time = time.time()
    with torch.no_grad():
        for inputs in loader:
            x = inputs[:, :3, :, :]
            gt = inputs[:, 3, :, :]
            mask = inputs[:, 4, :, :]
            
            optimizer.zero_grad()

            x = x.to(device)
            ''' Output '''
            output = net(x)
            output = output.permute(1, 0, 2, 3)
            pred = output[0]

            mask = mask.to(device)
            gt = gt.to(device)

            loss = loss(pred, gt)
            running_loss += loss.item()

    #end training 
    end_time = time.time()
    running_loss /= len(loader)

    print(' '*23, 'Validation Loss :', running_loss, '-- Time:', end_time - start_time, 's', end=verbose)
    torch.save(net.state_dict(), file_model_final)

    return running_loss
################################## RUN ##################################
os.makedirs(dir_model, exist_ok=True)

''' Check CUDA '''
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
''' Model '''
net = network.BuildUnet(num_ch, rate_drop).float()
net = net.to(device)
loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
''' Load Weight '''
if is_retrain:
    net.load_state_dict(torch.load(file_model_final))
elif is_finetune:
    net.load_state_dict(torch.load(file_model_best))
''' Training'''
print('Training...')
print(f'Training data num: {num_data}')

print('Loading data...')
data = loader.LoadData(num_data)
print(f'Training patch num: {len(data)}')
data_train, data_val = train_test_split(
    data, 
    test_size=rate_val, 
    shuffle=False)
trainloader = DataLoader(data_train, batch_size=size_batch, shuffle=True)
valloader = DataLoader(data_val, batch_size=size_batch, shuffle=False)
print(f'Number of training batches: {len(trainloader)}')
print(f'Number of validation batches: {len(valloader)}')

min_loss = float('inf')

for epoch in range(0, end_epoch):
    print("Epoch: {:5d}/{:5d}".format(epoch + 1, end_epoch), end=' ---- ')

    train_loss = train(net, device, trainloader, optimizer, loss)
    val_loss = train(net, device, valloader, optimizer, loss)

    df_log = pd.DataFrame({
        'epoch': [epoch+1], 
        'loss': [train_loss], 
        'val_loss': [val_loss]
        })
    df_log.set_index('epoch', inplace=True)
    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(net.state_dict(), file_model_best)

    try:
        is_header = not os.path.exists(file_log)
        df_log.to_csv(file_log, mode='a', header=is_header)
    except Exception as e:
        print("log_data Error: " + str(e))
print('\nTraining end.')

''' Plot loss graph '''
plots.plot_graph(dir_current, dir_model, file_log)