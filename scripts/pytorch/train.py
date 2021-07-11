import os
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

dir_current = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_current)
sys.path.append('../')
sys.path.append('../../')

import network, loader
import config
from utils import parser, plots

os.chdir(dir_current)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
def train(net, device, loader, optimizer, loss_fnc):
    #initialise counters
    running_loss = 0.0
    loss = []
    net.train()

    start_time = time.time()
    for data in loader:
        x = data[:, :3, :, :]
        gt = data[:, 3, :, :]
        mask = data[:, 4, :, :]

        optimizer.zero_grad()

        x = x.to(device)
        ''' Output '''
        out = net(x)
        out = out.permute(1, 0, 2, 3)
        pred = out[0]

        mask = mask.to(device)
        gt = gt.to(device)

        loss = loss_fnc(pred, gt)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    #end training 
    end_time = time.time()
    running_loss /= len(loader)

    if verbose != 0:
        if verbose == 1:
            line_end = '\n'
        elif verbose == 2:
            line_end = ' '*8 + '\r'
        print(' Train Loss : {:05f} -- Time: {:05f} s'.format(running_loss, end_time - start_time), end=line_end)
    torch.save(net.state_dict(), file_model_final)
    return running_loss
''' Validation '''
def validate(net, device, loader, optimizer, loss_fnc):
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

            loss = loss_fnc(pred, gt)
            running_loss += loss.item()
    #end training 
    end_time = time.time()
    running_loss /= len(loader)

    if verbose != 0:
        if verbose == 1:
            line_end = '\n'
        elif verbose == 2:
            line_end = ' '*8 + '\r'
        print(' '*24, 'Val Loss : {:05f} -- Time: {:05f} s'.format(running_loss, end_time - start_time), end=line_end)
    return running_loss
################################## RUN ##################################
os.makedirs(dir_model, exist_ok=True)

''' Check CUDA '''
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
''' Model '''
net = network.BuildUnet(num_ch, rate_drop).float()
net = net.to(device)
loss_fnc = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
''' Load Weight '''
if is_retrain:
    net.load_state_dict(torch.load(file_model_final))
elif is_finetune:
    net.load_state_dict(torch.load(file_model_best))
''' Training'''
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

print('Training...')
for epoch in range(0, end_epoch):
    print("Epoch: {:4d}/{:4d}".format(epoch + 1, end_epoch), end=' ---- ')

    train_loss = train(net, device, trainloader, optimizer, loss_fnc)
    val_loss = validate(net, device, valloader, optimizer, loss_fnc)

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
print('Training end.')

''' Plot loss graph '''
plots.plot_graph(dir_current, dir_model, file_log)