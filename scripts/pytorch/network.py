import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

import config


class EncodeBlock(nn.Module):
    """
    encode block
    """
    def __init__(self, ch, dropout):
        super(EncodeBlock, self).__init__()

        self.bn_1 = nn.BatchNorm2d(ch)
        self.drop_1 = nn.Dropout2d(dropout)
        self.conv_1 = nn.Conv2d(ch, ch, (3, 3), padding=(1, 1))

        self.bn_2 = nn.BatchNorm2d(ch)
        self.drop_2 = nn.Dropout2d(dropout)
        self.conv_2 = nn.Conv2d(ch, ch, (3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.bn_1(x)
        x = self.drop_1(x)
        x = self.conv_1(x)
        x = nn.Tanh(x)
        
        x = self.bn_2(x)
        x = self.drop_2(x)
        x = self.conv_2(x)
        x = nn.Tanh(x)

        return x

class DecodeBlock(nn.Module):
    """
    decode block
    """
    def __init__(self, in_ch, ch, dropout):
        super(DecodeBlock, self).__init__()

        self.conv_0 = nn.ConvTranspose2d(in_ch, ch, (3, 3), padding=(1, 1))
        self.up_0 = nn.Upsample((2, 2))

        self.bn_1 = nn.BatchNorm2d(ch)
        self.drop_1 = nn.Dropout2d(dropout)
        self.conv_1 = nn.ConvTranspose2d(ch, ch, (3, 3), padding=(1, 1))

        self.bn_2 = nn.BatchNorm2d(ch)
        self.drop_2 = nn.Dropout2d(dropout)
        self.conv_2 = nn.ConvTranspose2d(ch, ch, (3, 3), padding=(1, 1))

    def forward(self, x, shortcut):
        x = self.conv_0(x)
        x = self.up_0(x)
        x = torch.cat((x, shortcut), 0)

        x = self.bn_1(x)
        x = self.drop_1(x)
        x = self.conv_1(x)
        x = nn.Tanh(x)
        
        x = self.bn_2(x)
        x = self.drop_2(x)
        x = self.conv_2(x)
        x = nn.Tanh(x)

        return x
    
class BuildUnet(nn.Module):
    ''' Build U-net model
    Args:
        num_ch : input channel num
        rate_dropout : rate of dropout
    '''

    def __init__(
        self, 
        num_ch=3,
        rate_dropout=0.2,
    ):
        super(BuildUnet, self).__init__()

        self.conv_0 = nn.Conv2d(num_ch, 8, (3, 3), padding=(1, 1))
        self.en_0 = EncodeBlock(16, rate_dropout)

        self.pool_1 = nn.AvgPool2d((2, 2))
        self.en_1 = EncodeBlock(32, rate_dropout)

        self.pool_2 = nn.AvgPool2d((2, 2))
        self.en_2 = EncodeBlock(64, rate_dropout)

        self.pool_3 = nn.AvgPool2d((2, 2))
        self.en_3 = EncodeBlock(128, rate_dropout)

        self.de_2 = DecodeBlock(128, 64, rate_dropout)
        self.de_1 = DecodeBlock(64, 32, rate_dropout)
        self.de_0 = DecodeBlock(32, 16, rate_dropout)

        self.out = nn.ConvTranspose2d(16, 1, (1, 1), padding=(1, 1))

    def forward(self, x):
        e0 = self.conv_0(x)
        e0 = self.en_0(e0)

        e1 = self.pool_1(e0)
        e1 = self.en_1(e1)
        
        e2 = self.pool_1(e1)
        e2 = self.en_1(e2)

        e3 = self.pool_1(e2)
        e3 = self.en_1(e3)

        d2 = self.de_2(e3, e2)
        d1 = self.de_2(d2, e1)
        d0 = self.de_2(d1, e0)

        output = self.out(d0)

        return output