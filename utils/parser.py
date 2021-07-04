import argparse

''' Main parser '''
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test', help='model name')
parser.add_argument('--num', type=int, default=10, help='data num')

''' Training parser '''
parser_train = parser
parser_train.add_argument('--epoch', type=int, default=100, help='epoch num')
parser_train.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser_train.add_argument('--drop', type=float, default=0.2, help='dropout rate')
parser_train.add_argument('--batch', type=int, default=8, help='batch size')
parser_train.add_argument('--val', type=float, default=0.3, help='validation data rate')
parser_train.add_argument('--verbose', type=int, default=1, help='progress bar')
parser_train.add_argument('--retrain', action='store_true', help='add to re-train')
parser_train.add_argument('--finetune', action='store_true', help='add to fine-tune on real data')
parser_train.add_argument('--generator', action='store_true', help='add to use generator')
# parser_train.add_argument('--load', action='store_true', help='add to load data on memory, not to use generator')
# parser_train.add_argument('--aug', action='store_true', help='add to data augmentation')

''' Test parser '''
parser_test = parser
parser_test.add_argument('--data', default='synthetic', help='data type. "synthetic" or "real"')
parser_train.add_argument('--ply', action='store_true', help='add to save ply file')
parser_train.add_argument('--bmp', action='store_true', help='add to save depth image file (.bmp)')
