import argparse
import datetime
import json
import numpy as np
import os,sys
import time
import torch
import dill
import logging
from torch.optim import AdamW
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from CombineModel import Model
import argparse
from Config import config
from utils.common import data_format, read_from_file, train_val_split, save_model, write_to_file
from utils.DataProcess import Processor
from Trainer import Trainer

logging.getLogger().setLevel(logging.INFO)
batch_size = 32
with open('./dataset_train80.pkl','rb') as f:
    train_data = dill.load(f)
with open('./dataset_test80.pkl','rb') as f:
    val_data = dill.load(f)
train_num = len(train_data)
val_num = len(val_data)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle=False)

# args
parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true', help='训练模型')
parser.add_argument('--lr', default=5e-5, help='设置学习率', type=float)
parser.add_argument('--weight_decay', default=1e-3, help='设置权重衰减', type=float)
parser.add_argument('--epoch', default=30, help='设置训练轮数', type=int)
parser.add_argument('--do_test', action='store_true', help='预测测试集数据')
parser.add_argument('--load_model_path', default=None, help='已经训练好的模型路径', type=str)

args = parser.parse_args()
config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.epoch = args.epoch
config.out_hidden_size = 16
config.num_labels = 5
config.load_model_path = args.load_model_path
config.output_test_path = 'output_test.txt'

# Initilaztion
processor = Processor(config)
model = Model(config)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)
trainer = Trainer(config, processor, model, device)

# Train
def train():
    # train_loader = processor(train_data, config.train_params)
    # val_loader = processor(val_data, config.val_params)
    best_acc = 0.0
    # model.load_state_dict(torch.load('./combine_model80.pth'), strict=False)
    epoch = config.epoch
    for e in range(epoch):
        logging.info('-' * 20 + ' ' + 'Epoch ' + str(e+1) + ' ' + '-' * 20)
        tloss, tloss_list = trainer.train(train_loader)
        logging.info('Train Loss: {}'.format(tloss))
        vloss, vacc = trainer.valid(val_loader)
        logging.info('Valid Loss: {}'.format(vloss))
        logging.info('Valid Acc: {}'.format(vacc))
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), './combine_model80.pth')
            logging.info('Update best model!')

# Test
def test():
    model.load_state_dict(torch.load('./combine_model80.pth'))

    vloss, vacc = trainer.valid(val_loader)
    logging.info('Valid Loss: {}'.format(vloss))
    logging.info('Valid Acc: {}'.format(vacc))

# main
if __name__ == "__main__":
    if args.do_train:
        train()
    if args.do_test:
        test()
