import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import os
import PIL

from torchvision import datasets, transforms

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed

import models_YaTC

def build_dataset(path):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Resize([80,80]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dataset = datasets.ImageFolder(path, transform=transform)
    return dataset
batch_size = 32
device = torch.device('cuda:0')
data = build_dataset('/home/zhaihaonan/0929_user_png')
mydataloader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        drop_last=False
    )

model = models_YaTC.__dict__['TraFormer_YaTC'](
        img_size = 80,
        num_classes=5,
        drop_path_rate=0.1
    )

model.to(device)
model.load_state_dict(torch.load('./Vit_model.pth'))
with torch.no_grad():
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    output_arr = []
    output_label = []
    for batch in mydataloader:
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        output = model(images)
        output_arr.extend(output.tolist())
        output_label.label(target.tolist())

with open('output.txt','w',encoding='utf-8') as f:
    for i in output_arr:
        f.write(str(i)+'\n')
with open('label.txt','w',encoding='utf-8') as f:
    for i in output_label:
        f.write(str(i)+'\n')
