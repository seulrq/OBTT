import torch
import torch.nn as nn
import timm.models.vision_transformer
from timm.models.vision_transformer import Block, DropPath, Mlp

import skimage.filters.rank as sfr
from skimage.morphology import disk
import numpy as np
from functools import partial
import torch.optim as optim
import random

# LSTM for Series
class TextModel(nn.Module):
    def __init__(self, config, n_features, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, config.num_labels)  # output_size classes

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM层
        x = x[:, -1, :]  # 只取LSTM输出中的最后一个时间步
        x = self.fc(x)  # 通过一个全连接层
        return x
# Vit for img
class PatchEmbed(nn.Module):
    """ MTR matrix to Patch Embedding
    """
    def __init__(self, img_size=40, patch_size=2, in_chans=1, embed_dim=192):
        super().__init__()
        img_size = (int(img_size / 5), img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TrafficTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(TrafficTransformer, self).__init__(**kwargs)

        self.patch_embed = PatchEmbed(img_size=kwargs['img_size'], patch_size=kwargs['patch_size'],
                                         in_chans=kwargs['in_chans'], embed_dim=kwargs['embed_dim'])

        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)
        del self.norm  # remove the original norm

    def forward_packet_features(self, x, i):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        cls_pos = self.pos_embed[:, :1, :]
        packet_pos = self.pos_embed[:, i*80+1:i*80+81, :]
        pos_all = torch.cat((cls_pos, packet_pos), dim=1)
        x = x + pos_all
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        cls = x[:, :1, :]

        x = x[:, 1:, :]
        x = x.reshape(B, 4, 20, -1).mean(axis=1)
        x = torch.cat((cls, x), dim=1)

        self.fc_norm(x)

        return x

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, 5, -1)
        for i in range(5):
            packet_x = x[:, :, i, :]
            packet_x = packet_x.reshape(B, C, -1, 40)
            packet_x = self.forward_packet_features(packet_x, i)
            if i == 0:
                new_x = packet_x
            else:
                new_x = torch.cat((new_x, packet_x), dim=1)
        x = new_x

        for blk in self.blocks:
            x = blk(x)

        x = x.reshape(B, 5, 21, -1)[:, :, 0, :]
        x = x.mean(dim=1)

        outcome = self.fc_norm(x)
        return outcome

# Combined Model
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # text
        self.text_model = TextModel(config,4)
        self.text_model.load_state_dict(torch.load('./lstm_model.pth'))
        # image
        self.img_model = TrafficTransformer(
        img_size=40, patch_size=2, in_chans=1, embed_dim=192, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),num_classes=config.num_labels,drop_path_rate=0.1)
        self.img_model.load_state_dict(torch.load('/home/zhaihaonan/YaTC-main/output_dir/pretrained-model.pth'), strict=False)
        self.img_model.load_state_dict(torch.load('/home/zhaihaonan/YaTC-main/Vit_model.pth'))
        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.num_labels * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, texts, imgs, labels=None):
        text_feature = self.text_model(texts)

        img_feature = self.img_model(imgs)

        prob_vec = self.classifier(
            torch.cat([text_feature, img_feature], dim=1)
        )
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels