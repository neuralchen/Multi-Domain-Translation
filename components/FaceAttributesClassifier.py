#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: FaceAttributesClassifier.py
# Created Date: Tuesday September 24th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 28th September 2019 12:58:06 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################



import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils

class FaceAttributesClassifier(nn.Module):
    def __init__(self, conv_dim=48, image_size=384, attr_dim=13, fc_dim=512, n_layers=5):
        super(FaceAttributesClassifier, self).__init__()
        self.conv = nn.ModuleList()
        in_channels = 3
        for i in range(n_layers):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim * 2 ** i, 4, 2, 1,bias=False),
                nn.BatchNorm2d(conv_dim * 2 ** i, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = conv_dim * 2 ** i

        # for conv_ in self.conv:
        #     for layer in conv_:
        #         if isinstance(layer,nn.Conv2d):
        #             nn.init.xavier_uniform_(layer.weight)
        feature_size = image_size // 2**n_layers
        self.fc_att = nn.Sequential(
            nn.Linear(conv_dim * 2 ** (n_layers - 1) * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, attr_dim),#,
            nn.Sigmoid()
        )
        # for layer in self.fc_att:
        #     if isinstance(layer,nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.zeros_(layer.bias)
        # for layer in self.fc_adv:
        #     if isinstance(layer,nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.zeros_(layer.bias)

    def forward(self, x):
        y=x
        for layer in self.conv:
            y=layer(y)
        y = y.view(y.size()[0], -1)
        logit_att = self.fc_att(y)
        return logit_att