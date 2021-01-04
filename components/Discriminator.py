#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Discriminator.py
# Created Date: Tuesday September 24th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 4th January 2021 10:58:04 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################



import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.ActivatorTable import getActivator

MAX_DIM = 64 * 16
class Discriminator(nn.Module):
    def __init__(
                    self, 
                    dim=32, 
                    attr_dim=13,
                    n_layers=5,
                    actName='leakyrelu',
                    fc_dim=1024,
                    imageSize= 128):
        super(Discriminator, self).__init__()

        Act     = getActivator(actName)
        self.conv = nn.ModuleList()
        
        in_channels = 3
        for i in range(n_layers):
            currentDim = min(dim*2**i, MAX_DIM)
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, currentDim, 4, 2, 1,bias=False),
                nn.InstanceNorm2d(currentDim, affine=True, track_running_stats=True),
                Act
            ))
            in_channels = currentDim

        feature_size = imageSize // 2**n_layers
        self.fc_adv = nn.Sequential(
            nn.Linear(currentDim * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, 1)
        )
        self.fc_att = nn.Sequential(
            nn.Linear(currentDim * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, attr_dim)
        )

    def forward(self, x):
        y=x
        for layer in self.conv:
            y=layer(y)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        logit_att = self.fc_att(y)
        return logit_adv, logit_att