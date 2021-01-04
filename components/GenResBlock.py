#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: GenResBlock.py
# Created Date: Thursday September 26th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 17th October 2019 1:31:55 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# from components.CBN   import CBN
from components.BC import BC
class GenResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, num_classes=0,activation=F.relu):
        super(GenResBlock, self).__init__()

        self.activation     = activation
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes
        # Register layrs
        # self.b1 = nn.BatchNorm2d(in_ch)
        # self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        # self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        # self.b2 = nn.BatchNorm2d(h_ch)
        if num_classes==0:
            self.infer = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                activation,
                nn.Conv2d(in_ch, h_ch, ksize, 1, pad),
                nn.BatchNorm2d(h_ch),
                activation,
                nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
            )
        else:
            self.infer    = nn.ModuleList()
            self.infer.append(BC(in_ch, num_classes))
            self.infer.append(nn.Sequential(
                activation,
                nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
            ))
            self.infer.append(BC(h_ch, num_classes))
            self.infer.append(nn.Sequential(
                activation,
                nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
            ))

    def forward(self, x, y):
        return x + self.residual(x, y)

    def residual(self, x, y):
        # # h = self.b1(x, y)
        # h = self.b1(x)
        # h = self.activation(h)
        # h = self.c1(h)
        # # h = self.b2(h, y)
        # h = self.b2(h)
        # return self.c2(self.activation(h))
        # out = x
        out = self.infer(x)
        # for cell in self.infer:
        #     out = cell(out)
        return out