#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: GenResBlockConditional.py
# Created Date: Thursday September 26th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 4th November 2019 10:17:53 pm
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
        # self.infer    = nn.ModuleList()
        self.infer1 = BC(in_ch, num_classes)
        self.infer2 = nn.Sequential(
                        activation,
                        nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
                        )
        self.infer3 = BC(h_ch, num_classes)
        self.infer4 = nn.Sequential(
                        activation,
                        nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
                            )

    def forward(self, x, y):
        return x + self.residual(x, y)

    def residual(self, x, y):
        
        out = self.infer1(x,y)
        out = self.infer2(out)
        out = self.infer3(out,y)
        out = self.infer4(out)
        return out