#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: DeCov.py
# Created Date: Tuesday September 24th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 4th March 2020 11:12:20 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################




import torch
from torch import nn

class DeCov(nn.Module):
    def __init__(self,in_channels,out_channels,factor=2,kernel_size=3,stride=1,bias=False):
        super(DeCov, self).__init__()
        padding_size=(kernel_size-1)//2
        self.dconv=nn.Sequential(
            nn.Upsample(scale_factor=factor),
            nn.ReflectionPad2d((padding_size)),
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,bias=bias)
        )
        self.bias = bias
        self.__initialize__()
    def __initialize__(self):
        for layer in self.dconv:
            if isinstance(layer,nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if self.bias:
                    nn.init.zeros_(layer.bias)

            
    def forward(self,x):
        y=self.dconv(x)
        return y