#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: TVLoss.py
# Created Date: Tuesday October 8th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 8th October 2019 8:49:05 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self,TVLossWeight=1,imageSize=512,withMask=False,patchSize=128,seamWide=4):
        super(TVLoss,self).__init__()
        self.TVLossWeight   = TVLossWeight
        self.imageSize      = imageSize
        slideNum            = imageSize//patchSize
        self.withMask       = withMask
        if withMask:
            self.mask           = nn.Parameter(torch.zeros(3,imageSize,imageSize),requires_grad=False)
            for i in range(1,slideNum-1):
                index = i * slideNum
                self.mask[:,(index-seamWide):(index+seamWide),:] = 1.0
                self.mask[:,:,(index-seamWide):(index+seamWide)] = 1.0

    def forward(self,x):
        batchSize = x.size()[0]
        if self.withMask:
            x    = x*self.mask
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:self.imageSize-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:self.imageSize-1]),2).sum()
        return self.TVLossWeight*2*(h_tv/self.imageSize+w_tv/self.imageSize)/batchSize