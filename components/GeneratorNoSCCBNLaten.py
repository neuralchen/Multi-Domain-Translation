#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: GeneratorNoSCCBNLaten.py
# Created Date: Tuesday September 24th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 4th January 2021 10:57:21 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from components.DeCov import DeCov
from components.CBN   import CBN
from components.GenResBlock import GenResBlock
from utilities.ActivatorTable import getActivator

MAX_DIM = 64 * 16
class Generator(torch.nn.Module):
    def __init__(   
                    self, 
                    dim,
                    n_layers        = 4,
                    res_num         = 0,
                    attr_dim        = 13,
                    encActName      = "leakyrelu",
                    decActName      = "selu",
                    outActName      = "hardtanh"):
        super(Generator, self).__init__()

        self.n_layers   = n_layers - 1 
        self.res_num    = res_num
        self.dim        = dim
        self.encoder    = nn.ModuleList()
        self.lstus      = nn.ModuleList()
        
        self.decoderdconv   = nn.ModuleList()
        self.cbns       = nn.ModuleList()
        in_channels     = 3
        
        self.encAct = getActivator(encActName)
        self.decAct = getActivator(decActName)

        # append input layer
        self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels,self.dim,3,2,1,bias=False),
                nn.BatchNorm2d(self.dim,affine=True),
                self.encAct
            ))
        
        for i in range(self.n_layers):
            in_channels = min(self.dim*2**i, MAX_DIM)
            currentDim  = min(self.dim*2**(i+1), MAX_DIM)
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels,currentDim,3,2,1,bias=False),
                nn.BatchNorm2d(currentDim,affine=True),
                self.encAct
            ))
 
        if self.res_num != 0:
            self.resblocks  = nn.ModuleList()
            for i in range(res_num):
                currentDim = min(self.dim*2**(self.n_layers), MAX_DIM)
                self.resblocks.append(GenResBlock(currentDim,currentDim,num_classes=0,activation=self.decAct))
        self.cBN = CBN(currentDim,attr_dim)      
        skipRatio += 1
        for i in range(self.n_layers):
            i = self.n_layers-i
            currentInDim = min(self.dim*2**(i), MAX_DIM)
            currentOutDim = min(self.dim*2**(i-1), MAX_DIM)
            self.decoderdconv.append(DeCov(currentInDim,currentOutDim,2,3,1))
            self.cbns.append(nn.BatchNorm2d(currentOutDim,attr_dim))
        
        # insert output layer
        act = getActivator(outActName)
        if outActName == "hardtanh":
            addBias =True
        else:
            addBias = False
        self.toRGB = nn.Sequential(
                DeCov(dim,3,2,3,bias=addBias),
                act
            )

    def forward(self,x,a):
        z  = x
        for i,layer in enumerate(self.encoder):
            z=layer(z)
        out = z 
        if self.res_num != 0:
            for resblock in self.resblocks:
                out = resblock(out,a)
        
        out = self.cBN(out,a)      
        for i in range(self.n_layers):
            out = self.decoderdconv[i](out)
            out = self.cbns[i](out)
            out = self.decAct(out)
        out = self.toRGB(out)
        return out