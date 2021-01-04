#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: ActivatorTable.py
# Created Date: Thursday October 17th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 2nd November 2019 8:33:42 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

import torch.nn as nn

def getActivator(actName):
    actName = actName.lower()
    if actName   == "relu":
        return nn.ReLU()
    elif actName == "selu":
        return nn.SELU()
    elif actName == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif actName == "hardtanh":
        return nn.Hardtanh()
    elif actName == "tanh":
        return nn.Tanh()
    