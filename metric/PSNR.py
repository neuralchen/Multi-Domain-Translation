#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: PSNR.py
# Created Date: Friday October 11th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 12th November 2019 1:53:08 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import os

PIXEL_MAX = 2.0
lg255       = 2*torch.log10(torch.tensor(255.0))

def PSNR(x,y):
    x = (x + 1) / 2
    x = x.clamp_(0, 1) * 255
    x = x.byte()
    x = x.float()
    y = (y+1)/2
    y = y.clamp_(0, 1) * 255
    y = y.byte()
    y = y.float()
    mse     = ((x-y).pow(2).mean(dim=(1,2,3)))
    psnr    = 10 * (lg255 - torch.log10(mse))
    meanpsnr= psnr.mean()
    psnr    = psnr.numpy().tolist()
    return psnr,meanpsnr.item()

if __name__ == "__main__":
    transform = []
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    for _, _, files in os.walk("F:\\celeba-hq\\celeba-hq\\celeba-1024"):
        totalFiles=files
    for i  in range(20):
        filename1 = "F:\\celeba-hq\\celeba-hq\\celeba-1024\\"+totalFiles[i]
        filename2 = "F:\\celeba-hq\\celeba-hq\\celeba-512\\"+totalFiles[i]
        image1  = Image.open(filename1)
        image1  = transform(image1)

        
        
        image2  = Image.open(filename2)
        image2  = transform(image2)
        image2  = image2.unsqueeze(0)
        image2  = F.interpolate(image2,size=1024)
        if i == 0:
            res1    = image1.unsqueeze(0)
            res2    = image2
        else:
            res1    = torch.cat([res1,image1.unsqueeze(0)],0)
            res2    = torch.cat([res2,image2],0)
    out,mean   = PSNR(res1,res2)
    print(out)
    print(mean)
    