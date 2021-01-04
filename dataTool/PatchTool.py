#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: PatchTool.py
# Created Date: Wednesday September 25th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 12th November 2019 5:53:44 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################




import torch


class PatchToolClass:
    def __init__(self,
                microPatchSize,
                macroPatchSize,
                imgSize,
                device,
                microStride = None,
                macroStride = None):
        self.microPatchSize = microPatchSize
        self.macroPatchSize = macroPatchSize
        self.imgSize        = imgSize
        self.microStride    = microStride
        self.macroStride    = macroStride
        self.slidNum        = self.imgSize//self.microPatchSize
        self.batchNum       = self.slidNum ** 2
        self.device         = device
        if microPatchSize == None:
            self.microStride = self.microPatchSize
    # def toMicroPatch(self,originalImg):
    #     res = torch.tensor(self.batchNum*originalImg.size(0),3,)
    #     for i in range(self.slidNum):
    def splicePatch(self,imgPatchBatch):
        for i in range(self.slidNum):
            temp = imgPatchBatch[:,i*self.slidNum,:,:,:]
            for j in range(1,self.slidNum):
                temp = torch.cat([temp,imgPatchBatch[:,i*self.slidNum + j,:,:,:]],-1)
            if i ==0:
                res = temp
            else:
                res = torch.cat([res,temp],-2)
        return res  # Batch channel h w
    
    def splitToPatch(self,imgBatch):
        tempTensor = torch.zeros(imgBatch.size()[0]*self.batchNum,imgBatch.size()[1],self.microPatchSize,self.microPatchSize).to(self.device)
        for t in range(imgBatch.size()[0]):
            for i in range(self.slidNum):
                for j in range(self.slidNum):
                    tempTensor[t*self.batchNum+i*self.slidNum+j,:,:,:] = imgBatch[t, :,i*self.microPatchSize:(i+1)*self.microPatchSize,j*self.microPatchSize:(j+1)*self.microPatchSize]
        res = tempTensor
        return res

    def splitLabels(self,labels):
        bs         = labels.size()[0]
        tempTensor = torch.zeros(bs*self.batchNum,labels.size()[1]).to(self.device)
        for i in range(bs):
            tempTensor[i*bs:(i+1)*bs,:]=labels[i,:]
        return tempTensor
    # def spliceMacroPath(self,imgPatchBatch):
    #     for i in range(self.slidNum):
    #         temp = imgPatchBatch[:,i*self.slidNum,:,:,:]
    #         for j in range(1,self.slidNum):
    #             temp = torch.cat([temp,imgPatchBatch[:,i*self.slidNum + j,:,:,:]],-1)
    #         if i ==0:
    #             res = temp
    #         else:
    #             res = torch.cat([res,temp],-2)
    #     return res