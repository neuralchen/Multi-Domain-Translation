#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: batch_text.py
# Created Date: Thursday September 26th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 30th September 2019 7:15:08 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################



import torch

TextColorTable = {
    'red':[255.0,0.0,0.0],
    'blue':[0.0,0.0,255.0],
    'green':[0.0,255.0,0.0],
    'black':[0.0,0.0,0.0]
}
for key in TextColorTable:
       TextColorTable[key]=[(x/255.0 - 0.5)/0.5 for x in TextColorTable[key]]

class BatchTensorClass:
    def __init__(self,textSize = 15,textMargin=[10,10], color='red', backgroundColor='black'):
        """
        
        """
        textColor       = TextColorTable[color]
        backgroundColor = TextColorTable[backgroundColor]
        self.textSize   = textSize
        self.textMargin = textMargin
        textAreaPadding = 3
        # self.location   = location
        textThick       = int(textSize/3)
        self.textAreaSize   = 2*textAreaPadding+textSize
        self.positiveMark   = torch.zeros(3,self.textAreaSize,self.textAreaSize)
        self.negativeMark   = torch.zeros(3,self.textAreaSize,self.textAreaSize)
        
        for i in range(3):
            self.positiveMark[i,:,:] = backgroundColor[i]
            self.negativeMark[i,:,:] = backgroundColor[i]
            self.positiveMark[i,(textAreaPadding+textThick):(textAreaPadding+textThick*2),
                            textAreaPadding:(textAreaPadding+textSize)] = textColor[i]
            self.positiveMark[i,textAreaPadding:(textAreaPadding+textSize),
                            (textAreaPadding+textThick):(textAreaPadding+textThick*2)] = textColor[i]
            self.negativeMark[i,(textAreaPadding+textThick):(textAreaPadding+textThick*2),
                            textAreaPadding:(textAreaPadding+textSize)] = textColor[i]
        # if self.location == 'LU':
        #     self.positionSign = [1,1,1,1]
        # elif self.location == 'LB':
        #     self.positionSign = [-1,-1,1,1]
        # elif self.location == 'RU':
        #     self.positionSign = [1,1,-1,-1]
        # else:
        #     self.positionSign = [-1,-1,-1,-1]



    def BatchText(self,batchTensor,labels,textMargin=None):
        '''
        batchTensor must have shape B,C,W,H, value range:[-1.0,1.0]
        '''
        if textMargin is None:
            textMargin = self.textMargin  
        for i in range(batchTensor.shape[0]):
            
            if labels[i] == 1:
                batchTensor[i,:,textMargin[0]:(self.textAreaSize+textMargin[0]),
                    textMargin[1]:(self.textAreaSize+textMargin[1])] = self.positiveMark
            else:
                batchTensor[i,:,textMargin[0]:(self.textAreaSize+textMargin[0]),
                    textMargin[1]:(self.textAreaSize+textMargin[1])] = self.negativeMark
        return batchTensor