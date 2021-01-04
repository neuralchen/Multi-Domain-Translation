#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: LoadSpecifiedImagesCelebAforNoDiff.py
# Created Date: Saturday October 5th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 3rd November 2019 8:34:25 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import os

class LoadSpecifiedImagesCelebA:
    """Dataset class for the specified images in CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, crop_size, image_size, specifiedList,toPatch = False,microPatchSize=32):
        """Initialize and preprocess the CelebA dataset."""
        transform = []
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.specifiedList = specifiedList
        self.toPatch  = toPatch
        self.microPatchSize = microPatchSize
        self.imgSize        = image_size
        self.slidNum        = self.imgSize//self.microPatchSize
        self.batchNum       = self.slidNum ** 2
        self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for item in self.specifiedList:
            fileindex, _ = os.path.splitext(item)
            fileindex    = int(fileindex)-1
            split        = lines[fileindex].split()
            
            values = split[1:]
            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            filename  = os.path.join(self.image_dir, item)
            self.dataset.append([filename, label])

        print('Finished preprocessing the specified images in CelebA dataset...')

    def getSamples(self):
        """Return an batch images and its corresponding attribute labels."""
        images  = [] 
        labels  = []
        for t in range(self.num_images):
            image = Image.open(self.dataset[t][0])
            image = self.transform(image)
            if self.toPatch:
                tempTensor = torch.zeros(self.batchNum,3,self.microPatchSize,self.microPatchSize)
                for i in range(self.slidNum):
                    for j in range(self.slidNum):
                        tempTensor[i*self.slidNum+j,:,:,:] = image[:,i*self.microPatchSize:(i+1)*self.microPatchSize,j*self.microPatchSize:(j+1)*self.microPatchSize]
                res   = tempTensor
                label = torch.FloatTensor([self.dataset[t][1]]*self.batchNum)
            else:
                res   = image
                label = torch.FloatTensor(self.dataset[t][1])

            if t == 0:
                images = res.unsqueeze(0)
                labels = label.unsqueeze(0)
            else:
                images = torch.cat([images,res.unsqueeze(0)],0)
                labels = torch.cat([labels,label.unsqueeze(0)],0)
        return images, labels


    def __len__(self):
        """Return the number of images."""
        return self.num_images

def getNonConflictingImgsLablels(imagePath,
                                    imSize,
                                    imCropSize,
                                    attributesPath,
                                    selectedAttrs,
                                    specifiedImages,
                                    LoadPatchData,
                                    MicroPatchSize
                                ):
                                
    imagesNames         = specifiedImages
    SampleImgNum        = len(imagesNames)
    n_classes           = len(selectedAttrs)
    fixedSample         = LoadSpecifiedImagesCelebA(imagePath, 
                                    attributesPath, selectedAttrs,
                                    imCropSize,imSize,imagesNames,
                                    LoadPatchData,MicroPatchSize)
                                    
    fixedImages, fixedLabel = fixedSample.getSamples()
    fixeDelta = torch.zeros(n_classes+1,SampleImgNum,n_classes)
    for i in range(n_classes):
        fixeDelta[i,:,:] = fixedLabel
    # index 0 for reconstruction
    for index in range(1,n_classes):
        for index1 in range(SampleImgNum):
            fixeDelta[index,index1,index] = (fixeDelta[index,index1,index]+1)%2
            if fixedLabel[index1,index] == 0:
                if index == 0:
                    fixeDelta[index,index1,1] = 0 # set to non bangs
                    fixeDelta[index,index1,2] = 0 # set to non black hair
                    fixeDelta[index,index1,3] = 0 # set to non blond hair
                    fixeDelta[index,index1,4] = 0 # set to non brown hair
                elif index == 1:
                    #bangs
                    if fixedLabel[index1,0] == 0:
                        # if bald
                        fixeDelta[index,index1,0] = 0 # set to non bald
                        fixeDelta[index,index1,4] = 1 # set to hair color to brown
                elif index == 2:
                    # black hair
                    fixeDelta[index,index1,3] = 0 # set to non blond hair
                    fixeDelta[index,index1,4] = 0 # set to non brown hair
                elif index == 3:
                    # blond hair
                    fixeDelta[index,index1,2] = 0 # set to non black hair
                    fixeDelta[index,index1,4] = 0 # set to non brown hair
                elif index == 4:
                    fixeDelta[index,index1,2] = 0 # set to non black hair
                    fixeDelta[index,index1,3] = 0 # set to non blond hair
            else:
                # fixeDelta[index,index1,index] = 0
                if index == 0:
                    # bald
                    fixeDelta[index,index1,4] = 1 # set to hair color to brown
                if index == 2:
                    fixeDelta[index,index1,3] = 1 # translate black hair to blond hair
                elif index == 3:
                    fixeDelta[index,index1,4] = 1 # translate blond hair to Brown Hair
                elif index == 4:
                    fixeDelta[index,index1,2] = 1 # translate Brown hair to black Hair 
    return fixedImages,fixeDelta