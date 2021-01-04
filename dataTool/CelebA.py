#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: CelebA.py
# Created Date: Tuesday September 24th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 23rd November 2019 11:46:29 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


from torch.utils import data
from PIL import Image
import torch
import os
import random

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode,batchSize,imageSize, toPatch = False,microPatchSize=32):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.toPatch  = toPatch
        self.microPatchSize = microPatchSize
        self.imgSize        = imageSize
        # self.slidNum        = self.imgSize//self.microPatchSize
        # self.batchNum       = self.slidNum ** 2
        self.batchSize      = batchSize
        
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        # if self.mode == "train":
        #     random.seed(1234)
        #     random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) >= 2000 and (i+1)<4001:
                filename = os.path.join(self.image_dir, filename)
                self.test_dataset.append([filename, label])
            else:
                filename = os.path.join(self.image_dir, filename)
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        # image = Image.open(os.path.join(self.image_dir, filename))
        image = Image.open(filename)
        res   = self.transform(image)
        # if self.toPatch:
        #     tempTensor = torch.zeros(self.batchNum,3,self.microPatchSize,self.microPatchSize)
        #     for i in range(self.slidNum):
        #         for j in range(self.slidNum):
        #             tempTensor[i*self.slidNum+j,:,:,:] = res[:,i*self.microPatchSize:(i+1)*self.microPatchSize,j*self.microPatchSize:(j+1)*self.microPatchSize]
        #     res = tempTensor
        #     labels = torch.FloatTensor([label]*self.batchNum)
        # else:
        labels = torch.FloatTensor(label)

        return res, labels

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

if __name__ == "__main__":
    from torchvision.utils import save_image
    selected_attrs  = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                            'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    n_classes       = len(selected_attrs)
    datapath        = "F:\\DataSet\\CelebA\\Img\\img_align_celeba\\img_align_celeba"
    attpath         = "F:\\DataSet\\CelebA\\Anno\\list_attr_celeba.txt"
    imsize          = 128
    MicroPathSize   = 32
    imCropSize      = 178
    ToPatch         = True
    imagesNames = ['000019.jpg','000025.jpg','000032.jpg','000041.jpg','000052.jpg','000068.jpg','000071.jpg', '000077.jpg','000079.jpg',
                    # '000150.jpg','000173.jpg','000176.jpg','000204.jpg','000208.jpg','000228.jpg','000248.jpg','000252.jpg','000254.jpg','000256.jpg',
                    # '000272.jpg','000275.jpg','000287.jpg','000300.jpg','000302.jpg','000316.jpg','000335.jpg','000352.jpg','000383.jpg','000389.jpg','000392.jpg','000495.jpg','000537.jpg','000573.jpg','000667.jpg','000676.jpg','000687.jpg',
                    '182001.jpg','182002.jpg','182003.jpg','182004.jpg','182005.jpg','182006.jpg', '182007.jpg','182008.jpg','182009.jpg',
                    '182010.jpg','182011.jpg','182012.jpg', '182013.jpg','182014.jpg','182015.jpg','182016.jpg']
    SampleImgNum = len(imagesNames)


    slidNum         = imsize//MicroPathSize
    batchNum        = slidNum ** 2
    fixedSample     = LoadSpecifiedImagesCelebA(datapath, attpath, selected_attrs,imCropSize,imsize,imagesNames,ToPatch,MicroPathSize)
    fixedImages, fixedLabel = fixedSample.getSamples()
    fixeDelta       = torch.zeros(n_classes,SampleImgNum,batchNum,n_classes)
    for index in range(n_classes):
        for index1 in range(SampleImgNum):
            if fixedLabel[index1,0,index] == 0:
                fixeDelta[index,index1,:,index] = 1
                if index == 0 and fixedLabel[index1,0,1]==1: # from non-blad to blad, so we need to remove bangs in the same time
                    fixeDelta[index,index1,:,1] = -1
                if index == 2:
                    if fixedLabel[index1,0,3] == 1:
                        fixeDelta[index,index1,:,3] = -1
                    if fixedLabel[index1,0,4] == 1:
                        fixeDelta[index,index1,:,4] = -1
                if index == 3:
                    if fixedLabel[index1,0,2] == 1:
                        fixeDelta[index,index1,:,2] = -1
                    if fixedLabel[index1,0,4] == 1:
                        fixeDelta[index,index1,:,4] = -1
                if index == 4:
                    if fixedLabel[index1,0,2] == 1:
                        fixeDelta[index,index1,:,2] = -1
                    if fixedLabel[index1,0,3] == 1:
                        fixeDelta[index,index1,:,3] = -1
            else:
                if index == 2:
                    fixeDelta[index,index1,:,3] = 1 # translate black hair to blond hair
                if index == 3:
                    fixeDelta[index,index1,:,4] = 1 # translate blond hair to Brown Hair 
                if index == 4:
                    fixeDelta[index,index1,:,3] = 1
                    
                fixeDelta[index,index1,index] = -1
    pass
    # celeba_loader = getLoader(datapath, attpath, selected_attrs,178, 128, 4,'CelebA', 'train', 8,True,32)
    # wocao = iter(celeba_loader)
    # for t in range(100000):
    #     try:
    #         image,label = next(wocao)
    #     except:
    #         wocao = iter(celeba_loader)
    #         image,label = next(wocao)
    #     print(image.size())
    #     if image.size(1) != 16:
    #         print("error")
    #         break
    #     print(label.size())
        # print(label.view(-1,4))
        # a = image.view(4,3,128,128)
        # for i in range(4):
        #     temp = image[:,i*4,:,:,:]
        #     for j in range(1,4):
        #         temp = torch.cat([temp,image[:,i*4 + j,:,:,:]],-1)
        #     if i ==0:
        #         res = temp
        #     else:
        #         res = torch.cat([res,temp],-2)

        # save_image(denorm(res), "./%d-patch.jpg"%t, nrow=2, padding=2)
    # imagesNames = ['000019.jpg','000025.jpg','000032.jpg','000041.jpg','000052.jpg','000068.jpg','000071.jpg', '000077.jpg','000079.jpg',
    #                 # '000150.jpg','000173.jpg','000176.jpg','000204.jpg','000208.jpg','000228.jpg','000248.jpg','000252.jpg','000254.jpg','000256.jpg',
    #                 # '000272.jpg','000275.jpg','000287.jpg','000300.jpg','000302.jpg','000316.jpg','000335.jpg','000352.jpg','000383.jpg','000389.jpg','000392.jpg','000495.jpg','000537.jpg','000573.jpg','000667.jpg','000676.jpg','000687.jpg',
    #                 '182001.jpg','182002.jpg','182003.jpg','182004.jpg','182005.jpg','182006.jpg', '182007.jpg','182008.jpg','182009.jpg',
    #                 '182010.jpg','182011.jpg','182012.jpg', '182013.jpg','182014.jpg','182015.jpg','182016.jpg']
    # fixedSample = LoadSpecifiedImagesCelebA(datapath, attpath, selected_attrs,178,128,imagesNames,True,32)
    # fixedImages, fixedLabel = fixedSample.getSamples()
    # for i in range(4):
    #     temp = fixedImages[:,i*4,:,:,:]
    #     for j in range(1,4):
    #         temp = torch.cat([temp,fixedImages[:,i*4 + j,:,:,:]],-1)
    #     if i ==0:
    #         res = temp
    #     else:
    #         res = torch.cat([res,temp],-2)

    # save_image(denorm(res), "./%d-patch.jpg"%1, nrow=2, padding=2)