#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: CelebAHQ.py
# Created Date: Saturday October 5th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 10th November 2019 7:00:11 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################



from torch.utils import data
from PIL import Image
import torch
import os
import random

class CelebAHQ(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, 
                    transform, mode, batchSize, imageSize, toPatch = False, microPatchSize=32):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir      = image_dir
        self.attr_path      = attr_path
        self.selected_attrs = selected_attrs
        self.transform      = transform
        self.mode           = mode
        self.train_dataset  = []
        self.test_dataset   = []
        self.attr2idx       = {}
        self.idx2attr       = {}
        self.toPatch        = toPatch
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
        for _, _, files in os.walk(self.image_dir):
            totalFiles=files
            totalFiles=sorted(totalFiles)

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        # random.seed(1234)
        # random.shuffle(lines)
        for i,name in enumerate(totalFiles):
            fileindex, _= os.path.splitext(name)
            
            fileindex   = int(fileindex)-1
            split       = lines[fileindex].split()
            values      = split[1:]
            label       = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                filename = os.path.join(self.image_dir, name)
                self.test_dataset.append([filename, label])
            else:
                filename = os.path.join(self.image_dir, name)
                self.train_dataset.append([filename, label])

        # for i, line in enumerate(lines):
        #     split = line.split()
        #     filename = split[0]
        #     print(i)
        #     if filename in totalFiles:

        #         values = split[1:]
        #         label = []
        #         for attr_name in self.selected_attrs:
        #             idx = self.attr2idx[attr_name]
        #             label.append(values[idx] == '1')

        #         if (i+1) < 2000:
        #             filename = os.path.join(self.image_dir, filename)
        #             self.test_dataset.append([filename, label])
        #         else:
        #             filename = os.path.join(self.image_dir, filename)
        #             self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        # image = Image.open(os.path.join(self.image_dir, filename))
        image = Image.open(filename)
        res   = self.transform(image)
        if self.toPatch:
            tempTensor = torch.zeros(self.batchNum,3,self.microPatchSize,self.microPatchSize)
            for i in range(self.slidNum):
                for j in range(self.slidNum):
                    tempTensor[i*self.slidNum+j,:,:,:] = res[:,i*self.microPatchSize:(i+1)*self.microPatchSize,j*self.microPatchSize:(j+1)*self.microPatchSize]
            res = tempTensor
            labels = torch.FloatTensor([label]*self.batchNum)
        else:
            labels = torch.FloatTensor(label)

        return res, labels

    def __len__(self):
        """Return the number of images."""
        return self.num_images