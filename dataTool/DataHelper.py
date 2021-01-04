#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: DataHelper.py
# Created Date: Saturday October 5th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 19th February 2020 4:31:48 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

from torch.utils import data
from torchvision import transforms as T
import PIL

if __name__ == "__main__":
    from CelebA import CelebA
    from CelebAHQ import CelebAHQ
else:
    from dataTool.CelebA import CelebA
    from dataTool.CelebAHQ import CelebAHQ
from torchvision.datasets import ImageFolder


def getLoader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
            batch_size=16, dataset='CelebA', mode='train', num_workers=8,toPatch=False,microPatchSize=0):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size,interpolation=PIL.Image.BICUBIC))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset.lower() == 'celeba':
        dataset = CelebA(image_dir, attr_path, selected_attrs, 
            transform, mode,batch_size,image_size,toPatch,microPatchSize)
    elif dataset.lower() == 'celeba384':
        dataset = CelebA(image_dir, attr_path, selected_attrs, 
            transform, mode,batch_size,image_size,toPatch,microPatchSize)
    elif dataset.lower() == 'celebahq':
        dataset = CelebAHQ(image_dir, attr_path, selected_attrs, 
            transform, mode,batch_size,image_size,toPatch,microPatchSize)
    elif dataset.lower() == 'rafd':
        dataset = ImageFolder(image_dir, transform)
    data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,drop_last=True,shuffle=(mode=='train'),num_workers=num_workers,pin_memory=True)
    return data_loader

if __name__ == "__main__":
    # from torchvision.utils import save_image
    # selected_attrs  = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
    #                         'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    # n_classes       = len(selected_attrs)
    # datapath        = "F:\\celeba-hq\\celeba-hq\\celeba-1024"
    # attpath         = "F:\\DataSet\\CelebA\\Anno\\list_attr_celeba.txt"
    # imsize          = 1024
    # MicroPathSize   = 32
    # imCropSize      = 1024
    # ToPatch         = False
    # BatchSize       = 4
    # celebahq_loader = getLoader(datapath, attpath, selected_attrs,
    #                     imCropSize, imsize, BatchSize,'CelebAHQ', 'train', 8,ToPatch,32)
    # print(len(celebahq_loader))
    # print(celebahq_loader.__len__())
    import PIL
    print(PIL.Image.BICUBIC)

    # imagesNames = ['000019.jpg','000025.jpg','000032.jpg','000041.jpg','000052.jpg','000068.jpg','000071.jpg', '000077.jpg','000079.jpg',
    #                 # '000150.jpg','000173.jpg','000176.jpg','000204.jpg','000208.jpg','000228.jpg','000248.jpg','000252.jpg','000254.jpg','000256.jpg',
    #                 # '000272.jpg','000275.jpg','000287.jpg','000300.jpg','000302.jpg','000316.jpg','000335.jpg','000352.jpg','000383.jpg','000389.jpg','000392.jpg','000495.jpg','000537.jpg','000573.jpg','000667.jpg','000676.jpg','000687.jpg',
    #                 '182001.jpg','182002.jpg','182003.jpg','182004.jpg','182005.jpg','182006.jpg', '182007.jpg','182008.jpg','182009.jpg',
    #                 '182010.jpg','182011.jpg','182012.jpg', '182013.jpg','182014.jpg','182015.jpg','182016.jpg']
    # SampleImgNum = len(imagesNames)


    # slidNum         = imsize//MicroPathSize
    # batchNum        = slidNum ** 2
    # fixedSample     = LoadSpecifiedImagesCelebA(datapath, attpath, selected_attrs,
    #                     imCropSize,imsize,imagesNames,ToPatch,MicroPathSize)
    # fixedImages, fixedLabel = fixedSample.getSamples()
    # fixeDelta       = torch.zeros(n_classes,SampleImgNum,batchNum,n_classes)
    # for index in range(n_classes):
    #     for index1 in range(SampleImgNum):
    #         if fixedLabel[index1,0,index] == 0:
    #             fixeDelta[index,index1,:,index] = 1
    #             if index == 0 and fixedLabel[index1,0,1]==1: # from non-blad to blad, so we need to remove bangs in the same time
    #                 fixeDelta[index,index1,:,1] = -1
    #             if index == 2:
    #                 if fixedLabel[index1,0,3] == 1:
    #                     fixeDelta[index,index1,:,3] = -1
    #                 if fixedLabel[index1,0,4] == 1:
    #                     fixeDelta[index,index1,:,4] = -1
    #             if index == 3:
    #                 if fixedLabel[index1,0,2] == 1:
    #                     fixeDelta[index,index1,:,2] = -1
    #                 if fixedLabel[index1,0,4] == 1:
    #                     fixeDelta[index,index1,:,4] = -1
    #             if index == 4:
    #                 if fixedLabel[index1,0,2] == 1:
    #                     fixeDelta[index,index1,:,2] = -1
    #                 if fixedLabel[index1,0,3] == 1:
    #                     fixeDelta[index,index1,:,3] = -1
    #         else:
    #             if index == 2:
    #                 fixeDelta[index,index1,:,3] = 1 # translate black hair to blond hair
    #             if index == 3:
    #                 fixeDelta[index,index1,:,4] = 1 # translate blond hair to Brown Hair 
    #             if index == 4:
    #                 fixeDelta[index,index1,:,3] = 1
                    
    #             fixeDelta[index,index1,index] = -1