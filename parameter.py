#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: parameter.py
# Created Date: Tuesday September 24th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 4th January 2021 1:23:38 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

import argparse

def str2bool(v):
    return v.lower() in ('true')

def getParameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train", 
                            choices=['train', 'finetune','test'], help="The mode of current project")
    parser.add_argument('--chechpoint_step', type=int, default=95)
    # Training information
    parser.add_argument('--version', type=str, default='cbn')
    parser.add_argument('--TrainScriptName', type=str, default='cbn') # trainer_lstu_common localglobal_norm
    parser.add_argument('--experiment_description', type=str, default="实验记录")
    # test
    parser.add_argument('--TestScriptsName', type=str, default='global')    # common localglobal
    parser.add_argument('--test_version', type=str, default='wori')      # lsturestgate1 lsturestchannelgate1 lsturestspatialgatea2  spatialgatedeconv1 lsturestspatialgatea2 lstunoreset2 stu2 lsturestspatialgate1 lsturestchannelgate1 concat_puresc2 lstunoreset_1 final_lstuwithc1
    parser.add_argument('--test_chechpoint_step', type=int, default=79) #822000 972000 906000
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--total_images', type=int, default=2000)
    parser.add_argument('--save_testimages', type=str2bool, default=True)
    parser.add_argument('--EnableThresIntSetting', type=str2bool, default=True)
    parser.add_argument('--TestThresInt', type=float, default=1.5)
    parser.add_argument('--ModelFigureName', type=str, default="Ours")
    
    # Model hyper-parameters
    parser.add_argument('--ThresInt', type=float, default=0.5)
    parser.add_argument('--imsize', type=int, default=128)
    parser.add_argument('--imCropSize', type=int, default=128)
    
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--d_fc_dim', type=int, default=1024)
    
    parser.add_argument('--GEncActName', type=str, default='leakyrelu',choices=['relu', 'selu','leakyrelu'])
    parser.add_argument('--GDecActName', type=str, default='relu',choices=['relu', 'selu','leakyrelu'])
    parser.add_argument('--GOutActName', type=str, default='tanh',choices=['tanh','hardtanh'])
    parser.add_argument('--DActName', type=str, default='leakyrelu',choices=['relu', 'selu','leakyrelu'])
    parser.add_argument('--gLayerNum', type=int, default=5)
    parser.add_argument('--resNum', type=int, default=0)
    parser.add_argument('--dLayerNum', type=int, default=5)

    # Training setting
    parser.add_argument('--selected_attrs', nargs='+', help='selected attributes for the CelebA dataset',
                default=['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'])
    parser.add_argument('--selected_simple_attrs', nargs='+', help='selected attributes for the CelebA dataset',
                default=['Bald', 'Bangs', 'Black', 'Blond', 'Brown', 'Eyebrows', 'glass',
                'Male', 'Mouth', 'Mustache', 'NoBeard', 'Pale', 'Young'])
    parser.add_argument('--total_step', type=int, default=1000000, help='how many times to update the generator')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--LoadPatchData', type=str2bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--g_lr', type=float, default=0.0003)
    parser.add_argument('--d_lr', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=int, default=100)
    parser.add_argument('--lr_decay_enable', type=str2bool, default=True)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--D_step', type=int, default=5)
    parser.add_argument('--G_step', type=int, default=1)
    parser.add_argument('--GPWeight', type=float, default=10.0)
    parser.add_argument('--RecWeight', type=float, default=100)
    parser.add_argument('--GAttrWeight', type=float, default=10.0)
    parser.add_argument('--DAttrWeight', type=float, default=1.0)
    parser.add_argument('--logit_thre', type=float, default=0.5)
    parser.add_argument('--classification_loss_type', type=str, default='cross-entropy',choices=["hinge",'cross-entropy'])
    parser.add_argument('--PSNR', type=str2bool, default=True)
    # using pretrained
    parser.add_argument('--PretrainedModelPath', type=str, default="")
    
    # Misc
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--GPUs', type=str, default='0', help='gpuids eg: 0,1,2,3  --parallel True  ')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['lsun', 'CelebA','cifar10','CelebAHQ',"CelebA384"])
    parser.add_argument('--specifiedImages', nargs='+', help='selected images for validation',
                    # default=['000025.jpg','000357.jpg','000497.jpg','000902.jpg','000052.jpg',
                    #          '001210.jpg','001316.jpg', '001818.jpg','002764.jpg', '003105.jpg',
                    #          '000392.jpg','000495.jpg','000537.jpg','000573.jpg','000667.jpg',
                    #          '000676.jpg','000687.jpg','000302.jpg','000316.jpg','000335.jpg',"005156.jpg"])
                    default=['000025.jpg','000052.jpg','000085.jpg','000109.jpg','000110.jpg',
                            '000121.jpg','000129.jpg', '000152.jpg','000161.jpg', '000168.jpg',
                            #  '000239.jpg','000273.jpg','000306.jpg','000357.jpg','000383.jpg',
                             '000406.jpg','000497.jpg','000521.jpg','000710.jpg','000930.jpg'])
                    # default=['000001.jpg','000002.jpg','000003.jpg','000902.jpg','000052.jpg',
                    #          '001210.jpg','001316.jpg', '001818.jpg','002764.jpg', '003105.jpg',
                    #          '000392.jpg','000495.jpg','000537.jpg','000573.jpg','000667.jpg',
                    #          '000676.jpg','000687.jpg','000302.jpg','000316.jpg','000335.jpg']
    # parser.add_argument('--specifiedImages', nargs='+', help='selected images for validation',
    #                 default=['000019.jpg','000025.jpg','000032.jpg','000041.jpg','000052.jpg','000068.jpg','000071.jpg', '000077.jpg','000079.jpg',
    #                 '000150.jpg','000173.jpg','000176.jpg','000204.jpg','000208.jpg','000228.jpg','000248.jpg','000252.jpg','000254.jpg','000256.jpg',
    #                 '000272.jpg','000275.jpg','000287.jpg','000300.jpg','000302.jpg','000316.jpg','000335.jpg','000352.jpg','000383.jpg','000389.jpg',
    #                 '000392.jpg','000495.jpg','000537.jpg','000573.jpg','000667.jpg','000676.jpg','000687.jpg'])
    parser.add_argument('--specifiedTestImages', nargs='+', help='selected images for validation', # '000121.jpg','000124.jpg','000129.jpg','000132.jpg','000135.jpg','001210.jpg','001316.jpg', 
                    # default=['000406.jpg','000814.jpg','001818.jpg','002764.jpg', '003105.jpg',"005156.jpg","012622.jpg","012486.jpg",'009698.jpg','009530.jpg','008472.jpg'])
                    default=['012508.jpg','012772.jpg','012753.jpg','012591.jpg', '012495.jpg',"012486.jpg","012371.jpg","012343.jpg",'012332.jpg','012321.jpg','012317.jpg'])
    # Path
    parser.add_argument('--logRootPath', type=str, default='./TrainingLogs')
    parser.add_argument('--testRoot', type=str, default="F:\\TestFiles")
    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=6000)

    return parser.parse_args()