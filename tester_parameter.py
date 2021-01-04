#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: parameter.py
# Created Date: Tuesday September 24th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 12th November 2019 4:45:38 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################



import argparse

def str2bool(v):
    return v.lower() in ('true')

def getParameters():
    parser = argparse.ArgumentParser()
    # Training information
    parser.add_argument('--version', type=str, default='patch1')
    # test
    parser.add_argument('--TestScriptsName', type=str, default='global')
    parser.add_argument('--test_version', type=str, default='globalmodeltest')#spatialgatedeconv1 lsturestspatialgatea2 lstunoreset2 stu2 lsturestspatialgate1 lsturestchannelgate1 concat_puresc2 lstunoreset_1 final_lstuwithc1
    parser.add_argument('--read_node_local', type=str2bool, default=True)
    parser.add_argument('--model_node', type=str, default='localhost',choices=['4card', '8card','lyh','localhost'])
    parser.add_argument('--test_chechpoint_step', type=int, default=216000)
    parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--total_images', type=int, default=2000)
    parser.add_argument('--save_testimages', type=str2bool, default=False)
    parser.add_argument('--EnableThresIntSetting', type=str2bool, default=True)
    parser.add_argument('--TestThresInt', type=float, default=0.45)
    parser.add_argument('--ModelFigureName', type=str, default="Ours")
    
    # Model hyper-parameters
    parser.add_argument('--ThresInt', type=float, default=0.5)
    parser.add_argument('--imsize', type=int, default=128)
    parser.add_argument('--GlobalImSize', type=int, default=384)
    parser.add_argument('--imCropSize', type=int, default=178)
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--GEncActName', type=str, default='leakyrelu',choices=['relu', 'selu','leakyrelu'])
    parser.add_argument('--GDecActName', type=str, default='relu',choices=['relu', 'selu','leakyrelu'])
    parser.add_argument('--GOutActName', type=str, default='tanh',choices=['tanh','hardtanh'])
    parser.add_argument('--GSkipActName', type=str, default='leakyrelu',choices=['relu', 'selu','leakyrelu'])
    parser.add_argument('--GlobalGConvDim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--d_fc_dim', type=int, default=512)
    parser.add_argument('--DActName', type=str, default='leakyrelu',choices=['relu', 'selu','leakyrelu'])
    parser.add_argument('--gLayerNum', type=int, default=5)
    parser.add_argument('--skipNum', type=int, default=1)
    parser.add_argument('--skipRatio', type=float, default=1)
    parser.add_argument('--resNum', type=int, default=0)
    parser.add_argument('--dLayerNum', type=int, default=5)
    # Training setting
    parser.add_argument('--selected_attrs', nargs='+', help='selected attributes for the CelebA dataset',
                default=['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'])
    parser.add_argument('--selected_simple_attrs', nargs='+', help='selected attributes for the CelebA dataset',
                default=['Bald', 'Bangs', 'Black', 'Blond', 'Brown', 'Eyebrows', 'glass',
                'Male', 'Mouth', 'Mustache', 'NoBeard', 'Pale', 'Young'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--MicroPathSize', type=int, default=64)
    parser.add_argument('--logit_thre', type=float, default=0.5)
    parser.add_argument('--PSNR', type=str2bool, default=True)
    parser.add_argument('--ACC', type=str2bool, default=True)
    parser.add_argument('--ACCModel', type=str, default="./Pretrained/Classifier/70000_G.pth")

    # using pretrained
    parser.add_argument('--PretrainedModelPath', type=str, default="")
    parser.add_argument('--use_pretrained_model', type=str2bool, default=False)
    parser.add_argument('--chechpoint_step', type=int, default=264000)

    # Misc
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--use_system_cuda_way', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--GPUs', type=str, default='0', help='gpuids eg: 0,1,2,3  --parallel True  ')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['lsun', 'CelebA','cifar10','CelebAHQ'])
    parser.add_argument('--specifiedTestImages', nargs='+', help='selected images for validation',
                    default=['000121.jpg','000124.jpg','000129.jpg','000132.jpg','000135.jpg','001210.jpg','001316.jpg', '001818.jpg','002764.jpg', '003105.jpg'])
    # Path
    parser.add_argument('--logRootPath', type=str, default='./TrainingLogs')
    parser.add_argument('--testRoot', type=str, default="./TestFiles")
    parser.add_argument('--GlobalModel', type=str, default="D:\\Workspace\\PatchBasedFaceAttributesEditing\\Pretrained\\GlobalModel\\520000_G.pth")
    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=6000)
    return parser.parse_args()

if __name__ == "__main__":
    import  platform
    import  os
    from    torch.backends import cudnn
    from    utilities.Utilities import makeFolder
    
    config = getParameters()
    if config.use_system_cuda_way:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)
        config.cuda = 0
    # For fast training
    cudnn.benchmark = True
    # Create directories if not exist
    if platform.system() == "Linux":
        print("Current OS is Linux")
        dataSetPath ={
            'cifar10'   :'./data/cifar10',
            'cifar100'  :'./data/cifar100',
            'celeba'    :'/home/gdp/CXH/celeba/celeba_align/Img/img_align_celeba',
            'celebaatt' :'/home/gdp/CXH/celeba/celeba_align/Img/list_attr_celeba.txt',
            'celebahq'  :'../celeba/celeba-hq/celeba-256',
            'celebahqatt':'../celeba/celeba-hq/list_attr_celeba.txt',
            'lsun'      :''
        }
        config.image_path       = dataSetPath[config.dataset.lower()]
        config.attributes_path  = dataSetPath[config.dataset.lower()+'att']
    else:
        dataSetPath ={
            'cifar10'   :'./data/cifar10',
            'cifar100'  :'./data/cifar100',
            'celeba'    :'F:\\DataSet\\CelebA\\Img\\img_align_celeba\\img_align_celeba',
            'celebaatt' :'F:\\DataSet\\CelebA\\Img\\img_align_celeba\\list_attr_celeba.txt',
            'celebahq'  :'F:\\celeba-hq\\celeba-hq\\celeba-256',
            'celebahqatt':'F:\\Celeba\\data_crop_384_jpg\\list_attr_celeba.txt',
            'lsun'      :''
        }
        config.image_path       = dataSetPath[config.dataset.lower()]
        config.attributes_path  = dataSetPath[config.dataset.lower()+'att']
   
    if not os.path.exists(config.logRootPath):
        os.makedirs(config.logRootPath)
        
    makeFolder(config.logRootPath, config.test_version)
    currentProjectPath = os.path.join(config.logRootPath, config.test_version)

    makeFolder(currentProjectPath, "checkpoint")
    config.model_save_path = os.path.join(currentProjectPath, "checkpoint")

    makeFolder(config.testRoot, config.test_version)
    moduleName  = "TestScripts.tester_" + config.TestScriptsName
    tempstr     = "Start to run test script: {}".format(moduleName)
    print(tempstr)
    package     = __import__(moduleName, fromlist=True)
    testerClass = getattr(package, 'Tester')
    tester      = testerClass(config)
    tester.test()