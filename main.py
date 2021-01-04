#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: main.py
# Created Date: Tuesday September 24th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 4th January 2021 1:23:58 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

from    parameter import getParameters
from    dataTool.DataHelper import getLoader
from    torch.backends import cudnn
from    utilities.Utilities import makeFolder
from    utilities.Reporter import Reporter
import  torch
import  platform
import  os
import  json
import  shutil

def main(config):
    ignoreKey = ["TestScriptsName","test_version","read_node_local",
                "model_node","test_chechpoint_step","test_batch_size",
                "total_images","save_testimages","EnableThresIntSetting",
                "ModelFigureName","use_pretrained_model",
                "chechpoint_step","train","testRoot","ACCModel","cuda",
                "use_system_cuda_way","logRootPath"]
    if config.cuda >-1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)
        config.cuda = 0
    # For fast training
    cudnn.benchmark = True

    if config.mode != "test":

        # load the dataset path
        datapath = "./dataTool/dataPath.json"
        with open(datapath,'r') as cf:
            datastr = cf.read()
            dataobj = json.loads(datastr)
            config.image_path       = dataobj[config.dataset.lower()]
            config.attributes_path  = dataobj[config.dataset.lower()+'att']
            print("Get data path: %s"%config.image_path )
        
        # create dataloader
        data_loader = getLoader(
                                    config.image_path,
                                    config.attributes_path, 
                                    config.selected_attrs,
                                    config.imCropSize, 
                                    config.imsize,
                                    config.batch_size,
                                    dataset= config.dataset,
                                    num_workers=config.num_workers
                                )
        
        # create training log dirs
        if not os.path.exists(config.logRootPath):
                os.makedirs(config.logRootPath)
        makeFolder(config.logRootPath, config.version)
        currentProjectPath = os.path.join(config.logRootPath, config.version)
        
        makeFolder(currentProjectPath, "summary")
        config.log_path = os.path.join(currentProjectPath, "summary")

        makeFolder(currentProjectPath, "checkpoint")
        config.model_save_path = os.path.join(currentProjectPath, "checkpoint")

        makeFolder(currentProjectPath, "sample")
        config.sample_path = os.path.join(currentProjectPath, "sample")

        config.reporter = ''
        configjson      = json.dumps(config.__dict__)
        with open(os.path.join(config.logRootPath, config.version,'config.json'),'w') as cf:
            cf.writelines(configjson)

        report_file = os.path.join(config.logRootPath, config.version,config.version+"_report")
        config.reporter = Reporter(report_file)

        moduleName  = "TrainingScripts.trainer_" + config.TrainScriptName
        tempstr     = "Start to run training script: {}".format(moduleName)
        print(tempstr)
        print("Traning version: %s"%config.version)
        print("Training Script Name: %s"%config.TrainScriptName)
        print("Image Size: %d"%config.imsize)
        print("Image Crop Size: %d"%config.imCropSize)
        print("ThresInt: %d"%config.ThresInt)
        print("D : G = %d : %d"%(config.D_step,config.G_step))

        config.reporter.writeInfo(tempstr)
        package     = __import__(moduleName, fromlist=True)
        trainerClass= getattr(package, 'Trainer')
        trainer     = trainerClass(data_loader, config)
        trainer.train()
    else:

        # make a dir for saving test results
        makeFolder(config.testRoot, config.test_version)
        moduleName  = "TestScripts.tester_" + config.TestScriptsName
        tempstr     = "Start to run test script: {}".format(moduleName)
        print(tempstr)
        
        package     = __import__(moduleName, fromlist=True)
        testerClass = getattr(package, 'Tester')
        tester      = testerClass(config)
        tester.test()

if __name__ == '__main__':
    config = getParameters()
    main(config)