#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_lstu.py
# Created Date: Thursday October 31st 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 4th January 2021 1:25:07 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


import os
import sys
import time
import datetime
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from utilities.Utilities import *
from utilities.Reporter import Reporter

from components.GeneratorNoSCCBN import Generator
from components.Discriminator import Discriminator

from loss.LossUtili import getClassifierLoss
from loss.LossUtili import gradientPenalty

from dataTool.DataHelper import getLoader
from dataTool.LoadSpecifiedImagesCelebA1 import getNonConflictingImgsLablels
from utilities.CheckpointHelper import loadPretrainedModel,saveModel

from metric.PSNR import PSNR

DEBUG = False

class Trainer(object):
    def __init__(self, data_loader, config):
        # logger
        self.reporter = config.reporter

        # Data loader
        self.data_loader = data_loader

        # Model hyper-parameters
        self.g_conv_dim = config.g_conv_dim
        self.GEncActName= config.GEncActName
        self.GDecActName= config.GDecActName
        self.d_conv_dim = config.d_conv_dim
        self.DActName   = config.DActName
        self.GOutActName= config.GOutActName
        self.resNum     = config.resNum
        self.gLayerNum  = config.gLayerNum
        self.dLayerNum  = config.dLayerNum
        self.d_fc_dim   = config.d_fc_dim


        # Loss type and weights
        self.classification_loss_type = config.classification_loss_type
        self.GPWeight       = config.GPWeight
        self.RecWeight      = config.RecWeight
        self.GAttrWeight    = config.GAttrWeight
        self.DAttrWeight    = config.DAttrWeight
        self.g_lr           = config.g_lr
        self.d_lr           = config.d_lr
        self.lr_decay       = config.lr_decay
        self.lr_decay_step  = config.lr_decay_step
        self.beta1          = config.beta1
        self.beta2          = config.beta2
        
        # training information
        self.version        = config.version
        self.imsize         = config.imsize
        self.imCropSize     = config.imCropSize
        self.parallel       = config.parallel
        self.device         = torch.device('cuda:%d'%config.cuda)
        self.GPUs           = config.GPUs
        self.num_workers    = config.num_workers
        self.batch_size     = config.batch_size
        self.selected_attrs = config.selected_attrs
        self.simple_attrs   = config.selected_simple_attrs
        self.n_classes      = len(self.selected_attrs)
        self.specifiedImages= config.specifiedImages
        self.SampleImgNum   = len(config.specifiedImages)
        self.PSNR           = config.PSNR
        self.thres_int      = config.ThresInt

        # steps
        self.total_step     = config.total_step
        self.use_pretrained_model   = True if config.mode == "finetune" else False
        self.chechpoint_step= config.chechpoint_step
        self.log_step       = config.log_step
        self.sample_step    = config.sample_step
        self.model_save_step= config.model_save_step
        self.DStep          = config.D_step
        self.GStep          = config.G_step
        
        # Path
        self.summary_path   = config.log_path
        self.sample_path    = config.sample_path
        self.model_save_path= config.model_save_path 
        self.attributes_path= config.attributes_path
        self.image_path     = config.image_path
        self.build_model()
        self.reporter.writeConfig(config)
        self.reporter.writeModel(self.GlobalG.__str__())

        self.reporter.writeModel(self.GlobalD.__str__())
        self.writer = SummaryWriter(log_dir=self.summary_path)

        # Start with trained model
        if self.use_pretrained_model:
            params = {}
            loadPretrainedModel(self.chechpoint_step,
                                self.model_save_path,
                                self.GlobalG,
                                self.GlobalD,
                                self.device,
                                **params
                                )
    
    def build_model(self):
        self.GlobalG    = Generator(self.g_conv_dim,
                                    self.gLayerNum,
                                    self.resNum,
                                    self.n_classes,
                                    self.GEncActName,
                                    self.GDecActName,
                                    self.GOutActName).to(self.device)
                                    
        self.GlobalD    = Discriminator(self.d_conv_dim, 
                                        self.n_classes,
                                        self.dLayerNum,
                                        self.DActName,
                                        self.d_fc_dim,
                                        self.imsize).to(self.device)

        if self.parallel:
            print('use parallel...')
            print('gpuids ', self.GPUs)
            GPUs    = [int(i) for i in self.GPUs.split(',')]
            self.GlobalG    = nn.DataParallel(self.GlobalG,device_ids=GPUs)
            self.GlobalD    = nn.DataParallel(self.GlobalD,device_ids=GPUs)
        # Loss and optimizer
        self.gOptimizerList = []
        self.gOptimizerList.append(torch.optim.Adam(filter(lambda p: p.requires_grad, self.GlobalG.parameters()), self.g_lr, [self.beta1, self.beta2]))

        self.dOptimizerList = []
        self.dOptimizerList.append(torch.optim.Adam(filter(lambda p: p.requires_grad, self.GlobalD.parameters()), self.d_lr, [self.beta1, self.beta2]))
        # self.g_scheduler = lr_scheduler.StepLR(self.g_optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay)
        # self.d_scheduler = lr_scheduler.StepLR(self.d_optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay)
        self.l1loss         = torch.nn.L1Loss()
        self.classLoss      = getClassifierLoss(self.classification_loss_type)
        
    
    def dResetGrad(self):
        for item in self.dOptimizerList:
            item.zero_grad()
    
    def gResetGrad(self):
        for item in self.gOptimizerList:
            item.zero_grad()
            
    def dOptimizerStep(self):
        for item in self.dOptimizerList:
            item.step()
    
    def gOptimizerStep(self):
        for item in self.gOptimizerList:
            item.step()

    def train(self):

        if DEBUG:
            import pynvml
            pynvml.nvmlInit()
            GPUHandle   = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo     = pynvml.nvmlDeviceGetMemoryInfo(GPUHandle)
            DIVNUM      = 1024 ** 3
            totalmem    = float(meminfo.total)/DIVNUM
            
        maxBatchsize= self.batch_size
        # Data iterator
        data_iter = iter(self.data_loader)
        model_save_step = self.model_save_step
        # Start with trained model
        if self.use_pretrained_model:
            start = self.chechpoint_step + 1
        else:
            start = 0
        fixedImages,fixeDelta,fixedtraget = getNonConflictingImgsLablels(
                                    self.image_path,
                                    self.imsize,
                                    self.imCropSize,
                                    self.attributes_path,
                                    self.selected_attrs,
                                    self.specifiedImages)
        # Fixed input for debugging
        thres_int   = self.thres_int
        # Start time
        start_time = time.time()
        self.reporter.writeInfo("Start to train the model")
        print("start to train")
        
        for step in range(start, self.total_step):
            self.GlobalG.train()
            self.GlobalD.train()
            # ================== Read Data ================== #
            try:
                realImages, labelOriginal = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                realImages, labelOriginal = next(data_iter) # real image: B patchnum imagechannel H W
                                                            # label:  B patchnum class num
            # Compute loss with real images          
            randIdx         = torch.randperm(labelOriginal.size(0))
            labelTarget     = labelOriginal[randIdx]
            labelOriginal   = labelOriginal.to(self.device)
            labelTarget     = labelTarget.to(self.device)
            labelOrigCopy   = labelOriginal.clone()
            labelTargetCopy = labelTarget.clone()
            labelOriginal   = (labelOriginal*2-1)*thres_int
            labelTarget     = (labelTarget*2-1)*thres_int
            conditionTrain  = labelTarget - labelOriginal
            realImages      = realImages.to(self.device)
            
            # ================== Train D ================== #
            if (step+1)%self.DStep != 0:
                xFake       = self.GlobalG(realImages,conditionTrain)
                x_real_logit_gan,x_real_logit_att=self.GlobalD(realImages)

                x_fake_logit_gan,_  =self.GlobalD(xFake.detach())
                d_loss_gan  = -x_real_logit_gan.mean() + x_fake_logit_gan.mean()

                alpha       = torch.rand(realImages.size(0), 1, 1, 1).to(self.device)
                x_hat       = (alpha * realImages + (1 - alpha) * xFake).requires_grad_(True)
                out_src,_   = self.GlobalD(x_hat)
                gp          = gradientPenalty(out_src, x_hat,self.device)
                d_loss_att  = self.classLoss(x_real_logit_att,labelOrigCopy)
                d_loss      = d_loss_gan  + d_loss_att * self.DAttrWeight + gp * self.GPWeight
                self.dResetGrad()
                d_loss.backward(retain_graph=True)
                self.dOptimizerStep()
            else:
                # ================== Train G ================== #
                recLabel            = torch.zeros_like(labelOriginal)
                xFake               = self.GlobalG(realImages,conditionTrain)
                x_fake_logit_g,x_fake_logit_att_g = self.GlobalD(xFake)
                xRec                = self.GlobalG(realImages,recLabel)
                x_fake_loss_gan     = -x_fake_logit_g.mean()
                x_fake_loss_att     = self.classLoss(x_fake_logit_att_g,labelTargetCopy)
                x_loss_rec          = self.l1loss(realImages, xRec)
                g_loss              = x_fake_loss_gan + x_fake_loss_att * self.GAttrWeight + x_loss_rec * self.RecWeight

                self.gResetGrad()
                g_loss.backward()
                self.gOptimizerStep()
            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                if not DEBUG:
                    print("[{}] [{}], Step [{}/{}], d_loss: {:.3f}, g_loss: {:.3f}".format(self.version,
                            elapsed, step + 1, self.total_step, d_loss.item(), g_loss.item()))
                else:
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(GPUHandle)
                    print("[{}] [{}], Step [{}/{}], d_loss: {:.3f}, g_loss: {:.3f}, MEM: {:.3f}/{:.3f}".format(self.version,
                            elapsed, step + 1, self.total_step, d_loss.item(), g_loss.item(),float(meminfo.used)/DIVNUM,totalmem))
                self.writer.add_scalar('G/x_fake_loss_gan', x_fake_loss_gan.item(),(step + 1))
                self.writer.add_scalar('G/x_fake_loss_att', x_fake_loss_att.item(),(step + 1))
                self.writer.add_scalar('G/x_loss_rec',      x_loss_rec.item(), (step + 1))
                self.writer.add_scalar('G/g_loss',          g_loss.item(), (step + 1))

                self.writer.add_scalar('D/d_loss_gan',  d_loss_gan.item(),(step + 1))
                self.writer.add_scalar('D/gp',          gp.item(),(step + 1))
                self.writer.add_scalar('D/d_loss_att',  d_loss_att.item(), (step + 1))
                self.writer.add_scalar('D/d_loss',      d_loss.item(), (step + 1))
            # ================== Evaluate G ================== #
            if (step + 1) % self.sample_step == 0 or (step == 1):
                self.GlobalG.eval()
                with torch.no_grad():
                    res = fixedImages
                    for index in range(self.n_classes+1):
                        endPoint = maxBatchsize
                        Flag     = True
                        while(Flag):
                            startPoint  = endPoint-maxBatchsize
                            if startPoint>=self.SampleImgNum:
                                break
                            if endPoint >=self.SampleImgNum:
                                Flag    = False
                                endPoint= self.SampleImgNum
                            tempImages  = fixedImages[startPoint:endPoint,:,:,:]
                            tempImages  = tempImages.to(self.device)
                            tempdata    = fixeDelta[index,startPoint:endPoint,:]*2*thres_int
                            tempdata    = tempdata.to(self.device)
                            xFake       = self.GlobalG(tempImages,tempdata)

                            imgSamples  = xFake.cpu()
                            res         = torch.cat([res,imgSamples],0)
                            endPoint    += maxBatchsize
                            
                            del tempImages
                            del tempdata
                    save_image(denorm(res.data),
                        os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)),nrow=self.SampleImgNum)
                    del res
            # ================== Save Checkpoints ================== #
            if (step+1) % model_save_step==0:
                params = {}
                saveModel(step,self.model_save_path,self.GlobalG,self.GlobalD,
                        **params)