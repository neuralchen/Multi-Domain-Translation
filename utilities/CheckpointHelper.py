#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: CheckpointHelper.py
# Created Date: Saturday October 12th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 14th February 2020 5:59:00 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

import os
import torch

# def loadPretrainedModel(gModel,dModel,modelSavePath,chechpointStep,dlr=None,glr=None,*args):
#     gModel.load_state_dict(torch.load(os.path.join(
#         modelSavePath, '{}_LocalG.pth'.format(chechpointStep))))
#     dModel.load_state_dict(torch.load(os.path.join(
#         modelSavePath, '{}_GlobalD.pth'.format(chechpointStep))))
#     print('loaded trained models (step: {}) successful!'.format(chechpointStep))
#     if dlr:
#         glr.load_state_dict(torch.load(os.path.join(
#             modelSavePath, '{}_LocalGlr.pth'.format(chechpointStep))))
#         dlr.load_state_dict(torch.load(os.path.join(
#             modelSavePath, '{}_GlobalDlr.pth'.format(chechpointStep))))
#         print("Generator learning rate:%f"%glr.get_lr()[0])
#         print("Discriminator learning rate:%f"%dlr.get_lr()[0])
#     print('loaded trained models learning rate (step: {}) successful!'.format(chechpointStep))

def loadPretrainedModel(chechpointStep,modelSavePath,gModel,dModel,cuda,**kwargs):
    gModel.load_state_dict(torch.load(os.path.join(
        modelSavePath, 'Epoch{}_LocalG.pth'.format(chechpointStep)),map_location=cuda))
    dModel.load_state_dict(torch.load(os.path.join(
        modelSavePath, 'Epoch{}_GlobalD.pth'.format(chechpointStep)),map_location=cuda))
    print('loaded trained models (epoch: {}) successful!'.format(chechpointStep))
    if not kwargs:
        return
    for k,v in kwargs.items():
        v.load_state_dict(torch.load(os.path.join(
            modelSavePath, 'Epoch{}_{}.pth'.format(chechpointStep,k)),map_location=cuda))
        print("Loaded param %s"%k)

def loadPretrainedModelByDict(chechpointStep,modelSavePath,cuda,**kwargs):
    if not kwargs:
        return
    for k,v in kwargs.items():
        v.load_state_dict(torch.load(os.path.join(
            modelSavePath, 'Epoch{}_{}.pth'.format(chechpointStep,k)),map_location=cuda))
        print("Loaded param %s"%k)

def loadLR(chechpointStep,modelSavePath,dlr,glr):
    glr.load_state_dict(torch.load(os.path.join(
        modelSavePath, 'Epoch{}_LocalGlr.pth'.format(chechpointStep))))
    dlr.load_state_dict(torch.load(os.path.join(
        modelSavePath, 'Epoch{}_GlobalDlr.pth'.format(chechpointStep))))
    print("Generator learning rate:%f"%glr.get_lr()[0])
    print("Discriminator learning rate:%f"%dlr.get_lr()[0])
    
def saveLR(step,modelSavePath,dlr,glr):
    torch.save(glr.state_dict(),os.path.join(modelSavePath, 'Epoch{}_LocalGlr.pth'.format(step + 1)))
    torch.save(dlr.state_dict(),os.path.join(modelSavePath, 'Epoch{}_GlobalDlr.pth'.format(step + 1)))
    print("Epoch:{} models learning rate saved!".format(step+1))
    
# def saveModel(gModel,dModel,modelSavePath,step,dlr=None,glr=None,**kwargs):
#     torch.save(gModel.state_dict(),
#                         os.path.join(modelSavePath, '{}_LocalG.pth'.format(step + 1)))
#     torch.save(dModel.state_dict(),
#                         os.path.join(modelSavePath, '{}_GlobalD.pth'.format(step + 1)))
#     print("step:{} models saved!".format(step))
#     if dlr:
#         torch.save(glr.state_dict(),
#                             os.path.join(modelSavePath, '{}_LocalGlr.pth'.format(step + 1)))
#         torch.save(dlr.state_dict(),
#                             os.path.join(modelSavePath, '{}_GlobalDlr.pth'.format(step + 1)))
#         print("step:{} models learning rate saved!".format(step))
#     if *args

def saveModel(step,modelSavePath,gModel,dModel,**kwargs):
    torch.save(gModel.state_dict(),
                        os.path.join(modelSavePath, 'Epoch{}_LocalG.pth'.format(step + 1)))
    torch.save(dModel.state_dict(),
                        os.path.join(modelSavePath, 'Epoch{}_GlobalD.pth'.format(step + 1)))
    print("Epoch:{} models saved!".format(step+1))
    if not kwargs:
        return
    for k,v in kwargs.items():
        torch.save(v.state_dict(),
                            os.path.join(modelSavePath, 'Epoch{}_{}.pth'.format(step + 1,k)))
        print("Epoch:{} models param {} saved!".format(step+1,k))

def saveModelByDict(step,modelSavePath,**kwargs):
    if not kwargs:
        return
    for k,v in kwargs.items():
        torch.save(v.state_dict(),
                            os.path.join(modelSavePath, 'Epoch{}_{}.pth'.format(step + 1,k)))
        print("Epoch:{} models param {} saved!".format(step+1,k))
