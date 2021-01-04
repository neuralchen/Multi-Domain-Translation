#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_final.py
# Created Date: Friday November 8th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 4th January 2021 10:58:45 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


import os
import time
import datetime
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torchvision.utils import save_image

from utilities.Utilities import *
from utilities.Reporter import Reporter

from dataTool.DataHelper import getLoader
from utilities.BatchTensorClass import BatchTensorClass

from metric.PSNR import PSNR
from metric.FaceAttributesClassifier_128 import FaceAttributesClassifier

import prettytable as pt
import json

from tqdm import tqdm

import pytorch_ssim

class Tester(object):
    def __init__(self, config):
        for k in config.__dict__:
            setattr(self,k, config.__dict__[k])

        self.curProjectPath = os.path.join(config.testRoot, config.test_version)
        self.reporter       = Reporter(os.path.join(self.curProjectPath,'report.log'))
        self.reporter.writeConfig(config)

        self.device         = torch.device('cuda:%d'%config.cuda)

        self.testimagepath  = os.path.join(self.curProjectPath,'images')
        if not os.path.exists(self.testimagepath):
            os.makedirs(self.testimagepath )
        # load paramters
        self.n_classes      = len(getattr(self,"selected_attrs"))
        self.version        = config.test_version
        

        # self.thres_int      = config.ThresInt
        if config.EnableThresIntSetting:
            # self.thres_int      = config.TestThresInt
            self.thres_int      = 1.5
        else:
            self.thres_int      = getattr(self,"ThresInt")
        self.reporter.writeInfo("thres_int:"+str(self.thres_int))
        self.batch_size     = getattr(self,"test_batch_size")
        self.build_model()
        self.printTable     = pt.PrettyTable()
        self.printTable.field_names = getattr(self,"selected_simple_attrs") + ["Mean"]

        self.testTable     = pt.PrettyTable()
        self.testTable.field_names = [str(i) for i in range(self.batch_size)]
        self.ModelFigureName= config.ModelFigureName

    def build_model(self):
        device      = self.device
        package     = __import__("%s"%getattr(self, 'GeneratorScriptName'), fromlist=True)
        genClass    = getattr(package, 'Generator')
        self.GlobalG= genClass(getattr(self, 'g_conv_dim'),
                                getattr(self, 'gLayerNum'),
                                getattr(self, 'resNum'),
                                self.n_classes,
                                getattr(self, 'skipNum'), 
                                getattr(self, 'skipRatio'),
                                getattr(self, 'GEncActName'),
                                getattr(self, 'GSkipActName'),
                                getattr(self, 'GDecActName'),
                                getattr(self, 'GOutActName'),
                                getattr(self, "LSTUScriptName")).to(device)
                                # self.LSTUScriptName).to(self.device)
        model_save_path = getattr(self,"model_save_path")
        chechpoint_step = getattr(self,"test_chechpoint_step")
        ACCModel        = getattr(self,"ACCModel")               
        self.GlobalG.load_state_dict(torch.load(os.path.join(
                                                model_save_path,
                                                'Epoch{}_LocalG.pth'.format(chechpoint_step)),
                                                map_location=device)
                                            )
        print('loaded trained models (step: {})..!'.format(chechpoint_step))
        self.Classifier = FaceAttributesClassifier(attr_dim=self.n_classes).to(device)
        self.Classifier.load_state_dict(torch.load(ACCModel,map_location=device))
        print("load classifer model successful!")
    
    def getNonConflictingLablels(self,orignalLabel,class_n):
        fixeDelta = torch.zeros_like(orignalLabel)
        labelSize = orignalLabel.size()[0]
        outlabel = torch.zeros_like(orignalLabel)
        for index in range(labelSize):
            if orignalLabel[index,class_n] == 0:
                fixeDelta[index,class_n] = 1
                outlabel[index,class_n] = 1
                # if index == 0 and fixedLabel[index1,0,1]==1: # from non-blad to blad, so we need to remove bangs in the same time
                #     fixeDelta[index,index1,:,1] = -1
                if class_n == 0:
                    if orignalLabel[index,1] == 1:
                        fixeDelta[index,1] = -1
                if class_n == 1:
                    if orignalLabel[index,0] == 1:
                            fixeDelta[index,0] = -1
                if class_n == 2:
                    if orignalLabel[index,3] == 1:
                        fixeDelta[index,3] = -1
                    if orignalLabel[index,4] == 1:
                        fixeDelta[index,4] = -1
                if class_n == 3:
                    if orignalLabel[index,2] == 1:
                        fixeDelta[index,2] = -1
                    if orignalLabel[index,4] == 1:
                        fixeDelta[index,4] = -1
                if class_n == 4:
                    if orignalLabel[index,2] == 1:
                        fixeDelta[index,2] = -1
                    if orignalLabel[index,3] == 1:
                        fixeDelta[index,3] = -1
            else:
                fixeDelta[index,class_n] = -1
                outlabel[index,class_n] = -1
                if class_n == 2:
                    fixeDelta[index,3] = 1 # translate black hair to blond hair
                if class_n == 3:
                    fixeDelta[index,4] = 1 # translate blond hair to Brown Hair
                if class_n == 4:
                    fixeDelta[index,3] = 1
        return fixeDelta,outlabel
            
    def test(self):
        # Start time
        start_time = time.time()
        data_loader = getLoader(getattr(self,"image_path"), 
                                getattr(self,"attributes_path"),
                                getattr(self,"selected_attrs"),
                                getattr(self,"imCropSize"),
                                getattr(self,"imsize"),
                                self.batch_size,
                                dataset= getattr(self,"dataset"),
                                num_workers=0,
                                toPatch=False,
                                mode='test',
                                microPatchSize=0
                            )

        chechpoint_step = getattr(self,"chechpoint_step")
        device          = self.device
        total_images    = getattr(self,"total_images")
        save_testimages = getattr(self,"save_testimages")
        imsize          = getattr(self,"imsize")

        # Fixed input for debugging
        # BatchText   = BatchTensorClass(30,[10,10])
        data_iter = iter(data_loader)
        # BatchText1   = BatchTensorClass(30,[60,10],color='blue')
        total = total_images
        self.GlobalG.eval()
        self.Classifier.eval()
        acc_cnt_generate = np.zeros([self.n_classes])
        total_psnr = 0.0
        total_ssim = 0.0
        with torch.no_grad():
            for iii in tqdm(range(total//self.batch_size)):
                try:
                    realImages, labelOriginal = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    realImages, labelOriginal = next(data_iter)
                res = realImages
                realImages = realImages.to(device)
                labelOriginal = labelOriginal.to(device)
                
                for index in range(self.n_classes+1):
                    if index<(self.n_classes):
                        templabel,gt_label   = self.getNonConflictingLablels(labelOriginal,index)
                        templabel   = templabel.to(device) *self.thres_int 
                        xFake       = self.GlobalG(realImages,templabel)
                        if imsize > 128:
                            xFakelow = F.interpolate(xFake,size=128)
                        else:
                            xFakelow = xFake
                        wocao       = self.Classifier(xFakelow)
                        pred_laebl  = torch.round(wocao)*2-1
                        # gtpre       = torch.round(self.Classifier(realImages))*2-1
                        # print(self.selected_attrs[index])
                        # self.testTable.add_row(labelOriginal[:,index].cpu().numpy())
                        # self.testTable.add_row(gtpre[:,index].cpu().numpy())
                        # self.testTable.add_row(templabel[:,index].cpu().numpy())
                        # self.testTable.add_row(pred_laebl[:,index].cpu().numpy())
                        # self.testTable.add_row(wocao[:,index].cpu().numpy())
                        # print(self.testTable)
                        # self.testTable.clear_rows()
                        # self.reporter.writeInfo()
                        # gtpre       = torch.round(self.Classifier(realImages))*2-1
                        # print("getpre:")
                        # print(gtpre)
                        # print("prelabel:")
                        # print(labelOriginal)
                        # print("prelabel:")
                        # print(pred_laebl)
                        pre = pred_laebl[:,index]
                        gt  = gt_label[:,index].to(device)
                        acc_generate= (pre==gt).cpu().numpy()
                        acc_cnt_generate[index] += np.sum(np.sum(acc_generate,axis=0),axis=0)
                        if save_testimages:
                            imgSamples  = xFake.cpu()
                            res         = torch.cat([res,imgSamples],0)
                    else:
                        recLabel    = torch.zeros_like(labelOriginal)
                        xFake       = self.GlobalG(realImages,recLabel)
                        _,meanPSNR = PSNR(xFake.cpu(),realImages.cpu())
                        total_psnr += (meanPSNR * float(labelOriginal.size()[0]))

                        imgs1_ssim      = (realImages+1)/2
                        imgs2_ssim      = (xFake+1)/2
                        per_ssim_value  = pytorch_ssim.ssim(imgs1_ssim.cpu(), imgs2_ssim.cpu()).item()
                        total_ssim      += per_ssim_value * self.batch_size

                        if save_testimages:
                            imgSamples  = xFake.cpu()
                            res         = torch.cat([res,imgSamples],0)
                
                if save_testimages:
                    print("Save test data")
                    save_image(denorm(res.data),
                                os.path.join(self.testimagepath, '{}_fake.png'.format(iii + 1)),nrow=self.batch_size)#,nrow=self.batch_size)
                # return
        tabledata  = {"acc":[],'psnr':0.0,'ssim':0.0,'step':0}
        total_ssim = total_ssim / total_images
        tabledata['step'] = chechpoint_step
        tabledata['ssim'] = total_ssim
        total_psnr = total_psnr/float(total)
        acc_gen =acc_cnt_generate/(total)
        acc     = ["%.3f"%x for x in acc_gen]
        meanAcc = sum(acc_gen)/13.0
        acc     += ["%.3f"%meanAcc]
        tabledata['acc'] = acc
        self.printTable.add_row(acc)
        print("The %s model logs:"%self.version)
        print(self.printTable)
        self.reporter.writeInfo("\n"+self.printTable.__str__())
        print("The mean PSNR is %.2f"%total_psnr)
        tabledata['psnr'] = total_psnr
        print("The average SSIM is %.4f"%total_ssim)
        self.reporter.writeInfo("The mean PSNR is %.2f"%total_psnr)
        self.reporter.writeInfo("The average SSIM is %.4f"%total_ssim)
        bestResultPath = os.path.join(self.curProjectPath,'bestResult.json')
        bestResultCsv  = os.path.join(self.curProjectPath,'bestResult.csv')
        ResultCsv  = os.path.join(self.curProjectPath,'step%d_Result.csv'%chechpoint_step)
        attrname = ["Bald","Bangs","Black",
                    "Blond","Brown","Eyebrows",
                    "Glass","Male","Mouth",
                    "Mustache","NoBeard","Pale",
                    "Young","Average"]
        if os.path.exists(bestResultPath):
            with open(bestResultPath,'r') as cf:
                score = cf.read()
                score = json.loads(score)
                if score['acc'][13] < acc[13]:
                    with open(bestResultPath,'w') as cf:
                        scorejson = json.dumps(tabledata)
                        cf.writelines(scorejson)
                    import csv
                    headers = ['Name','Arch','Score']
                    rows = []
                    for i in range(14):
                        rows.append({
                            "Name":attrname[i],
                            "Arch":self.ModelFigureName,
                            "Score":acc[i],
                        })

                    with open(bestResultCsv,'w',newline='')as f:
                        f_csv = csv.DictWriter(f,headers)
                        f_csv.writeheader()
                        f_csv.writerows(rows)
        else:
            with open(bestResultPath,'w') as cf:
                scorejson = json.dumps(tabledata)
                cf.writelines(scorejson)

        import csv
        headers = ['Name','Arch','Score']
        rows = []
        for i in range(14):
            rows.append({
                "Name":attrname[i],
                "Arch":self.ModelFigureName,
                "Score":acc[i],
            })

        with open(ResultCsv,'w',newline='')as f:
            f_csv = csv.DictWriter(f,headers)
            f_csv.writeheader()
            f_csv.writerows(rows)    
        
        
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))