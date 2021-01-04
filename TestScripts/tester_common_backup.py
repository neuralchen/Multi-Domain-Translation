#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_final.py
# Created Date: Friday November 8th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 14th November 2019 10:54:07 am
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
# from dataTool.PatchTool import PatchToolClass

# from components.Generator_init_bc_init import Generator_Pretrained

from dataTool.DataHelper import getLoader
from utilities.BatchTensorClass import BatchTensorClass
from dataTool.PatchTool import PatchToolClass


from dataTool.LoadSpecifiedImagesCelebA import LoadSpecifiedImagesCelebA

from metric.PSNR import PSNR
from metric.FaceAttributesClassifier_128 import FaceAttributesClassifier

import prettytable as pt
import json
from nodesInfo.sshupload import fileUploaderClass
from tqdm import tqdm

class Tester(object):
    def __init__(self, config):
        if config.read_node_local:
            with open('nodesInfo/modelpositions.json','r') as cf:
                nodelocaltionstr = cf.read()
                nodelocaltioninf = json.loads(nodelocaltionstr)
                self.model_node  = nodelocaltioninf[config.test_version]
                self.model_node  = self.model_node['server']
                print("model node %s"%self.model_node)
        else:
            self.model_node = config.model_node
        self.version        = config.test_version
        self.reporter       = Reporter(os.path.join(config.testRoot,self.version,'report.log'))
        self.reporter.writeInfo("version:"+str(self.version))
        self.logRootPath    = config.logRootPath
        self.testimagepath  = os.path.join(config.testRoot,self.version,'images')
        self.total_images   = config.total_images
        self.testRoot       = config.testRoot
        if not os.path.exists(self.testimagepath):
            os.makedirs(self.testimagepath )
        
        if self.model_node.lower() != "localhost":
            with open('nodesInfo/nodes.json','r') as cf:
                nodestr = cf.read()
                nodeinf = json.loads(nodestr)
            nodeinf = nodeinf[self.model_node.lower()]
            currentProjectPath  = os.path.join(self.logRootPath, self.version)
            uploader            = fileUploaderClass(nodeinf["ip"],nodeinf["user"],nodeinf["passwd"])
            remoteFile          = nodeinf['basePath'] + self.version + "/config.json"
            localFile           = os.path.join(currentProjectPath,"config.json")
            uploader.sshScpGet(remoteFile,localFile)
            print("success get the config file from server %s"%nodeinf['ip'])
            
        with open(os.path.join(config.logRootPath, self.version,'config.json'),'r') as cf:
            cfstr = cf.read()
            configObj = json.loads(cfstr)
        # Model hyper-parameters
        self.g_conv_dim = configObj['g_conv_dim']
        self.reporter.writeInfo("g_conv_dim"+str(self.g_conv_dim))
        self.GEncActName= configObj['GEncActName']
        self.reporter.writeInfo("GEncActName"+self.GEncActName)
        self.GDecActName= configObj['GDecActName']
        self.reporter.writeInfo("GDecActName:"+self.GDecActName)
        self.GSkipActName= configObj['GSkipActName']
        self.reporter.writeInfo("GSkipActName:"+self.GDecActName)
        self.resNum     = configObj['resNum']
        self.reporter.writeInfo("resNum:"+str(self.resNum))
        self.skipNum    = configObj['skipNum']
        self.reporter.writeInfo("skipNum:"+str(self.skipNum))
        self.gLayerNum  = configObj['gLayerNum']
        self.reporter.writeInfo("gLayerNum:"+str(self.gLayerNum))
        self.skipRatio  = configObj['skipRatio']
        self.reporter.writeInfo("skipRatio:"+str(self.skipRatio))
        self.GScriptName= configObj['GeneratorScriptName']
        self.reporter.writeInfo("GScriptName:"+str(self.GScriptName))
        self.GOutActName= configObj['GOutActName']
        self.reporter.writeInfo("GOutActName:"+str(self.GOutActName))
        # training information
        self.save_testimages= config.save_testimages
        
        self.imsize         = configObj['imsize']
        self.reporter.writeInfo("imsize:"+str(self.imsize))
        self.imCropSize     = configObj['imCropSize']
        self.reporter.writeInfo("imCropSize:"+str(self.imCropSize))
        self.device         = torch.device('cuda:%d'%config.cuda)
        # self.selected_attrs = configObj['selected_attrs']
        self.selected_attrs = config.selected_attrs
        self.reporter.writeInfo("selected_attrs:"+str(self.imCropSize))
        self.simple_attrs   = configObj['selected_simple_attrs']
        self.n_classes      = len(self.selected_attrs)
        self.specifiedImages= config.specifiedImages
        self.SampleImgNum   = len(config.specifiedImages)
        # self.thres_int      = config.ThresInt
        if config.EnableThresIntSetting:
            self.thres_int      = config.TestThresInt
        else:
            self.thres_int      =  configObj['ThresInt']
        self.reporter.writeInfo("thres_int:"+str(self.thres_int))
        self.batch_size     = config.test_batch_size
        self.ACCModel       = config.ACCModel

        # steps
        self.chechpoint_step= config.test_chechpoint_step
        self.reporter.writeInfo("chechpoint_step:"+str(self.chechpoint_step))
        # Path
        self.dataset        = config.dataset
        # self.sample_path    = config.sample_path
        self.model_save_path= config.model_save_path
        self.attributes_path= config.attributes_path
        self.image_path     = config.image_path
        self.getModel()
        self.build_model()
        self.printTable     = pt.PrettyTable()
        self.printTable.field_names = self.simple_attrs + ["Mean"]

        self.testTable     = pt.PrettyTable()
        self.testTable.field_names = [str(i) for i in range(self.batch_size)]
        self.ModelFigureName= config.ModelFigureName


    def getModel(self):
        server = self.model_node.lower()
        with open('nodesInfo/nodes.json','r') as cf:
            nodestr = cf.read()
            nodeinf = json.loads(nodestr)
        if server == "localhost":
            return
        else:
            nodeinf = nodeinf[server]
            # makeFolder(self.logRootPath, self.version)
            currentProjectPath  = os.path.join(self.logRootPath, self.version)
            # makeFolder(currentProjectPath, "checkpoint")
            
            remoteFile          = nodeinf['basePath']+ self.version + "/checkpoint/" + "%d_LocalG.pth"%self.chechpoint_step
            localFile           = os.path.join(currentProjectPath,"checkpoint","%d_LocalG.pth"%self.chechpoint_step)
            if os.path.exists(localFile):
                print("checkpoint already exists")
            else:
                uploader        = fileUploaderClass(nodeinf["ip"],nodeinf["user"],nodeinf["passwd"])
                uploader.sshScpGet(remoteFile,localFile)
                print("success get the model from server %s"%nodeinf['ip'])
    
    def build_model(self):
        package     = __import__("components.%s"%self.GScriptName, fromlist=True)
        genClass    = getattr(package, 'Generator')
        self.GlobalG= genClass(self.g_conv_dim,
                                self.gLayerNum,
                                self.resNum,
                                self.n_classes,
                                self.skipNum, 
                                self.skipRatio,
                                self.GEncActName,
                                self.GSkipActName,
                                self.GDecActName,
                                self.GOutActName).to(self.device)
                                
        self.GlobalG.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_LocalG.pth'.format(self.chechpoint_step)),map_location=self.device))
        print('loaded trained models (step: {})..!'.format(self.chechpoint_step))
        self.Classifier = FaceAttributesClassifier(attr_dim=self.n_classes).to(self.device)
        self.Classifier.load_state_dict(torch.load(self.ACCModel,map_location=self.device))
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
        data_loader = getLoader(
                                self.image_path, self.attributes_path,
                                self.selected_attrs, self.imCropSize, 
                                self.imsize, self.batch_size, dataset= self.dataset,
                                num_workers=0,toPatch=False,mode='test',
                                microPatchSize=0
                            )
        # Fixed input for debugging
        # BatchText   = BatchTensorClass(30,[10,10])
        data_iter = iter(data_loader)
        # BatchText1   = BatchTensorClass(30,[60,10],color='blue')
        total = self.total_images
        self.GlobalG.eval()
        self.Classifier.eval()
        acc_cnt_generate = np.zeros([self.n_classes])
        total_psnr = 0.0
        with torch.no_grad():
            for iii in tqdm(range(total//self.batch_size)):
                try:
                    realImages, labelOriginal = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    realImages, labelOriginal = next(data_iter)
                res = realImages
                realImages = realImages.to(self.device)
                labelOriginal = labelOriginal.to(self.device)
                
                for index in range(self.n_classes+1):
                    if index<(self.n_classes):
                        templabel,gt_label   = self.getNonConflictingLablels(labelOriginal,index)
                        templabel   = templabel.to(self.device) * 2*self.thres_int 
                        xFake       = self.GlobalG(realImages,templabel)
                        wocao       = self.Classifier(xFake)
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
                        gt  = gt_label[:,index].to(self.device)
                        acc_generate= (pre==gt).cpu().numpy()
                        acc_cnt_generate[index] += np.sum(np.sum(acc_generate,axis=0),axis=0)
                        if self.save_testimages:
                            imgSamples  = xFake.cpu()
                            res         = torch.cat([res,imgSamples],0)
                    else:
                        recLabel    = torch.zeros_like(labelOriginal)
                        xFake       = self.GlobalG(realImages,recLabel)
                        _,meanPSNR = PSNR(xFake.cpu(),realImages.cpu())
                        total_psnr += (meanPSNR * float(labelOriginal.size()[0]))
                        if self.save_testimages:
                            imgSamples  = xFake.cpu()
                            res         = torch.cat([res,imgSamples],0)
                
                if self.save_testimages:
                    save_image(denorm(res.data),
                                os.path.join(self.testimagepath, '{}_fake.png'.format(iii + 1)),nrow=self.batch_size)
                # return
        tabledata  = {"acc":[],'psnr':0.0,'step':0}
        tabledata['step'] = self.chechpoint_step
        total_psnr = total_psnr/float(total)
        acc_gen =acc_cnt_generate/(total)
        acc     = ["%.3f"%x for x in acc_gen]
        meanAcc = sum(acc_gen)/13.0
        acc     += ["%.3f"%meanAcc]
        tabledata['acc'] = acc
        self.printTable.add_row(acc)
        print("The %s model logs:"%self.version)
        self.reporter.writeInfo("Batch Size: %d"%self.batch_size)
        self.reporter.writeInfo("Total images: %d"%total)
        self.reporter.writeInfo("The %s model logs:"%self.version)
        print(self.printTable)
        self.reporter.writeInfo("\n"+self.printTable.__str__())
        print("The mean PSNR is %.2f"%total_psnr)
        tabledata['psnr'] = total_psnr
        self.reporter.writeInfo("The mean PSNR is %.2f"%total_psnr)
        bestResultPath = os.path.join(self.testRoot,self.version,'bestResult.json')
        bestResultCsv  = os.path.join(self.testRoot,self.version,'bestResult.csv')
        ResultCsv  = os.path.join(self.testRoot,self.version,'step%d_Result.csv'%self.chechpoint_step)
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