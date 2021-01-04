#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: GradientPenalty.py
# Created Date: Friday September 27th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 28th September 2019 12:58:42 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################



import torch

class GradientPenaltyLoss:

    def __init__(self,Discriminator):
        self.D = Discriminator

    def gradient_penalty(self, realImages,fakeImages):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(realImages.size()).cuda()
        alpha = torch.rand(realImages.size(0), 1, 1, 1).cuda()
        x_hat = (alpha * realImages + (1 - alpha) * fakeImages).requires_grad_(True)
        output,_   = self.D(x_hat)
        dydx = torch.autograd.grad(outputs=output,
                                   inputs=x_hat,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = (torch.sum(dydx**2, dim=1)).sqrt()
        return ((dydx_l2norm-1)**2).mean()