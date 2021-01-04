######################################################################
#  script name  : FaceAttributesClassifer.py
#  author       : Chen Xuanhong
#  created time : 2019/9/23 17:29
#  modification time ：2019/9/23 17:39
#  modified by  : Chen Xuanhong
######################################################################
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils

class FaceAttributesClassifier(nn.Module):
    def __init__(
                    self,
                    conv_dim=48,
                    image_size=384,
                    attr_dim=13,
                    fc_dim=768,
                    n_layers=10):
        super(FaceAttributesClassifier, self).__init__()
        self.conv = nn.ModuleList()
        in_channels = 3
        for i in range(n_layers):
            j = i / 2 - 1
            if i % 2 == 0:
                self.conv.append(nn.Sequential(
                    nn.Conv2d(in_channels, int(conv_dim * 2 ** j), 3, 1, 1, bias=False),
                    nn.BatchNorm2d(int(conv_dim * 2 ** j), affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                ))

                in_channels = int(conv_dim * 2 ** j)

            else:
                self.conv.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(in_channels, affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ))
        feature_size = image_size // 2**(n_layers//2)
        feature_size = 4
        self.fc_att = nn.Sequential(
            nn.Linear(conv_dim * 2**(n_layers//2 - 2) * feature_size**2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, attr_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        y=x
        for layer in self.conv:
            y=layer(y)
        y = y.view(y.size()[0], -1)
        logit_att = self.fc_att(y)
        return logit_att
