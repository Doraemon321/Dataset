# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:49:08 2021

@author: PQD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
import modules
import utils

class PAN(nn.Module):
    def __init__(self, num_class, bn=False):
        super(PAN, self).__init__()
        self.LAEconv0 = modules.LAEconv(3, 64, radius=0.06, m=1, bn=bn)
        self.LAEconv1 = modules.LAEconv(64, 128, radius=0.12, m=1, bn=bn)
        self.LAEconv2 = modules.LAEconv(128, 256, radius=0.3, m=1, bn=bn)
        self.PSAB1 = modules.self_attention_module(256, 256, 256, merge='add')
        self.LAEconv3 = modules.LAEconv(256, 512, radius=0.5, m=1, bn=bn)
        self.PSAB2 = modules.self_attention_module(512, 512, 512, merge='add')
        self.fp1 = modules.pointnet_fp_module(mlp=None)
        self.LAEconv4 = modules.LAEconv(768, 256, radius=0.3, m=1, bn=bn)
        self.PSAB3 = modules.self_attention_module(256, 256, 256, merge='add')
        self.fp2 = modules.pointnet_fp_module(mlp=None)
        self.LAEconv5 = modules.LAEconv(384, 256, radius=0.12, m=1, bn=bn)
        self.fp3 = modules.pointnet_fp_module(mlp=None)
        self.LAEconv6 = modules.LAEconv(320, 128, radius=0.06, m=1, bn=bn)
        self.fc = modules.conv1d(128, num_class, kernel_size=1, stride=1, act_fn=None, bn=bn)
        
    def forward(self, xyz, fea=None):
        if fea is not None:
            fea = fea.transpose(1,2).contiguous()   #(B, C, N)
        idx = utils.furthest_point_sampling(xyz, 512)   #(B, 512)
        xyz = xyz.transpose(1,2).contiguous()   #(B, 3, N)
        xyz1 = utils.gather_points(xyz, idx)    #(B, 3, 512)
        
        fea0 = self.LAEconv0(xyz, fea)  #(B, 64, N)
        
        fea1 = utils.gather_points(fea0, idx)   #(B, 64, 512)
        fea1 = self.LAEconv1(xyz1, fea1)    #(B, 128, 512)
        
        idx = utils.furthest_point_sampling(xyz1.transpose(1,2).contiguous(), 128)  #(B, 128)
        xyz2 = utils.gather_points(xyz1, idx)   #(B, 3, 128)
        fea2 = utils.gather_points(fea1, idx)   #(B, 128, 128)
        fea2 = self.LAEconv2(xyz2, fea2)    #(B, 256, 128)
        fea2 = self.PSAB1(fea2) #(B, 256, 128)
        
        idx = utils.furthest_point_sampling(xyz2.transpose(1,2).contiguous(), 64)  #(B, 64)
        xyz3 = utils.gather_points(xyz2, idx)   #(B, 3, 64)
        fea3 = utils.gather_points(fea2, idx)   #(B, 256, 64)
        fea3 = self.LAEconv3(xyz3, fea3)    #(B, 512, 64)
        fea3 = self.PSAB2(fea3) #(B, 512, 64)
        
        fea2 = self.fp1(xyz2, xyz3, fea2, fea3) #(B, 768, 128)
        fea2 = self.LAEconv4(xyz2, fea2)    #(B, 256, 128)
        fea2 = self.PSAB3(fea2) #(B, 256, 128)
        
        fea1 = self.fp2(xyz1, xyz2, fea1, fea2) #(B, 384, 512)
        fea1 = self.LAEconv5(xyz1, fea1)    #(B, 256, 512)
        
        fea0 = self.fp3(xyz, xyz1, fea0, fea1)  #(B, 320, N)
        fea0 = self.LAEconv6(xyz, fea0) #(B, 128, N)
        
        fea0 = self.fc(fea0)
        return fea0