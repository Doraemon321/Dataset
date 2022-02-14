# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:37:43 2021

@author: PQD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 padding_mode='zeros', act_fn=nn.ReLU(inplace=True), bn=False, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(conv1d, self).__init__()
        self.use_bn = bn
        if bn:
            bias = False
        self.activation = act_fn
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, 
                              dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        if bn:
            self.bn = nn.BatchNorm1d(out_channel, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 padding_mode='zeros', act_fn=nn.ReLU(inplace=True), bn=False, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(conv2d, self).__init__()
        self.use_bn = bn
        if bn:
            bias = False
        self.activation = act_fn
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, 
                              dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        if bn:
            self.bn = nn.BatchNorm2d(out_channel, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def knn(pc, k):
    """
    inputs:
        pc: (B, C, N) tensor
        k: int
    return:
        idx: (B, N, k) tensor
    """
    inner = -2 * torch.matmul(pc.transpose(1,2), pc)
    pc2 = torch.sum(pc**2, dim=1, keepdim=True)
    dis_mat = -pc2 - inner - pc2.transpose(1,2)
    idx = dis_mat.topk(k=k, dim=-1)[1]
    idx = idx.int()
    return idx

def get_edge_feature(fea, idx):
    """
    inputs:
        fea: (B, C, N) tensor
        idx: (B, N, k) tensor
    return:
        edge: (B, C*2, N, k) tensor
    """
    fea_knn = utils.group_points(fea, idx)
    fea = fea.unsqueeze(3).repeat(1, 1, 1, fea_knn.shape[3])
    fea_knn = torch.cat([fea, fea_knn], dim=1)
    return fea_knn

def get_edge_point(xyz, idx):
    """
    inputs:
        xyz: (B, 3, N) tensor
        idx: (B, N, k) tensor
    return:
        edge: (B, 10, N, k) tensor
    """
    xyz_knn = utils.group_points(xyz, idx)
    xyz = xyz.unsqueeze(3).repeat(1, 1, 1, xyz_knn.shape[3])
    xyz_r = xyz - xyz_knn
    xyz_d = xyz_r.square().sum(dim=1, keepdim=True).sqrt()
    xyz_knn = torch.cat([xyz_d, xyz, xyz_knn, xyz_r], dim=1)
    return xyz_knn

class GA_Edgeconv(nn.Module):
    def __init__(self, in_channel, out_channel, k=16, bn=False):
        super(GA_Edgeconv, self).__init__()
        self.k = k
        self.mf1 = conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1), bn=bn)
        self.mf2 = conv2d(out_channel, out_channel, kernel_size=(1,1), stride=(1,1), bn=bn)
        self.mf3 = conv2d(out_channel, out_channel, kernel_size=(1,1), stride=(1,1), bn=bn)
        self.mg1 = conv2d(10, out_channel, kernel_size=(1,1), stride=(1,1), bn=bn)
        self.mg2 = conv2d(out_channel, out_channel, kernel_size=(1,1), stride=(1,1), bn=bn)
        
    def forward(self, xyz, fea=None):
        if fea is None:
            fea = xyz
        idx = knn(fea, self.k)
        xyz = get_edge_point(xyz, idx)
        fea = get_edge_feature(fea, idx)
        xyz = self.mg1(xyz)
        xyz = self.mg2(xyz)
        fea = self.mf1(fea)
        fea = self.mf2(fea)
        fea = self.mf3(fea)
        fea = torch.mul(xyz, fea)
        fea = fea.max(dim=-1)[0]
        return fea

class GADGCNN(nn.Module):
    def __init__(self, num_class, k=16, bn=False):
        super(GADGCNN, self).__init__()
        self.edgeconv1 = GA_Edgeconv(3*2, 32, k=k, bn=bn)
        self.edgeconv2 = GA_Edgeconv(32*2, 128, k=k, bn=bn)
        self.conv1 = conv1d(128, 128, kernel_size=1, stride=1, bn=bn)
        self.conv2 = conv1d(128, 1024, kernel_size=1, stride=1, bn=bn)
        self.conv3 = conv1d(1184, 512, kernel_size=1, stride=1, bn=bn)
        self.conv4 = conv1d(512, 256, kernel_size=1, stride=1, bn=bn)
        self.conv5 = conv1d(256, 128, kernel_size=1, stride=1, bn=bn)
        self.conv6 = conv1d(128, num_class, kernel_size=1, stride=1, act_fn=None, bn=bn)
        self.dp1 = nn.Dropout(0.4)
        self.dp2 = nn.Dropout(0.4)
        self.dp3 = nn.Dropout(0.4)
        
    def forward(self, xyz, fea=None):
        xyz = xyz.transpose(1,2).contiguous() #(B, 3, N)
        if fea is not None:
            fea = fea.transpose(1,2).contiguous() #(B, C, N)
        fea1 = self.edgeconv1(xyz, fea) #(B, 32, N)
        fea2 = self.edgeconv2(xyz, fea1) #(B, 128, N)
        fea = self.conv1(fea2) #(B, 128, N)
        fea = self.conv2(fea) #(B, 1024, N)
        fea = fea.max(dim=-1, keepdim=True)[0] #(B, 1024, 1)
        fea = fea.repeat(1, 1, xyz.shape[2]) #(B, 1024, N)
        fea = torch.cat([fea1, fea2, fea], dim=1) #(B, 1184, N)
        fea = self.conv3(fea) #(B, 512, N)
        fea = self.dp1(fea)
        fea = self.conv4(fea) #(B, 256, N)
        fea = self.dp2(fea)
        fea = self.conv5(fea) #(B, 128, N)
        fea = self.dp3(fea)
        fea = self.conv6(fea) #(B, num_class, N)
        return fea
    