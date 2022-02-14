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

def knn(a, b, k):
    """
    inputs:
        a: (B, C, N1) tensor
        b: (B, C, N2) tensor
        k: int
    return:
        idx: (B, N1, k) tensor
    """
    inner = -2 * torch.matmul(a.transpose(1,2), b)
    a2 = torch.sum(a**2, dim=1, keepdim=True)
    b2 = torch.sum(b**2, dim=1, keepdim=True)
    dis_mat = -a2.transpose(1,2) - inner - b2
    idx = dis_mat.topk(k=k, dim=-1)[1]
    idx = idx.int()
    return idx

def get_edge_feature(fea1, fea2, idx):
    """
    inputs:
        fea1: (B, C, N) tensor
        fea2: (B, C, N) tensor
        idx: (B, N, k) tensor
    return:
        edge: (B, C*2, N, k) tensor
    """
    fea_knn = utils.group_points(fea2, idx)
    fea = fea1.unsqueeze(3).repeat(1, 1, 1, fea_knn.shape[3])
    fea_knn = torch.cat([fea, fea_knn], dim=1)
    return fea_knn

def get_edge_point(xyz1, xyz2, idx):
    """
    inputs:
        xyz1: (B, 3, N) tensor
        xyz2: (B, 3, N) tensor
        idx: (B, N, k) tensor
    return:
        edge: (B, 10, N, k) tensor
    """
    xyz_knn = utils.group_points(xyz2, idx)
    xyz = xyz1.unsqueeze(3).repeat(1, 1, 1, xyz_knn.shape[3])
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
        idx = knn(fea, fea, self.k)
        xyz = get_edge_point(xyz, xyz, idx)
        fea = get_edge_feature(fea, fea, idx)
        xyz = self.mg1(xyz)
        xyz = self.mg2(xyz)
        fea = self.mf1(fea)
        fea = self.mf2(fea)
        fea = self.mf3(fea)
        fea = torch.mul(xyz, fea)
        fea = fea.max(dim=-1)[0]
        return fea

class SA_module(nn.Module):
    def __init__(self, in_channel, out_channel, k=4, bn=False):
        super(SA_module, self).__init__()
        self.k = k
        self.mf = conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1), bn=bn)
        self.mg = conv2d(10, out_channel, kernel_size=(1,1), stride=(1,1), bn=bn)
    
    def forward(self,xyz,  xyz_s, fea, fea_s):
        idx = knn(xyz, xyz_s, self.k)
        xyz = get_edge_point(xyz, xyz_s, idx)
        fea = get_edge_feature(fea, fea_s, idx)
        xyz = self.mg(xyz)
        fea = self.mf(fea)
        fea = torch.mul(xyz, fea)
        fea = fea.max(dim=-1)[0]
        return fea

class SADGCNN(nn.Module):
    def __init__(self, num_class, k1=16, k2=4, bn=False):
        super(SADGCNN, self).__init__()
        self.edgeconv1 = GA_Edgeconv(3*2, 32, k=k1, bn=bn)
        self.edgeconv2 = GA_Edgeconv(32*2, 128, k=k1, bn=bn)
        self.edgeconv2_s = GA_Edgeconv(32*2, 128, k=k1, bn=bn)
        self.sa1 = SA_module(32*2, 32, k=k2, bn=bn)
        self.sa2 = SA_module(128*2, 256, k=k2, bn=bn)
        self.conv1 = conv1d(256, 256, kernel_size=1, stride=1, bn=bn)
        self.conv2 = conv1d(256, 1024, kernel_size=1, stride=1, bn=bn)
        self.conv3 = conv1d(1312, 512, kernel_size=1, stride=1, bn=bn)
        self.conv4 = conv1d(512, 256, kernel_size=1, stride=1, bn=bn)
        self.conv5 = conv1d(256, 128, kernel_size=1, stride=1, bn=bn)
        self.conv6 = conv1d(128, num_class, kernel_size=1, stride=1, act_fn=None, bn=bn)
        self.dp1 = nn.Dropout(0.4)
        self.dp2 = nn.Dropout(0.4)
        self.dp3 = nn.Dropout(0.4)
        
    def forward(self, xyz, xyz_s):
        xyz = xyz.transpose(1,2).contiguous() #(B, 3, N)
        xyz_s = xyz_s.transpose(1,2).contiguous() #(B, 3, N)
        fea1 = self.edgeconv1(xyz) #(B, 32, N)
        fea1_s = self.edgeconv1(xyz_s) #(B, 32, N)
        fea1 = self.sa1(xyz, xyz_s, fea1, fea1_s) #(B, 32, N)
        fea2 = self.edgeconv2(xyz, fea1) #(B, 128, N)
        fea2_s = self.edgeconv2_s(xyz_s, fea1_s) #(B, 128, N)
        fea2 = self.sa2(xyz, xyz_s, fea2, fea2_s) #(B, 256, N)
        fea = self.conv1(fea2) #(B, 256, N)
        fea = self.conv2(fea) #(B, 1024, N)
        fea = fea.max(dim=-1, keepdim=True)[0] #(B, 1024, 1)
        fea = fea.repeat(1, 1, xyz.shape[2]) #(B, 1024, N)
        fea = torch.cat([fea1, fea2, fea], dim=1) #(B, 1312, N)
        fea = self.conv3(fea) #(B, 512, N)
        fea = self.dp1(fea)
        fea = self.conv4(fea) #(B, 256, N)
        fea = self.dp2(fea)
        fea = self.conv5(fea) #(B, 128, N)
        fea = self.dp3(fea)
        fea = self.conv6(fea) #(B, num_class, N)
        return fea
    