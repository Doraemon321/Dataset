# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:10:50 2021

@author: PQD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import ext

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

class sample_and_group(nn.Module):
    def __init__(self, npoint, radius, nsample, use_xyz=True):
        super(sample_and_group, self).__init__()
        self.npoint, self.radius, self.nsample, self.use_xyz = npoint, radius, nsample, use_xyz
    
    def forward(self, xyz, features):
        """
        inputs:
            xyz: (B, 3, N) tensor
            features: (B, C, N) tensor or None
        return:
            new_xyz: (B, 3, npoint)
            new_features: (B, C+3, npoint, nsample) or (B, C, npoint, nsample) tensor
            idx: (B, npoint, nsample) tensor
            grouped_xyz: (B, 3, npoint, nsample) tensor
        """
        xyz_trans = xyz.transpose(1, 2).contiguous()    #(B, N, 3)
        new_xyz = utils.gather_points(xyz, utils.furthest_point_sampling(xyz_trans, self.npoint))  #(B, 3, m)
        idx = utils.ball_query(new_xyz.transpose(1, 2).contiguous(), xyz_trans, self.nsample, self.radius)
        grouped_xyz = utils.group_points(xyz, idx)
        grouped_xyz -= new_xyz.unsqueeze(3).repeat(1,1,1,self.nsample)
        if features is not None:
            grouped_features = utils.group_points(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        return new_xyz, new_features

class pointnet_sa_module(nn.Module):
    def __init__(self, npoint, radius, nsample, mlp, use_xyz=True, bn=False):
        super(pointnet_sa_module, self).__init__()
        self.sample = sample_and_group(npoint, radius, nsample, use_xyz)
        self.mlps = nn.ModuleList()
        if use_xyz:
            mlp[0][0] += 3
        for _, mlp_channel in enumerate(mlp):
            self.mlps.append(conv2d(mlp_channel[0], mlp_channel[1], kernel_size=(1,1), stride=(1,1), bn=bn))
        
    def forward(self, xyz, features):
        """
        inputs:
            xyz: (B, 3, N) tensor
            features: (B, C, N) tensor
        return:
            new_xyz: (B, 3, npoint) tensor
            new_feature: (B, mlp[-1][-1], npoint) tensor
        """
        new_xyz, new_features = self.sample(xyz, features)
        for i in range(len(self.mlps)):
            new_features = self.mlps[i](new_features)
        new_features = new_features.max(dim=3)[0]
        return new_xyz, new_features

class pointnet_fp_module(nn.Module):
    def __init__(self, mlp, bn=False):
        super(pointnet_fp_module, self).__init__()
        if mlp is not None:
            self.mlps = nn.ModuleList()
            for _, mlp_channel in enumerate(mlp):
                self.mlps.append(conv1d(mlp_channel[0], mlp_channel[1], kernel_size=1, stride=1, bn=bn))
        else:
            self.mlps = None
    
    def forward(self, xyz1, xyz2, features1, features2):
        """
        inputs:
            xyz1: (B, 3, n1) tensor
            xyz2: (B, 3, n2) tensor, n2 < n1
            features1: (B, c1, n1) tensor or None
            features2: (B, c2, n2) tensor
        return:
            new_features1: (B, mlp[-1][-1], n1) tensor
        """
        dist, idx = utils.three_nn(xyz1, xyz2)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_features = utils.three_interpolate(features2, idx, weight)
        if features1 is not None:
            new_features1 = torch.cat([interpolated_features, features1], dim=1)
        else:
            new_features1 = interpolated_features
        if self.mlps is not None:
            for i in range(len(self.mlps)):
                new_features1 = self.mlps[i](new_features1)
        return new_features1

class self_attention_module(nn.Module):
    def __init__(self, in_channel, temp_channel, out_channel, merge='concat', act_fn=nn.ReLU(inplace=True)):
        super(self_attention_module, self).__init__()
        self.Wq = nn.Conv1d(in_channel, temp_channel, kernel_size=1, stride=1, bias=False)
        self.Wk = nn.Conv1d(in_channel, temp_channel, kernel_size=1, stride=1, bias=False)
        self.Wv = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.merge = merge
        self.activation = act_fn
        
    def forward(self, fea):
        """
        inputs:
            fea: (B, in_channel, N) tensor
        return:
            new_fea: (B, out_channel, N) tensor
        """
        Q = self.Wq(fea)
        K = self.Wk(fea)
        V = self.Wv(fea)
        S = torch.matmul(K.transpose(1, 2), Q)
        S = F.softmax(S, dim=1)
        new_fea = torch.matmul(V, S)
        if self.merge == 'concat':
            new_fea = torch.cat([new_fea, fea], dim=1)
        elif self.merge == 'add':
            new_fea = new_fea + fea
        if self.activation is not None:
            new_fea = self.activation(new_fea)
        return new_fea

class LAEconv(nn.Module):
    def __init__(self, in_channel, out_channel, radius, m, act_fn=nn.ReLU(inplace=True), bn=False):
        super(LAEconv, self).__init__()
        self.W = nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1), bias=False)
        self.altha = nn.Conv2d(out_channel, 1, kernel_size=(1,1), stride=(1,1), bias=False)
        self.mlp = conv1d(out_channel, out_channel, kernel_size=1, stride=1, act_fn=act_fn, bn=bn)
        self.radius = radius
        self.m = m
        
    def forward(self, xyz, fea=None):
        if fea is None:
            fea = xyz
        xyz_trans = xyz.transpose(1, 2).contiguous()
        idx = utils.multi_directional_knn(xyz_trans, self.radius, self.m)
        grouped_fea = utils.group_points(fea, idx)
        grouped_fea_r = grouped_fea - fea.unsqueeze(3).repeat(1,1,1,self.m*16)
        weight = self.W(grouped_fea_r)
        weight = F.softmax(self.altha(weight), dim=-1)
        grouped_fea = self.W(grouped_fea)
        grouped_fea = weight * grouped_fea
        new_fea = torch.sum(grouped_fea, dim=-1)
        new_fea = self.mlp(new_fea)
        return new_fea

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
    dis, idx = dis_mat.topk(k=k, dim=-1)
    return dis, idx

def batch_intersection_union(pred, label, num_class):
    pred += 1
    label += 1
    intersection = pred * ((pred == label).long())
    area_inter = torch.histc(intersection.float(), bins=num_class, min=1, max=num_class)
    area_pred = torch.histc(pred.float(), bins=num_class, min=1, max=num_class)
    area_label = torch.histc(label.float(), bins=num_class, min=1, max=num_class)
    area_union = area_pred + area_label - area_inter
    return area_inter, area_union