# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:15:39 2020

@author: dell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
import ext

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        """
        inputs:
            xyz: (B, N, 3) tensor, N>npoint
            npoint: int
        return:
            idx: (B, npoint) tensor
        """
        output = ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        return ()

furthest_point_sampling = FurthestPointSampling.apply

class GatherPoints(Function):
    @staticmethod
    def forward(ctx, points, idx):
        """
        inputs:
            points: (B, C, N) tensor
            idx: (B, npoint) tensor
        return: 
            gathered: (B, C, npoint) tensor
        """
        ctx.save_for_backward(idx, points)
        output = ext.gather_points(points, idx)
        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        idx, points = ctx.saved_tensors
        N = points.size(2)
        grad = ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad, None

gather_points = GatherPoints.apply

class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        """
        inputs:
            unknown: (B, 3, n) tensor
            known: (B, 3, m) tensor
        return: 
            dist: (B, n, 3) tensor
            idx: (B, n, 3) tensor
        """
        unknown = unknown.transpose(1, 2).contiguous()
        known = known.transpose(1, 2).contiguous()
        dist2, idx = ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)
        ctx.mark_non_differentiable(dist, idx)
        return dist, idx
    
    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()

three_nn = ThreeNN.apply

class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        """
        inputs:
            features: (B, C, m) tensor
            idx: (B, n, 3) tensor
            weight: (B, n, 3) tensor
        return:
            new_features: (B, C, n) tensor
        """
        ctx.save_for_backward(idx, weight, features)
        output = ext.three_interpolate(features, idx, weight)
        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)
        features_grad = ext.three_interpolate_grad(grad_out.contiguous(), idx, weight, m)
        return features_grad, torch.zeros_like(idx), torch.zeros_like(weight)

three_interpolate = ThreeInterpolate.apply

class GroupPoints(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        inputs:
            features: (B, C, n) tensor
            idx: (B, npoint, nsample) tensor
        return:
            new_features: (B, C, npoint, nsample) tensor
        """
        ctx.save_for_backward(idx, features)
        return ext.group_points(features, idx)
    
    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)
        features_grad = ext.group_points_grad(grad_out.contiguous(), idx, N)
        return features_grad, torch.zeros_like(idx)

group_points = GroupPoints.apply

class BallQuery(Function):
    @staticmethod
    def forward(ctx, new_xyz, xyz, nsample, radius):
        """
        inputs:
            new_xyz: (B, npoint, 3) tensor
            xyz:(B, N, 3) tensor
            nsample: int
            radius: float
        return:
            idx: (B, npoint, nsample) tensor
        """
        output = ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        return ()

ball_query = BallQuery.apply

class MultiDirectionalKNN(Function):
    @staticmethod
    def forward(ctx, xyz, radius, m):
        """
        inputs:
            xyz: (B, N, 3) tensor
            radius: float
            nsample: int
        return:
            idx: (B, N, m) tensor
        """
        idx = ext.multi_directional_knn(xyz, radius, m)
        ctx.mark_non_differentiable(idx)
        return idx
    
    @staticmethod
    def backward(ctx, grad_out):
        return ()

multi_directional_knn = MultiDirectionalKNN.apply