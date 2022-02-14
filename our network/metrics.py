# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:29:02 2021

@author: PQD
"""

import torch
import torch.nn as nn
import numpy as np
from models.modelD import SADGCNN
import h5py
import modules
import os

NUM_CLASS = 7
NUM_POINT = 2048
MODEL_NAME = 'modelD real dataset fine-tuned'
DATA_FILE = 'data_diffi'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

seg = SADGCNN(NUM_CLASS, bn=True).to(device)
seg.load_state_dict(torch.load('trained/'+MODEL_NAME+'.pt'))
seg.eval()

filename = os.path.join(os.path.dirname(os.getcwd()), 'data\\'+DATA_FILE+'.h5')
f = h5py.File(filename, 'r')
data_ = f['data'][:]
label_ = f['label'][:]
scene_ = f['scene'][:]
f.close()

correct = 0.
total = 0.
inter = 0.
union = 0.
for i in range(data_.shape[0]):
    data = data_[i, :, :]
    label = label_[i, :]
    
#    permutation = np.random.permutation(data.shape[0])
#    permutation = permutation[0:NUM_POINT]
#    data = data[permutation, :]
#    label = label[permutation]
    data = data[0:NUM_POINT, :]
    label = label[0:NUM_POINT]
    pc = torch.from_numpy(data).to(device).float()
    pc = pc.view(1, -1, 3).contiguous()
    label = torch.from_numpy(label).to(device).int()
    label = label.view(1, -1).contiguous()
    scene = scene_[i, :, :]
#    scene = np.random.permutation(scene)
    scene = scene[0:NUM_POINT,:]
    scene = torch.from_numpy(scene).to(device).float()
    scene = scene.view(1, -1, 3).contiguous()
    
    pred = seg(pc, scene)
    
    pred = pred.squeeze()
    pred = pred.t()
    pred_class = pred.max(dim=1)[1]
    color = pred_class.cpu().numpy().astype(np.float64)
    label = label.squeeze()
#    mask = torch.nonzero(label, as_tuple=True)
#    label = label[mask]
#    pred_class = pred_class[mask]
    correct += pred_class.eq(label.view_as(pred_class)).sum().item()
    total += label.shape[0]
    temp_i, temp_u = modules.batch_intersection_union(pred_class, label, NUM_CLASS)
    inter += temp_i
    union += temp_u
    
    color = color.reshape([-1,1])
    points = np.concatenate([pc.squeeze().cpu().numpy(), color], axis=-1)
correct /= total
#mask = torch.nonzero(union, as_tuple=True)

mask = torch.Tensor([0, 1, 2, 3, 4]).to(device).long()
inter = inter[mask]
union = union[mask]
iou = inter / union
mean_iou = iou.mean()

print('accuracy: ', correct)
print('iou: ', iou * 100)
print('miou: ', mean_iou * 100)
