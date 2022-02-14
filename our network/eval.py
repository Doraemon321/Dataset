# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:29:02 2021

@author: PQD
"""

import torch
import torch.nn as nn
import numpy as np
from get_data import get_train_data, get_test_data
from models.modelD import SADGCNN
import h5py
import os

NUM_CLASS = 7
NUM_POINT = 2048
MODEL_NAME = 'modelD real dataset fine-tuned'
DATA_FILE = os.path.dirname(os.getcwd()) + '\\data\\data_approach.h5'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

seg = SADGCNN(NUM_CLASS, bn=True).to(device)
seg.load_state_dict(torch.load('trained/'+MODEL_NAME+'.pt'))
seg.eval()

f = h5py.File(DATA_FILE, 'r')
data_ = f['data'][:, :, :3]
scene_ = f['scene'][:]
f.close()

color_select = np.array([[0.0, 0.0, 0.0],
                         [0.0, 0.67, 0.0],
                         [0.0, 1.0, 0.0],
                         [1.0, 0.67, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, 1.0, 1.0]])
folder = os.path.exists('eval/'+MODEL_NAME)
if not folder:
    os.makedirs('eval/'+MODEL_NAME)
for i in range(data_.shape[0]):
    data = data_[i, :, :]
    
#    data = np.random.permutation(data)
    data = data[0:NUM_POINT,:]
    pc = torch.from_numpy(data).to(device).float()
    pc = pc.view(1, -1, 3).contiguous()
    scene = scene_[i, :, :]
    scene = np.random.permutation(scene)
    scene = scene[0:NUM_POINT,:]
    scene = torch.from_numpy(scene).to(device).float()
    scene = scene.view(1, -1, 3).contiguous()
    
    pred = seg(pc, scene)
    
    pred = pred.squeeze()
    pred = pred.t()
    label = pred.max(dim=1)[1].cpu()
    label = label.numpy().astype(np.int)
    color = color_select[label, :]
    points = np.concatenate([pc.squeeze().cpu().numpy(), color], axis=-1)
    np.savetxt('eval/'+MODEL_NAME+'/'+str(i)+'.txt', points, fmt='%.6f')
