# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:57:32 2020

@author: dell
"""

import numpy as np
import h5py
import torch
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

filepath = os.path.dirname(os.getcwd())
def getDataFiles(list_filename):
    return [filepath+'\\'+line.rstrip() for line in open(list_filename)]

def load(h5_filename, num_points):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:, 0:num_points, :]
    label = f['label'][:, 0:num_points]
    f.close()
    return (data, label)

TRAIN_FILES_VIRTUAL = getDataFiles(filepath+'\\data\\train_files.txt')
TEST_FILES_VIRTUAL = getDataFiles(filepath+'\\data\\test_files.txt')
TRAIN_FILES_REAL = getDataFiles(filepath+'\\data\\train_files_r.txt')
TEST_FILES_REAL = getDataFiles(filepath+'\\data\\test_files_r.txt')
TRAIN_FILES_DIFFICULT = getDataFiles(filepath+'\\data\\train_files_diffi.txt')
TEST_FILES_DIFFICULT = getDataFiles(filepath+'\\data\\test_files_diffi.txt')

def get_data(file_names, num_points):
    last_data,last_label=load(file_names[0], num_points)
    if len(file_names) > 1:
        for n in range(len(file_names)-1):
            now_data,now_label=load(file_names[n+1], num_points)
            last_data=np.concatenate((last_data,now_data),axis=0)
            last_label=np.concatenate((last_label,now_label),axis=0)
    return (last_data,last_label)

def get_train_data(num_points, dataset):
    if dataset == 'virtual':
        train_data,train_label=get_data(TRAIN_FILES_VIRTUAL, num_points)
    if dataset == 'real':
        train_data,train_label=get_data(TRAIN_FILES_REAL, num_points)
    if dataset == 'difficult':
        train_data,train_label=get_data(TRAIN_FILES_DIFFICULT, num_points)
    train_data=train_data[:,0:num_points,:]
    train_label=train_label[:,0:num_points]
    return (train_data,train_label)
    

def get_test_data(num_points, dataset):
    if dataset == 'virtual':
        test_data,test_label=get_data(TEST_FILES_VIRTUAL, num_points)
    if dataset == 'real':
        test_data,test_label=get_data(TEST_FILES_REAL, num_points)
    if dataset == 'difficult':
        test_data,test_label=get_data(TEST_FILES_DIFFICULT, num_points)
    test_data=test_data[:,0:num_points,:]
    test_label=test_label[:,0:num_points]
    return (test_data,test_label)
    
def rotate(pc):
    rotated = torch.zeros_like(pc).to(device)
    for i in range(pc.shape[0]):
        angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rotation_matrix = torch.tensor([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]]).float().to(device)
        temp = pc[i,...]
        rotated[i,...] = torch.matmul(temp, rotation_matrix)
    return rotated

def jitter(pc, sigma=0.01, clip=0.03):
    random = torch.randn(pc.size()) * sigma
    jitterd = random.clamp(-clip, clip).to(device)
    jitterd += pc
    return jitterd

def shift(pc, shift_range=0.05):
    random = torch.rand(pc.size()) * shift_range * 2 - shift_range
    random = random.to(device)
    shifted = random + pc
    return shifted

def processing(pc):
#    pc = rotate(pc)
    pc = jitter(pc)
    pc = shift(pc)
    return pc

if __name__=='__main__':
    test_data,test_label=get_train_data(1024, 'virtual')
    print(test_data.shape)
    print(test_label.shape)
#    print(test_obs.shape)
            
