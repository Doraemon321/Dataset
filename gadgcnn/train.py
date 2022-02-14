# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:47:33 2020

@author: PQD
"""

import torch
import torch.nn as nn
import torch.optim as optim
import modules
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR, StepLR
from model import GADGCNN
from get_data import get_train_data, get_test_data, processing
import time
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 8
NUM_POINT = 2048
MAX_EPOCH = 500
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_CLASS = 7
EARLYSTOP_EPOCH = 30
DATASET = 'virtual' #or 'real', 'difficult'
trained_epoch = 0
weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
model_name = 'virtual dataset'

log_out = open('log/'+model_name+'.txt', 'w')
def log_write(string):
    log_out.write(string+'\n')
    log_out.flush()
    print(string)

train_point, train_label = get_train_data(NUM_POINT, DATASET)
test_point, test_label = get_test_data(NUM_POINT, DATASET)
train_point = torch.from_numpy(train_point)
train_label = torch.from_numpy(train_label)
test_point = torch.from_numpy(test_point)
test_label = torch.from_numpy(test_label)
train_dataset = TensorDataset(train_point, train_label)
test_dataset = TensorDataset(test_point, test_label)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

seg = GADGCNN(NUM_CLASS, bn=True).to(device)
if DATASET != 'virtual':
    seg.load_state_dict(torch.load('trained/virtual dataset.pt'))

#def weights_init(m):
#    classname = m.__class__.__name__
#    if classname.find('Conv') != -1:
#        nn.init.xavier_normal_(m.weight.data)
#        nn.init.constant_(m.bias.data, 0)
#    elif classname.find('BatchNorm') != -1:
#        nn.init.constant_(m.weight.data, 1)
#        nn.init.constant_(m.bias.data, 0)
#    elif classname.find('Linear') != -1:
#        nn.init.xavier_normal_(m.weight.data)
#        nn.init.constant_(m.bias.data, 0)
#        
#seg.apply(weights_init)

#print(seg)
total_params = sum(p.numel() for p in seg.parameters())
log_write('Total parameters: %d.' %(total_params))
total_trainable_params = sum(p.numel() for p in seg.parameters() if p.requires_grad)
log_write('Training parameters: %d.' %(total_trainable_params))
log_write('Cross entropy loss weight')
log_write(str(weight.cpu().numpy()))

#optimizer = optim.SGD(seg.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
optimizer = optim.Adam(seg.parameters(), lr=LEARNING_RATE)
#scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1./(1e-2*epoch+1), last_epoch=-1)
scheduler = StepLR(optimizer, step_size=5, gamma=0.7, last_epoch=-1)

loss_fn = nn.CrossEntropyLoss(weight=weight)

best_test_acc = 0.
best_acc_epoch = 0
best_test_loss = 100.
best_loss_epoch = 0
best_test_miou = 0.
count = 0
acc = []
loss = []
miou = []

def train(model, device, train_dataloader, optimizer):
    model.train()
    train_loss = 0.
    correct = 0.
    inter = 0.
    union = 0.
    start = time.time()
    for i, (point, label) in enumerate(train_dataloader):
        point, label = point.to(device).float(), label.to(device).long()
        point = processing(point)
        pred = model(point)
        loss = loss_fn(pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch = point.size(0)
        pred_class = pred.max(dim=1)[1]
        train_loss += loss.item() * batch
        correct += pred_class.eq(label.view_as(pred_class)).sum().item()
        temp_i, temp_u = modules.batch_intersection_union(pred_class, label, NUM_CLASS)
        inter += temp_i
        union += temp_u
    train_loss /= len(train_dataloader.dataset)
    correct /= len(train_dataloader.dataset) * NUM_POINT
    mean_iou = inter / union
    mean_iou = mean_iou.mean()
    end = time.time()
    outstr = 'Train loss: %.6f, train acc: %.6f, train miou: %.6f, time used: %.1f s' %(train_loss, correct, mean_iou, end-start)
    log_write(outstr)

def test(model, device, test_dataloader):
    model.eval()
    test_loss = 0.
    correct = 0.
    inter = 0.
    union = 0.
    start = time.time()
    with torch.no_grad():
        for i, (point, label) in enumerate(test_dataloader):
            point, label = point.to(device).float(), label.to(device).long()
            pred = model(point)
            loss = loss_fn(pred, label)
            
            batch = point.size(0)
            pred_class = pred.max(dim=1)[1]
            test_loss += loss.item() * batch
            correct += pred_class.eq(label.view_as(pred_class)).sum().item()
            temp_i, temp_u = modules.batch_intersection_union(pred_class, label, NUM_CLASS)
            inter += temp_i
            union += temp_u
    test_loss /= len(test_dataloader.dataset)
    correct /= len(test_dataloader.dataset) * NUM_POINT
    mean_iou = inter / union
    mean_iou = mean_iou.mean()
    end = time.time()
    outstr = 'Test loss: %.6f, test acc: %.6f, test miou: %.6f, time used: %.1f s' %(test_loss, correct, mean_iou, end-start)
    log_write(outstr)
    return correct, test_loss, mean_iou

for epoch in range(MAX_EPOCH - trained_epoch):
    log_write('--------' + str(epoch + trained_epoch) + '--------')
    train(seg, device, train_dataloader, optimizer)
    test_acc, test_loss, test_miou = test(seg, device, test_dataloader)
    acc.append(test_acc)
    loss.append(test_loss)
    miou.append(test_miou)
    scheduler.step()
    if test_acc >= best_test_acc:
        best_test_acc = test_acc
        best_test_miou = test_miou
        best_acc_epoch = epoch + trained_epoch
        torch.save(seg.state_dict(), 'trained/'+model_name+'.pt')
        log_write('Saved best.')
    if test_loss <= best_test_loss:
        best_test_loss = test_loss
        best_loss_epoch = epoch + trained_epoch
        count = 0
    else:
        count += 1
    if count==EARLYSTOP_EPOCH:
        break

log_write('Best loss: %.6f in %d epoch, best acc: %.6f and miou: %.6f in %d epoch' 
          %(best_test_loss, best_loss_epoch, best_test_acc, best_test_miou, best_acc_epoch))
log_out.close()
loss = np.array(loss).reshape(-1, 1)
acc = np.array(acc).reshape(-1, 1) * 100
miou = np.array(miou).reshape(-1, 1) * 100
log = np.concatenate([loss, acc, miou], axis=-1)
np.savetxt('acc/'+model_name+'.txt', log, fmt='%.6f')