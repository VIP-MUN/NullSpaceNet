#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov. 11 20:26:58 2020

@author: abdelpakey
"""

"""
This is the implementation of the paper 
"NullSpaceNet: Nullspace Convoluional Neural Network with Differentiable Loss Function"
submitted to ECCV2020
"""

import argparse
import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from torchvision import transforms,models
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from itertools import product
from collections import OrderedDict
from collections import namedtuple
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import numpy as np
import math
import pdb
import scipy.linalg as la
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from helpers import *
from sklearn.manifold import TSNE
import time
import seaborn as sns
from torch.optim import lr_scheduler
from collections import OrderedDict
from collections import namedtuple
from itertools import product



class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
torch.manual_seed(30)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')           
# model = Deep_w_classifier().to(device)
# model= vgg_no_classifier().to(device)
# model= vgg_w_classifier().to(device)
model =vgg_two_parts().to(device)
model.load_state_dict(torch.load('./model/NullSpaceNet_only.pth'))
print("number of params: ", sum(p.numel() for p in model.parameters()))
params= OrderedDict(
        # For pretrained VGG(weather the network gradient = True or False) set lr=1e-4, for not-pretrined (e.g., train from scratch) set lr=1e-3.
        lr= [1e-4]#1e-4
        ,train_batch_size=[280]#350
        ,test_batch_size= [300] # 300
        
        ,shuffle=[True]
        ,weight_decay=[1e-5] #1e-5
          
        
        , epochs= [500]
        
        )



#dataset and dataloader
transform = transforms.Compose(
    [   #transforms.Resize(96),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                         [0.229, 0.224, 0.225])  # Imagenet standards
      ]   )
dataset= "stl10"

criterion_cross_entropy = nn.CrossEntropyLoss()
print("Params to learn:")
params_to_update = []
for name,param in model.named_parameters():
       # print(name,param.requires_grad)
   
       if param.requires_grad == True:
           params_to_update.append(param)
           print("\t",name)

sklearn_lda = LDA(n_components=None)
tb = SummaryWriter()
for run  in RunBuilder.get_runs(params):
    comment ='Requires_grad: {} - Config: {} '.format( param.requires_grad, {run}) #f'-{run, param.requires_grad}'
    # For pretrained VGG(weather the network gradient = True or False) use Adam optimizer
    tb = SummaryWriter(comment=comment)
    optimizer = optim.Adam(model.parameters(), lr=run.lr,weight_decay=run.weight_decay) # starts from 10^-2 : -5 annealed gemtercially
    # For pretrained VGG(weather the network gradient = True or False) set step_size=1 gamma=0.99, for not-pretrined (e.g., train from scratch) set step_size=10.
    scheduler= lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.99,last_epoch=-1)
    if dataset == "imagenet":
        import torchvision.datasets as datasets
        input_transform = None
        datasets.imagenet.ARCHIVE_META['devkit']= "https://github.com/goodclass/PythonAI/raw/master/imagenet/ILSVRC2012_devkit_t12.tar.gz"
        
        imagenet_data = datasets.ImageNet('/home/abdelpakey/Desktop/ILSVRC2017_DET/ILSVRC/Data/DET/')
        data_loader =DataLoader(imagenet_data,
                                                  batch_size=200,
                                                  shuffle=True,
                                                  num_workers=18)
    
        
        train_dataset=datasets.ImageNet(root='./dataset', split='train',  transform=transform, download=True)
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=run.train_batch_size, num_workers=10,drop_last=True)

        test_dataset=datasets.ImageNet(root='./dataset', split='test',  transform=transform, download=False)
        test_loader = DataLoader(train_dataset, shuffle=False, batch_size=run.train_batch_size, num_workers=10,drop_last=True)
    
    if dataset == "stl10":

        train_dataset=datasets.STL10(root='./dataset', split='train',  transform=transform, download=True)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=run.train_batch_size, num_workers=10,drop_last=True)  #2500
    
        test_dataset = datasets.STL10(root='./dataset', split='test', download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=run.test_batch_size, num_workers=8)
    
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=run.train_batch_size, num_workers=10,drop_last=True)  #2500
    
        test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=run.test_batch_size, num_workers=8)  #300
  
    if dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=run.train_batch_size, num_workers=10,drop_last=True)  #2500
    
        test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=run.test_batch_size, num_workers=8)  #300
    
    # print("Lr: ", run.lr,"Btach_size: ",run.batch_size, "Shuffle: ",run.shuffle, "Beat3: ",run.beta3)
    loss_entropy=None
    
    for epoch in range(0,run.epochs):
            model.train()
            running_loss=0.0
            acc=0.0
            loss=0
            loss_entropy=0
            for i, batch in enumerate (train_loader):
                imgs = batch[0].to(device)
                labels = batch[1].to(device)
                optimizer.zero_grad()
                conv,out_simple= model(imgs)
                loss=loss_nullspacenet( conv.unsqueeze(2).unsqueeze(3),labels)
                # loss = criterion_cross_entropy(out_simple, labels)#+reg
                loss.backward( )
                optimizer.step()
                running_loss += loss.item()*imgs.size(0)
            if epoch%20==0:
                    
                if loss:
                    loss_config="NSFT"                    
                if loss_entropy:
                    loss_config="CEL"
                if loss_entropy and loss:
                    loss_config="CEL_NSFT"
                torch.save(model.state_dict(),"./model/%s_%s_%s_BatchSize%d_Epoch%d.pth" % ("loss_config", "is_classifier",dataset,run.train_batch_size,epoch+1))

            epoch_loss = running_loss/ len(train_dataset )
            tb.add_scalar('Loss', epoch_loss,epoch)

            print('Epoch: {} Loss: {:.8f} '.format( epoch,epoch_loss))
            scheduler.step()
            print('Current learning rate is :',  [group['lr'] for group in optimizer.param_groups])
            
            # To calculate the accuracy in case of attached CCEL (with FC) uncomment the block below
            # total = 0
            # correct = 0
            # t=time.time()
            # model.load_state_dict(torch.load('./model/NullSpaceNet_only.pth'))
            # with torch.no_grad():
            #     model.eval()
            #     for data in test_loader:
            #         images, labels = data
            #         _,outputs = model(images.cuda())
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labels.size(0)
            #         correct += (predicted == labels.cuda()).sum().item()
            #
            # print('Accuracy of the network on the 10000 test images: %d %%' % (
            #     100 * correct / total))
            # print("Elapsed time: " ,time.time() - t)
                


# Visualization
test_visual =False

if test_visual == True:
    model.eval()
    model_nullspace= vgg_two_parts().to(device)
    model_nullspace.load_state_dict(torch.load('./model/frozen_nsft_train_classfier.pth'))

    batch = next(iter(test_loader))
    model_output_nullspace,_ = model_nullspace(batch[0].cuda())

   
    label=batch[1].cpu().detach().numpy() 
    X_nullspace=model_output_nullspace.cpu().detach().numpy()
    # 2D
    plot_step_lda(X_nullspace ,label,'Output of the NullSpaceNet')
    #3D
    d3_visualization(X_nullspace,label)  
    # t-SNE
    tsne (X_nullspace,label=batch[1])

    

