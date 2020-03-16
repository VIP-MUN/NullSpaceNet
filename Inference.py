#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:27:29 2020

@author: abdelpakey
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



device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

transform = transforms.Compose(
    [#transforms.Resize(64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(),
      transforms.ToTensor(),
     
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       # transforms.Normalize([0.485, 0.456, 0.406],
       #                          [0.229, 0.224, 0.225])  # Imagenet standards
      
      ])
 

dataset= "stl10"

if dataset == "stl10":

    train_dataset=datasets.STL10(root='./dataset', split='train', folds=None, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=380, num_workers=18)  #2500

    test_dataset = datasets.STL10(root='./dataset', split='test', download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=300, num_workers=18)

if dataset == "cifar10":        
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2800, num_workers=8)

    test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100, num_workers=8)


dataiter = iter(train_loader)
imgs, labels = dataiter.next()
imgs=imgs.to(device)
labels= labels.to(device)
# model= Deep_w_classifier().to(device)
model = vgg_two_parts().to(device) 
model.load_state_dict(torch.load('./model/NullSpaceNet_only.pth'))

# model_alex.load_state_dict(torch.load('./model/cnn_alex_crossentropy.pth'))


out_simple,_  = model(imgs) #inference time 0.06425905227661133

sklearn_lda = LDA(n_components=None)
X_nullspace_numpy=(out_simple).cpu().detach().numpy()
labels=labels.cpu().detach().numpy()
X_lda = sklearn_lda.fit(X_nullspace_numpy, labels)  


model.eval()

with torch.no_grad():
        #Accuracy

        b=next(iter(test_loader))
        # t=time.time()
        out,_  = model(b[0].cuda())
        b_= out.cpu().detach().numpy()
        # X_lda.predict_proba(b_)
        t=time.time()
        print("Accuracy: " , X_lda.score(b_, b[1])*100)
        print("Elapsed time: " ,time.time() - t)
