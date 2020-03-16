#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov. 13 23:51:04 2020

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.linalg.decomp import eigh
import scipy
from scipy import linalg, matrix
# from gesvd import GESVD
# svd = GESVD()



def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


def tsne (input,label):    
    input = input
    # time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(input)    
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_subset={}
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x="tsne-2d-one",
    y="tsne-2d-two",
    hue=label,
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.99
    
    )

def orthonormalize(vectors):
    """    
        Orthonormalizes the vectors using gram schmidt procedure.    
    
        Parameters:    
            vectors: torch tensor, size (dimension, n_vectors)    
                    they must be linearly independant    
        Returns:    
            orthonormalized_vectors: torch tensor, size (dimension, n_vectors)    
    """    
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'    
    orthonormalized_vectors = torch.zeros_like(vectors)    
    orthonormalized_vectors[:, 0] = vectors[:, 0].clone() / torch.norm(vectors[:, 0], p=2)    
    
    for i in range(1, orthonormalized_vectors.size(1)):    
        vector = vectors[:, i].clone()   
        V = orthonormalized_vectors[:, :i] .clone()   
        PV_vector= torch.mv(V, torch.mv(V.t(), vector))    
        orthonormalized_vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)    
    
    return orthonormalized_vectors

def my_nullspace(At, rcond=None):    
    ut, st, vht = torch.Tensor.svd(At, some=False,compute_uv=True)
    vht=vht.t()        
    Mt, Nt = ut.shape[0], vht.shape[1] 
    if rcond is None:
        rcond = torch.finfo(st.dtype).eps * max(Mt, Nt)
    tolt = torch.max(st) * rcond
    numt= torch.sum(st > tolt, dtype=int)
    nullspace = vht[numt:,:].T.cpu().conj()
    return nullspace
def loss_nullspacenet (H,y):

    H=H.type(torch.float32).cuda()
    y=y.cuda()
    C=10
    d= H.shape[1]
    lamb=0.001 # 0.00001
    mean_vectors_list = []
    mean_vectors_list.append([torch.mean(H[y==cl], dim=0,dtype=torch.float32) for cl in range(C) ])
    
    mean_vectors_tensor=torch.stack(mean_vectors_list[0])

    normalized_feat_for_each_class=[(H [y==cnt]  - mean_vectors_tensor[cnt] ) for cnt in range(C) ]
    normalized_feat_for_each_class_tensor=torch.cat(normalized_feat_for_each_class).squeeze(1).squeeze(2).squeeze(2).cuda()

    S_w= torch.zeros((d,d),requires_grad=True,dtype=torch.float32).cuda()
    S_w +=lamb* torch.eye(*S_w.size(),out=torch.empty_like(S_w)) #was right after next loop
    for cl in range (C):
        Sc= torch.matmul(normalized_feat_for_each_class_tensor[y==cl].T, 
                          normalized_feat_for_each_class_tensor[y==cl])
    
        Nc=torch.tensor(normalized_feat_for_each_class_tensor[y==cl].shape[0],dtype=torch.float32)
        S_w +=Sc/(Nc-1)/C#(Nc-1))#/C#S_w +=Sc/(Nc-1)/C

    total_pop_mean = torch.mean(H,dim=0).unsqueeze(0) 
    
    X_bar= (H - total_pop_mean).squeeze(2).squeeze(2)  #X_bar=torch.Size([2500, 10])

    St = torch.matmul(X_bar.T,X_bar)/(H.shape[0]-1)
    Sb= St-S_w
    u,sig,_ =  torch.svd(X_bar.T, some=True,compute_uv=True)
    # u, sig,_=svd(X_bar.T.cpu())
    # t= torch.matrix_rank(St)
    # u1=U[:,:t]
    u1=u#[:,: t]
    # v1=v[:,:t]
    # sigt = torch.diag(sig)
    sb_bar= torch.mm(torch.mm(u1.T,Sb),u1 )
    sw_bar = torch.mm(torch.mm(u1.T,S_w),u1 )
    st_bar = torch.mm(torch.mm(u1.T,St),u1 )
    W=my_nullspace(sw_bar).cuda()  #,rcond=1.
    e_vals, M  = torch.symeig((W.T.mm(sb_bar)).mm(W), eigenvectors = True)
    # _,e_vals, M =torch.svd((W.T.mm(sb_bar)).mm(W), compute_uv=True)
    # e_vals, _ = torch.sort(e_vals, descending=False)
    top_k_evals = e_vals[-9:]
    thresh = torch.min(top_k_evals) + 1.
    top_k_evals = top_k_evals[(top_k_evals <= thresh).nonzero()]
    loss =  -torch.mean(top_k_evals)
    return loss

def plot_step_lda(inpu,y,text):
    label_dict = {0: '0', 1: '1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'}
    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(0,10),('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h'),('blue', 'red', 'green','cyan', 'magenta', 'yellow','dimgray', 'darkred', 'beige','olive')):

        plt.scatter(x=inpu[:,0].real[y == label],
                y=inpu[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.7,
                label=label_dict[label]
                )

    plt.xlabel('Dimensin-1')
    plt.ylabel('Dimensin-2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(text)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()
    
    
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image



def d3_visualization(tensor,label):
    
   fig = plt.figure()
   
   ax = Axes3D(fig) #<-- Note the difference from your original code...

   ax.scatter(
   xs=tensor[:,0], 
   ys=tensor[:,1],
   zs=tensor[:,2],
   c=label, 
   cmap='tab10'
   )
   ax.set_xlabel('First-dim')
   ax.set_ylabel('Second-dim')
   ax.set_zlabel('Third-dim')
   plt.show()




class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
class vgg_w_classifier (nn.Module):
    
    def __init__(self):
            super(vgg_w_classifier,self).__init__()
            self.net= models.vgg16_bn(pretrained= True)
            # net= models.alexnet(pretrained= True)
            
            for param in self.net.parameters():
                param.requires_grad=True   #False 

            self.net.classifier=  nn.Sequential(
                 nn.Linear(in_features=25088, out_features=4096, bias=True),
                 nn.ReLU(inplace=True),
                 nn.Dropout(p=0.5, inplace=False),
                 nn.Linear(in_features=4096, out_features=4096, bias=True),
                 nn.ReLU(inplace=True),
                 nn.Dropout(p=0.5, inplace=False),
                 nn.Linear(in_features=4096, out_features=10,bias=True)
                )
            

            
#            
    def forward(self,x):

              x= self.net(x)
              # x= self.seq(x)
        
              return x
class vgg_no_classifier (nn.Module):
    
    def __init__(self):
            super(vgg_no_classifier,self).__init__()
            net= models.vgg16_bn(pretrained= True)

            
            for param in net.parameters():
                param.requires_grad=True   #False 


            self.feat= net.features
            self.avg= net.avgpool
            self.nscnn=nn.Sequential(
                
                 nn.Conv2d(in_channels=512, 
                                   out_channels=10, 
                                   kernel_size=3, #3 for stl10
                                   padding=0),
                 nn.BatchNorm2d(10, eps=1e-05,
                             momentum=0.1, 
                             affine=True, 
                             track_running_stats=True),
                 
                nn.ReLU()
                
                )

    def forward(self,x):

              x= self.feat(x)

              model= self.nscnn(x)
        
              return model.view(model.size(0), -1)
class vgg_two_parts (nn.Module):
    
    def __init__(self):
            super(vgg_two_parts,self).__init__()
            net= models.vgg16_bn(pretrained= True)
            num_feat= net.classifier[6].in_features
            num_feat2= net.classifier[0].out_features
            net.classifier[0]=  nn.Linear(1000, num_feat2)
            net.classifier[6]=  nn.Linear(num_feat, 10)

            for param in net.parameters():
                param.requires_grad=True   #False
            self.net_fet = net.features
            
            
            self.nscnn=nn.Sequential(
                
                 nn.Conv2d(in_channels=512, 
                                   out_channels=1000, 
                                   kernel_size=3, #3 for stl10
                                   padding=0)
                )
            for param in self.nscnn.parameters():
                param.requires_grad=True   #False

            self.classifier=  nn.Sequential(
                 nn.Linear(in_features=1000, out_features=800, bias=True),
                 nn.ReLU(inplace=False),
 
                 nn.Linear(in_features=800, out_features=500, bias=True),
                 nn.ReLU(inplace=False),

                nn.Linear(in_features=500, out_features=300, bias=True),
                nn.ReLU(inplace=False),
 
                 nn.Linear(in_features=300, out_features=10,bias=True)
                )
             

            
#            
    def forward(self,x):
              x_feat= self.net_fet(x)
              conv= self.nscnn(x_feat)
              flatt= conv.view(conv.size(0), -1)
              c= self.classifier(flatt)
        
              return flatt, c
