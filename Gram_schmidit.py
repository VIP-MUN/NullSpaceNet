#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 16:24:25 2020

@author: 
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
from adamod import AdaMod
import pdb
import scipy.linalg as la
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from helpers import *
from sklearn.manifold import TSNE
import time
import seaborn as sns
 
import scipy as sp
from rank_nullspace import nullspace
from sklearn.preprocessing import KernelCenterer

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

