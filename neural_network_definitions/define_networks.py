import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import torch.nn.functional as F
import torchvision
import torch.optim as optim

import random
import os

import sys
import subprocess
import time
import argparse
import copy

class TwoLayer(nn.Module):
    def __init__(self, D, l1_width,device="cpu",nonlinearity="relu"):
        """
        Parameters:
        -----------

        D : input dimension
        l1_width : number of hidden nodes
        """
        super(TwoLayer, self).__init__()
        self.D = D
        self.l1_width = l1_width
        self.nonlinearity = nonlinearity

        self.fc1 = nn.Linear(D, l1_width)
        self.fc2 = nn.Linear(l1_width, 2)
        self.fc1.weight.to(device)
        self.fc2.weight.to(device)



    def forward(self, x):
        # input to hidden
        #x = x.to(torch.float32) #to(torch.double)
        if self.nonlinearity=="relu" or self.nonlinearity=="":
            x = F.relu(self.fc1(x))
        elif self.nonlinearity=="linear":
            x = (self.fc1(x))
        elif self.nonlinearity=="quadratic":
            x = (self.fc1(x))**2.
        elif self.nonlinearity=="cubic":
            x = (self.fc1(x))**3.
        elif self.nonlinearity=="fourth_power":
            x = (self.fc1(x))**4.

        x = self.fc2(x)
        return x
    
    def reset_all_layers(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class TwoLayerNet_NTK(nn.Module):

    def __init__(self,x_width=784,l1_width=120,num_classes=2, train_layers=[True,True],
                 train_biases=[False,True],device="cpu", alpha=1.):
        super(TwoLayerNet_NTK, self).__init__()
        self.fc1 = nn.Linear(x_width, l1_width,bias=train_biases[0])
        torch.nn.init.normal_(self.fc1.weight, mean=0.0) #, std=std0)
        self.fc1.weight.requires_grad = train_layers[0]
        if train_biases[0]:
            self.fc1.bias.requires_grad = train_layers[0]
        self.fc1.weight.to(device)

        self.init_fc1 = copy.deepcopy(self.fc1)
        self.init_fc1.requires_grad = False
        #if train_biases[0]:
        #    self.init_fc1.bias.requires_grad = False

        self.fc2 = nn.Linear(l1_width, num_classes,bias=train_biases[1])
        torch.nn.init.normal_(self.fc2.weight, mean=0.0) #, std=std0)
        self.fc2.weight.requires_grad = train_layers[1]
        if train_biases[1]:
            self.fc2.bias.requires_grad = train_layers[1]
        self.fc2.weight.to(device)
        self.init_fc2 = copy.deepcopy(self.fc2)
        self.init_fc2.requires_grad = False

        #if train_biases[1]:
        #    self.init_fc2.bias.requires_grad = False


        self.x_width=x_width
        self.l1_width=l1_width
        self.alpha=alpha

    def reset_all_layers(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        tmp_length=len(x)
        x = x.to(torch.float32)
        x = (x.reshape(-1, tmp_length))
        x_normed = (self.x_width**(-0.5))*x

        x = F.relu(self.init_fc1(x_normed))
        x_out0 = self.init_fc2(x)

        x = F.relu(self.fc1(x_normed))
        x_out = self.fc2(x)

        x=(x_out-x_out0)*self.alpha
        x = 1./self.l1_width*x
        x = torch.transpose(x,0,1)
        x = x.reshape(max(x.size()))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


