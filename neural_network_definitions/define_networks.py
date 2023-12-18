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

