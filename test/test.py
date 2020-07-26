#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:36:40 2019

@author: gongchaoyun
"""

import pandas as pd
from PIL import Image



class aaa:
    def a(self,b,c,out):
        out.append(b)
        out.append(c)

    def b(self,b,c):
        out = []
        self.a(b,c,out)
        print(out)
        return out
    

a =aaa()
b = 1
c =3
mm = a.b(b,c)

x = [1,2,3,4]
xx = x[0:3]

import torch
 
import torchvision.models as models

from torchvision import datasets, transforms
 
from torchsummary import summary
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets.MNIST('./data', train=True, download=True)
model = models.resnet18().to(device)
summary(model, (3,256,256))

