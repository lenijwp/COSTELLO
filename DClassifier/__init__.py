import os
import sys
BaseDir=os.path.dirname(__file__)

import torch
from torch import nn

class simpleNet_1(nn.Module):

    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(simpleNet_1, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, out_dim)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class simpleNet_2(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet_2, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
 
class Activation_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Activation_Net_tanh(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net_tanh, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def CreateModel(id,input_dim,out_dim):
    # from ModelDefine import simpleNet_1,Activation_Net,simpleNet_2
    if id==0:
        return simpleNet_1(input_dim,256,out_dim)
    if id==1:
        return simpleNet_1(input_dim,512,out_dim)
    if id==2:
        return simpleNet_2(input_dim,256,64,out_dim)
    if id==3:
        return simpleNet_2(input_dim,512,64,out_dim)
    if id==4:
        return Activation_Net(input_dim,256,16,out_dim)
    if id==5:
        return Activation_Net(input_dim,512,64,out_dim)
    if id==6:
        return Activation_Net(input_dim,256,64,out_dim)
    if id==7:
        return simpleNet_1(input_dim,1024,out_dim)
    if id==8:
        return simpleNet_2(input_dim,256,32,out_dim)
    if id==9:
        return simpleNet_2(input_dim,512,32,out_dim)
    if id==10:
        return simpleNet_2(input_dim,512,256,out_dim)
    if id==11:
        return Activation_Net_tanh(input_dim,256,16,out_dim)
    if id==12:
        return Activation_Net_tanh(input_dim,512,64,out_dim)
    if id==13:
        return Activation_Net_tanh(input_dim,256,64,out_dim)

if __name__=='__main__':
    for i in range(14):
        model=CreateModel(i,768,2)
        print(model)