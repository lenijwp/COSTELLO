import os
from os import path as osp
import json
from datasets import load_dataset
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from scipy.spatial.distance import cosine
import math
from tqdm import tqdm,trange 
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import dataset
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

def WriteJson(data,path):
    '''
    '''
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=4)

def LoadJson(path):
    '''
    '''
    res=[]
    with open(path,mode='r',encoding='utf-8') as f:
        dicts = json.load(f)
        res=dicts
    return res

def readfile(filename):
    # f = open("mapfile","r")
    f = open(filename,"r") 

    lines = f.readlines()
    res = []
    for line in lines:
        res.append(line.strip('\n'))

    
    f.close()
    return res


def GaussianFuzzing(model,mu,sigma):
    
    dict = model.state_dict()
    changed=False

    for key in dict.keys():
        if ('query.weight' in key) or ('key.weight' in key) or ('value.weight' in key):
            if len(dict[key].shape) !=2:
                continue

            tmp = dict[key]
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    tmp[i][j]+=random.gauss(mu,sigma)
            
            dict[key]=tmp
            changed = True
    
    return changed, dict

def WeightShuffling(model,p):

    dict = model.state_dict()
    changed=False

    for key in dict.keys():
        if ('query.weight' in key) or ('key.weight' in key) or ('value.weight' in key):
            if len(dict[key].shape) !=2:
                continue
        
            masks = np.random.binomial(1,p,size = dict[key].shape[0])

            tmp = dict[key].numpy()

            for i in range(tmp.shape[0]):
                if masks[i]==1:
                    tmp[i] = np.random.permutation(tmp[i])
            
            tmp = torch.from_numpy(tmp)
            dict[key]=tmp

            # for i in range(tmp.shape[0]):
            #     for j in range(tmp.shape[1]):
            #         tmp[i][j]+=random.gauss(mu,sigma)
            
            # dict[key]=tmp

            changed = True
    
    return changed, dict

def NeuronEffectBlocking(model,p):
    dict = model.state_dict()
    changed=False

    for key in dict.keys():
        if ('query.weight' in key) or ('key.weight' in key) or ('value.weight' in key):
            if len(dict[key].shape) !=2:
                continue
        
            masks = np.random.binomial(1,p,size = dict[key].shape[1])

            tmp = dict[key].numpy()

            tmp = tmp.T

            for i in range(tmp.shape[0]):
                if masks[i]==1:
                    tmp[i] = np.zeros(tmp.shape[1])
            
            tmp = torch.from_numpy(tmp.T)
            dict[key]=tmp

            # for i in range(tmp.shape[0]):
            #     for j in range(tmp.shape[1]):
            #         tmp[i][j]+=random.gauss(mu,sigma)
            
            # dict[key]=tmp

            changed = True
    
    return changed, dict



file = 'plms.txt'
plms = readfile(file)

baseDir = '../mutatedPLMs'
cache_dir = '/data/jwp/Models/huggingface/'

pos=0
for plm in plms:
    print('begin to mutate No.{} plm:............'.format(pos))
    pos+=1
    try:
        #tokenizer = AutoTokenizer.from_pretrained(plm, cache_dir=cache_dir,model_max_length=512)
        model = AutoModelForSequenceClassification.from_pretrained(plm, cache_dir=cache_dir)
    except:
        print("failed loading {}....".format(plm))
        continue

    plmname = plm.replace('/','-')

    for k in trange(5):
        
        _, new_dict = GaussianFuzzing(model, 0, 0.01)
        if _ == True:
            old_dict = model.state_dict()
            model.load_state_dict(new_dict)
            model.save_pretrained(osp.join(baseDir,plmname+'-GaussianFuzzing'+str(k)))
            model.load_state_dict(old_dict)
            tip={}
            tip['iniplm']=plm
            WriteJson(tip,osp.join(baseDir,plmname+'-GaussianFuzzing'+str(k),'tip.json'))

        
        _, new_dict = WeightShuffling(model, 0.01)
        if _ == True:
            old_dict = model.state_dict()
            model.load_state_dict(new_dict)
            model.save_pretrained(osp.join(baseDir,plmname+'-WeightShuffling'+str(k)))
            model.load_state_dict(old_dict)
            tip={}
            tip['iniplm']=plm
            WriteJson(tip,osp.join(baseDir,plmname+'-WeightShuffling'+str(k),'tip.json'))
        

        _, new_dict = NeuronEffectBlocking(model, 0.01)
        if _ == True:
            old_dict = model.state_dict()
            model.load_state_dict(new_dict)
            model.save_pretrained(osp.join(baseDir,plmname+'-NeuronEffectBlocking'+str(k)))
            model.load_state_dict(old_dict)
            tip={}
            tip['iniplm']=plm
            WriteJson(tip,osp.join(baseDir,plmname+'-NeuronEffectBlocking'+str(k),'tip.json'))