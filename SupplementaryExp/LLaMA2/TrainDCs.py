import os
import sys
import json
import argparse
import os.path as osp
import sys
import time
import math
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModel
)
BaseDir=os.path.dirname(__file__)
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import numpy as np

class My_dataset(Dataset):
    def __init__(self,x,y):
        '''
        x: torch.tensors
        y: a list of label
        '''
        self.x=x
        self.y=y

    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return len(self.x)

def WriteJson(data,path):
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=4)

def LoadJson(path):
    res=[]
    with open(path,mode='r',encoding='utf-8') as f:
        dicts = json.load(f)
        res=dicts
    return res




parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='../../data/initial_data/sst_train.json',help='path of input files under initial_data')
parser.add_argument('--output_dir',type=str,default='./DCs',help='the path ot new dictionary')
parser.add_argument('--gpu',type=str,default='0',help='gpu id, if value is default then use cpu')
parser.add_argument('--epoch',type=int,default=150,help='customodel')

args = parser.parse_args()


### load dataset
inputdata=LoadJson(args.dataset)





device=("cuda:"+args.gpu) if torch.cuda.is_available() else "cpu"

### get the embeddings of sentences and prepare training data for Downstream Classifiers
sentences=[]
labels=[]
for idx in inputdata.keys():
    sentences.append(inputdata[idx][0])
    labels.append(inputdata[idx][1])

embeddings = np.load('./sst2-train.npy')
# embeddings = embeddings.reshape((embeddings.shape[0]*embeddings.shape[1],embeddings.shape[2]))
embeddings = torch.from_numpy(embeddings)
embeddings = torch.tensor(embeddings,dtype = torch.float32)

del inputdata




print(embeddings.shape)
traindata=My_dataset(embeddings,labels)
trainLoader=DataLoader(traindata,batch_size=64,shuffle=True)




output_dir=args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

import DClassifier as DCM

DC_record={}
epochs=args.epoch

def evalute(loader):
    Dclf.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = Dclf(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


for model_k in range(14):
    Dclf = DCM.CreateModel(model_k,embeddings.shape[1],len(list(set(labels))))
    Dclf.to(device)

    print('Begin to train downstream classifer {}'.format(model_k))

    DC_record[model_k]={'best_train_acc':0.0}
    optimizer = torch.optim.SGD(Dclf.parameters(), lr=0.002)
    loss_fun = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    savePath= os.path.join(output_dir,str(model_k)+'.pt')

    for epoch in trange(epochs):
        for step, (x, y) in enumerate(trainLoader):
            x, y = x.to(device), y.to(device)
            Dclf.train()
            logits = Dclf(x)
            loss = loss_fun(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        if epoch % 1 == 0:
            val_acc = evalute(trainLoader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(Dclf.state_dict(),savePath)
                DC_record[model_k]['best_train_acc']=val_acc

    WriteJson(DC_record,os.path.join(output_dir,'DC_info.json'))

    print('Downstream classifer {} achieve the best accuracy {}'.format(model_k,best_acc))

