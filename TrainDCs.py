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
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

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


def feature_extraction(inputs):
    '''
    inputs: a str or a list or string
    outputs: embeddings
    '''
    global tokenizer
    global model
    global device
    
    if isinstance(inputs,str):  # for a single sentence
        inputs_1= tokenizer(inputs, return_tensors="pt",padding=True,truncation=True).to(device)
        outputs_1 = model(**inputs_1)
        

        flag = hasattr(outputs_1,'pooler_output')
        if flag==True:
            pooler_output_1 = outputs_1.pooler_output
            if device =='cpu':
                return pooler_output_1[0].detach()
            else:
                return pooler_output_1[0].detach().cpu()
        else:
            last_hidden_states_1 = outputs_1.last_hidden_state
            if device =='cpu':
                return last_hidden_states_1[0][0].detach()
            else:
                return last_hidden_states_1[0][0].detach().cpu()

    else:   # for a list of sentence
        batch_size=8   # Can be adjusted according to the actual gpu resources.
        epoch=math.ceil(float(len(inputs))/batch_size)
        first=0
        res=0
        for i in range(epoch):
            begin_ix=i*batch_size
            end_ix=(i+1)*batch_size

            if end_ix>len(inputs):
                end_ix = len(inputs)
            tmps=inputs[begin_ix:end_ix]
            
            inputs_1= tokenizer(tmps, return_tensors="pt",max_length=512,pad_to_max_length = True,truncation=True).to(device)
            # inputs_1= tokenizer(tmps, return_tensors="pt",padding=True,truncation=True).to(device)
            outputs_1 = model(**inputs_1)
 
            flag = hasattr(outputs_1,'pooler_output')

            if flag==True:

            #pooler_output_1 = outputs_1.last_hidden_state

                pooler_output_1 = outputs_1.pooler_output

                if device =='cpu':
                    pooler_output_1=pooler_output_1.detach()
                else:
                    pooler_output_1=pooler_output_1.detach().cpu()

                #print(pooler_output_1.shape)

                if first==0:
                    res=pooler_output_1[:,:]
                    first=1
                else:
                    res=torch.cat((res,pooler_output_1[:,:]))
            
            else:
                last_hidden_states_1 = outputs_1.last_hidden_state
                if device =='cpu':
                    last_hidden_states_1=last_hidden_states_1.detach()
                else:
                    last_hidden_states_1=last_hidden_states_1.detach().cpu()
                if first==0:
                    res=last_hidden_states_1[:,0,:]
                    first=1
                else:
                    res=torch.cat((res,last_hidden_states_1[:,0,:]))

            
        return res


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,help='training data')
parser.add_argument('--plm',type=str,help='name of pretrained languague model to test')
parser.add_argument('--cache',type=str,default='/data/jwp/Models/huggingface/',help='dict of huggingface cache')
parser.add_argument('--output_dir',type=str,help='the dictionary of classifiers')
parser.add_argument('--gpu',type=str,default='',help='gpu id, if value is default then use cpu')
parser.add_argument('--customodel',type=str,default='None',help='name of customodel')
parser.add_argument('--epoch',type=int,default=100,help='epoch')
parser.add_argument('--customcache',type=str,default='../mutatedPLMs',help='path of customodel, if here, the plm is replaced..')

args = parser.parse_args()


### load dataset
inputdata=LoadJson(args.dataset)

### load the pretrained language model
# if args.cache=='':
#     tokenizer = AutoTokenizer.from_pretrained(args.plm,model_max_length=512)
#     plm = AutoModel.from_pretrained(args.plm)
# else:
#     tokenizer = AutoTokenizer.from_pretrained(args.plm,cache_dir=args.cache,model_max_length=512)
#     plm = AutoModel.from_pretrained(args.plm,cache_dir=args.cache)

tokenizer = AutoTokenizer.from_pretrained(args.plm,cache_dir=args.cache,model_max_length=512)

if args.customodel =='None':
    model = AutoModel.from_pretrained(args.plm,cache_dir=args.cache)

else:
    model = AutoModel.from_pretrained(os.path.join(args.customcache,args.customodel))

device=("cuda:"+args.gpu) if torch.cuda.is_available() else "cpu"
model.to(device)
### get the embeddings of sentences and prepare training data for Downstream Classifiers
sentences=[]
labels=[]
for idx in inputdata.keys():
    sentences.append(inputdata[idx][0])
    labels.append(inputdata[idx][1])

embeddings=feature_extraction(sentences)

del model
del tokenizer
del inputdata




print(embeddings.shape)
traindata=My_dataset(embeddings,labels)
trainLoader=DataLoader(traindata,batch_size=64,shuffle=True)


# begin to create and train Downstream Classifier
# TODO : delete the arg.plm

if args.customodel =='None':
    output_name = str(args.plm)
else:
    output_name = str(args.customodel)

output_dir=os.path.join(args.output_dir,output_name.replace('/','-'))

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
    optimizer = torch.optim.Adam(Dclf.parameters(), lr=0.001)
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

