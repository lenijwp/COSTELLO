import os
import sys
import json
import argparse
import os.path as osp
import sys
import time
import numpy as np
import math
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModel
)
BaseDir='/data/jwp/codes/nlptest/ContrasTesting'
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import DClassifier as DCM
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
# import cohens_d as cD
from scipy.stats import ttest_rel,wilcoxon
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score



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

def Calculate_distance(X1,X2,norm):
    diff=X1-X2
    if norm=='l1':
        return np.linalg.norm(diff,ord=1)
    if norm=='l1':
        return np.linalg.norm(diff,ord=2)
    if norm=='linf':
        return np.linalg.norm(diff,ord=np.inf)
    if norm=='cos':
        return -1*cosine(X1,X2)

def Sigmoid(x):
    s = 1/(1 + math.exp(-x))
    return s

def CrossEntropyLoss(P,Q):
    delte=1e-7
    P = np.exp(P)/np.sum(np.exp(P))
    Q = np.exp(Q)/np.sum(np.exp(Q))
    return -np.sum(P*np.log(Q+delte))

def Logits_Lp(P,Q):
    return Calculate_distance(P,Q,'l1')

def Predict_Lp(P,Q):
    P = np.exp(P)/np.sum(np.exp(P))
    Q = np.exp(Q)/np.sum(np.exp(Q))
    return Calculate_distance(P,Q,'l1')

softNet=torch.nn.Softmax(dim=1)

def GetSoftmax(x):
    x=[x]
    x = torch.tensor(x)
    y = softNet(x)
    
    y = y.numpy().tolist()
    return y[0]

def getLabel(x):
    return x.index(max(x))

parser = argparse.ArgumentParser()
parser.add_argument('--bugs',type=str,help='path of test results')
parser.add_argument('--contrastset',type=str,help='path of test suite')
parser.add_argument('--plm',type=str,help='name of pretrained languague model to test')
parser.add_argument('--cache',type=str,default='/data/jwp/Models/huggingface/',help='dict of huggingface cache')
parser.add_argument('--dclf_dir',type=str,help='the dictionary of classifiers')
parser.add_argument('--gpu',type=str,default='',help='gpu id, if value is default then use cpu')
parser.add_argument('--customodel',type=str,default='None',help='name of customodel')
parser.add_argument('--results',type=str,help='output path')
parser.add_argument('--customcache',type=str,default='../mutatedPLMs',help='path of customodel, if here, the plm is replaced..')

# inputparams=[
#     '--bugs','./newdata/results/textattack-bert-base-uncased-SST-2',
#     '--plm','bert-base-uncased',
#     '--dclf_dir','./newdata/DMS',
#     '--gpu','1',
#     '--customodel','None',
#     '--contrastset', './data/contrast_set/imdb_allenai.json',
#     '--results','./newdata/testoutput/eval_dc.json'
# ]
# args = parser.parse_args(args=inputparams)

args = parser.parse_args()

VotingNum= 14

if args.customodel =='None':
    plmname = str(args.plm)
else:
    plmname = str(args.customodel)

dclf_dir=os.path.join(args.dclf_dir,plmname.replace('/','-'))


### load the pretrained language model
tokenizer = AutoTokenizer.from_pretrained(args.plm,cache_dir=args.cache,model_max_length=512)

if args.customodel =='None':
    model = AutoModel.from_pretrained(args.plm,cache_dir=args.cache)

else:
    model = AutoModel.from_pretrained(os.path.join(args.customcache,args.customodel))

device=("cuda:"+args.gpu) if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
### get the embeddings of sentences and prepare training data for Downstream Classifiers

if os.path.exists(args.results)!=True:
    tmp={}
    WriteJson(tmp,args.results)

if os.path.exists(args.bugs)==True:
    ContrastSet = LoadJson(args.contrastset)
    ReportedBugs = LoadJson(args.bugs)


    for Mutate_Type in list(ContrastSet.keys()):

        Data = ContrastSet[Mutate_Type]
        lens = len(Data)

        embeddings=[]
        for unit in Data:
            embeddings.append(feature_extraction(unit))


        Reported = ReportedBugs[Mutate_Type]
        TabelReported = np.zeros(lens)

        for ele in Reported:
            TabelReported[ele[0]]=1

        
        

        print('Begin to test on {} contrast set'.format(Mutate_Type))

        diff_c=np.zeros((len(Data),VotingNum))
        diff_f=np.zeros((len(Data),VotingNum))

        for model_k in trange(VotingNum):
            Dclf = DCM.CreateModel(model_k,embeddings[0].shape[1],2) #TODO make it automatic
            savePath= os.path.join(dclf_dir,str(model_k)+'.pt')
            _state_dict = torch.load(savePath)
            Dclf.load_state_dict(_state_dict)
            Dclf.to(device)
            Dclf.eval()

            for i in range(len(embeddings)):
                unit=embeddings[i]
                x=unit.to(device)
                with torch.no_grad():
                    logits = Dclf(x)
                    logits = logits.cpu().numpy()

                    res_s = GetSoftmax(logits[0])
                    res_sc = GetSoftmax(logits[1])
                    res_sf = GetSoftmax(logits[2])

                    pos = getLabel(res_s)
                    pos_c = getLabel(res_sc)
                    pos_f = getLabel(res_sf)

                    # if pos_c==pos_f:
                    diff_c[i][model_k]=abs(res_sc[pos]-res_s[pos])
                    diff_f[i][model_k]=abs(res_sf[pos]-res_s[pos])
                    
                    # else:
                    #     diff_c[i][model_k]=abs(pos_c-pos)
                    #     diff_f[i][model_k]=abs(pos_f-pos)
                    
        # scaler=preprocessing.MinMaxScaler()
        # all_diff = np.concatenate((diff_c,diff_f),axis=0)
        # scaler.fit(all_diff)
        # normaled_diff_c=scaler.transform(diff_c)
        # normaled_diff_f=scaler.transform(diff_f)

        diff = diff_c-diff_f

        same=0
        alpha=0.05
        testRes=np.zeros(diff_c.shape[0])

        TabelEval = np.zeros(lens)

        for i in range(diff_c.shape[0]):
            # CohensD=cD.cohensd_2paired(diff_c[i],diff_f[i])
            # if CohensD<0.5:
            #     continue
            # Res_Ttest=wilcoxon(diff_c[i],diff_f[i],correction=False,alternative='greater',method='auto')
            Res_Ttest=wilcoxon(diff[i],correction=False,alternative='greater',method='auto')
            if Res_Ttest.pvalue<(alpha):
                
                TabelEval[i]=1
                if TabelReported[i]==1:
                    same+=1
        
        p = precision_score(TabelEval, TabelReported, average='binary')
        r = recall_score(TabelEval, TabelReported, average='binary')
        f1 = f1_score(TabelEval, TabelReported, average='binary')

        tmp={}
        tmp["sizeofSuite"]=int(lens)
        tmp["badEmbs"]=int(np.sum(TabelReported))
        tmp["badEval"]=int(np.sum(TabelEval))
        tmp["overlap"]=same
        tmp["precision"]=p
        tmp["recall"]=r
        tmp["f1"]=f1

        TabelEval = np.zeros(lens)
        same=0

        # print(diff[0:10])

        for i in range(diff_c.shape[0]):
 
            # Res_Ttest=wilcoxon(diff_c[i],diff_f[i],correction=False,alternative='less',method='auto')
            Res_Ttest=wilcoxon(diff[i],correction=False,alternative='less',method='auto')
            if Res_Ttest.pvalue>=(alpha):
                
                TabelEval[i]=1
                if TabelReported[i]==1:
                    same+=1
        
        p = precision_score(TabelEval, TabelReported, average='binary')
        r = recall_score(TabelEval, TabelReported, average='binary')
        f1 = f1_score(TabelEval, TabelReported, average='binary')

        tmp["new_badEval"]=int(np.sum(TabelEval))
        tmp["new_overlap"]=same
        tmp["new_precision"]=p
        tmp["new_recall"]=r
        tmp["new_f1"]=f1

        EvalResults= LoadJson(args.results)
        if plmname not in EvalResults:
            EvalResults[plmname]={}
        
        EvalResults[plmname][Mutate_Type]=tmp
        WriteJson(EvalResults,args.results)

else:
    print("Bug Files isn's existing!!!")



