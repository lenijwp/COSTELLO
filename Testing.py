import os
import json
from datasets import load_dataset
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import math
from tqdm import tqdm,trange 
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import dataset
import torch
from torch import nn
from scipy import stats
from numba import jit
from torch.utils.data import Dataset, DataLoader


from sklearn.metrics.pairwise import cosine_similarity, paired_distances

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
        batch_size=4   # Can be adjusted according to the actual gpu resources.
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
    if norm=='l2':
        return np.linalg.norm(diff,ord=2)
    if norm=='linf':
        return np.linalg.norm(diff,ord=np.inf)
    if norm=='cos':
        return 1-cosine(X1,X2)

@jit
def EuclideanDistance(x, y):
    """
    get the Euclidean Distance between to matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x:
    :param y:
    :return:
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis

# @jit
# def ManhattanDistance(x, y):
#     print(x.shape)
#     dis = np.zeros((x.shape[0],y.shape[0]))

#     for i in range(x.shape[0]):
#         for j in range(y.shape[0]):
#             dis[i][j] = np.linalg.norm(x[i]-y[j],ord=1)

#     return dis

# # @jit
# def ManhattanDistance(x, y):
    
#     differences = np.abs(x[:, np.newaxis, :] - x[np.newaxis, :, :])
#     print(differences.shape)
#     distances = differences.sum(axis=-1)
#     return distances

# @jit
def ManhattanDistance(x, y):
    
    dis = np.zeros((x.shape[0],y.shape[0]))

    for i in trange(x.shape[0]):
        dis[i] = np.sum(np.abs(x[i] - y), axis=1)

    return dis

@jit
def CosineDistance(x, y):
    xx = np.sum(x ** 2, axis=1) ** 0.5
    x = x / xx[:, np.newaxis]
    yy = np.sum(y ** 2, axis=1) ** 0.5
    y = y / yy[:, np.newaxis]
    dist = 1 - np.dot(x, y.transpose())  # 1 - 余弦距离
    return dist


import math
def GetThreshold(norm,threstype,savecache=None, token_dict_path=''):
    global tokenizer

    if token_dict_path=='':
        wordset = tokenizer.get_vocab()
    else:
        wordset = LoadJson(token_dict_path)

    wordEmb=[]

    

    wordlist=[]

    for key in wordset:
        wordlist.append(str(key))

    wordbatch=64

    lenS = math.ceil(len(wordlist)/wordbatch)

    for i in trange(lenS):
        beg = i*wordbatch
        end = (i+1)*wordbatch

        if end>len(wordlist):
            end = len(wordlist)

        embs = feature_extraction(wordlist[beg:end])

        for ele in embs:
            wordEmb.append(ele.numpy())

    wordEmb = np.array(wordEmb)


    if norm=='l1':
        worddis = ManhattanDistance(wordEmb,wordEmb)
    elif norm=='l2':
        worddis = EuclideanDistance(wordEmb,wordEmb)
    elif norm=='cos':
        worddis = CosineDistance(wordEmb,wordEmb)
    for i in range(worddis.shape[0]):
        worddis[i][i]=10000000


    closeDis=np.zeros(worddis.shape[0])

    for i in range(worddis.shape[0]):
        closeDis[i] = worddis[i].min()

    if savecache!=None:
        np.save(savecache, closeDis)

    dist = getattr(stats, 'norm')
    parameters = dist.fit(closeDis)

    if threstype=='2sigma':
        th = parameters[0]-2*math.sqrt(parameters[1])
    elif threstype=='1sigma':
        th = parameters[0]-math.sqrt(parameters[1])
    elif threstype=='min':
        th = min(closeDis)


    

    if th<0:
        th=0

    return th


def GetThresholdfromCache(norm,threstype,path):

    closeDis=np.load(path)

    dist = getattr(stats, 'norm')
    parameters = dist.fit(closeDis)
    if threstype=='2sigma':
        th = parameters[0]-2*math.sqrt(parameters[1])
    elif threstype=='1sigma':
        th = parameters[0]-math.sqrt(parameters[1])
    elif threstype=='min':
        th = min(closeDis)

    #th = parameters[0]


    if th<0:
        th=0

    return th
    # return min(closeDis)


parser = argparse.ArgumentParser()

parser.add_argument('--contrastset',type=str,help='path of input files under initial_data')
parser.add_argument('--plm',type=str,help='name of pretrained languague model to test')
parser.add_argument('--cache',type=str,default='/data/jwp/Models/huggingface/',help='dict of huggingface cache')
parser.add_argument('--gpu',type=str,default='',help='gpu id, if value is default then use cpu')
parser.add_argument('--outputdir',type=str,help='path of input files under initial_data')
parser.add_argument('--customodel',type=str,default='None',help='customodel')
parser.add_argument('--customcache',type=str,default='../mutatedPLMs',help='customodel, if here, the plm is replaced..')
parser.add_argument('--norm',type=str,default='l2',choices=['l2','l1','cos'],help='norm of distance')
parser.add_argument('--thres',type=str,default='min',choices=['min','1sigma','2sigma','zero'],help='norm of distance')
parser.add_argument('--tokendict',type=str,default='',help='path of customed token diction')
parser.add_argument('--tokencache',type=str,default='',help='name of new cache')

args = parser.parse_args()

norm = args.norm

beta = 1.0

tokenizer = AutoTokenizer.from_pretrained(args.plm,cache_dir=args.cache,model_max_length=512)

if args.customodel =='None':
    model = AutoModel.from_pretrained(args.plm,cache_dir=args.cache)

else:
    model = AutoModel.from_pretrained(os.path.join(args.customcache,args.customodel))

device=("cuda:"+str(args.gpu)) if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

if args.customodel =='None':
    output_name =  str(args.plm)
else:
    output_name = str(args.customodel)

output_name = output_name.replace('/','-')


TH = 0

if args.thres!='zero':

    if os.path.exists(os.path.join(args.tokencache,output_name+'.npy'))==True:
        print("begin to load threshold")
        TH = GetThresholdfromCache(args.norm,args.thres,os.path.join(args.tokencache,output_name+'.npy'))
    else:
        print("begin to calculate threshold.....")
        TH = GetThreshold(args.norm,args.thres,os.path.join(args.tokencache,output_name+'.npy'),args.tokendict)

print("threshold is {}".format(TH))

ContrastSet = LoadJson(args.contrastset)

# print(ContrastSet.keys())

#begin to test

Results={}





for Mutate_Type in list(ContrastSet.keys()):

    Data = ContrastSet[Mutate_Type]

    Results[Mutate_Type]=[]

    print('Begin to test on {} contrast set'.format(Mutate_Type))

    for i in trange(len(Data)):
        Triple = Data[i][0:3]
        
        Embs = feature_extraction(Triple)

        Emb_seed = Embs[0].numpy()
        Emb_close = Embs[1].numpy()
        Emb_far = Embs[2].numpy()

        dis_sc = Calculate_distance(Emb_seed, Emb_close, norm)
        dis_sf = Calculate_distance(Emb_seed,Emb_far,norm)

        if dis_sc - dis_sf >TH:
            Results[Mutate_Type].append([i,Triple])
        

    print('Find {} / {} bad triples, rate is {}...'.format(len(Results[Mutate_Type]),len(Data),
        len(Results[Mutate_Type])/len(Data)))




WriteJson(Results,os.path.join(args.outputdir,output_name))