import os
import json
from random import random
from datasets import load_dataset
import torch
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
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import random
from itertools import cycle

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

class ContrastSet(dataset.Dataset):
    '''
    TODO: To build the DataLoader for PyTorch Training .
    '''

    def __init__(self, DATA):

        self.nums = len(DATA)
        self.data = DATA



    def __getitem__(self, index):
        
        return self.data[index]['input_ids'], self.data[index]['token_type_ids'], self.data[index]['attention_mask']
        #return self.data[index]['input_ids'], self.data[index]['attention_mask']

    def __len__(self):
        return self.nums

class RefSet(dataset.Dataset):
    '''
    TODO: To build the DataLoader for PyTorch Training .
    '''

    def __init__(self, Input,Output):

        self.nums = len(Input)
        self.data = Input
        self.output = Output



    def __getitem__(self, index):
        
        return self.data[index]['input_ids'], self.data[index]['token_type_ids'], self.data[index]['attention_mask'], self.output[index]
        #return self.data[index]['input_ids'], self.data[index]['attention_mask']

    def __len__(self):
        return self.nums

class L2Loss(nn.Module):
    
    def __init__(self,weight=None,size_average=True):
        super(L2Loss, self).__init__()
    
    def forward(self,emb0,emb1,emb2):
        dis_c = torch.norm(emb0 - emb1, p=2, dim = 1)
        dis_f = torch.norm(emb0 - emb2, p=2, dim = 1)

        return torch.sum(dis_c-dis_f)
    

from scipy import stats
import math
from numba import jit
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

@jit
def ManhattanDistance(x, y):
    dis = np.zeros((x.shape[0],y.shape[0]))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            dis[i][j] = np.linalg.norm(x[i]-y[j],ord=1)

    return dis

@jit
def CosineDistance(x, y):
    xx = np.sum(x ** 2, axis=1) ** 0.5
    x = x / xx[:, np.newaxis]
    yy = np.sum(y ** 2, axis=1) ** 0.5
    y = y / yy[:, np.newaxis]
    dist = 1 - np.dot(x, y.transpose())  # 1 - 余弦距离
    return dist


def GetThreshold(thres):

    if thres=='zero':
        return 0

    global tokenizer

    wordset = tokenizer.get_vocab()

    

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

    worddis = EuclideanDistance(wordEmb,wordEmb)
    for i in range(worddis.shape[0]):
        worddis[i][i]=10000000


    closeDis=np.zeros(worddis.shape[0])

    for i in range(worddis.shape[0]):
        closeDis[i] = worddis[i].min()

    # if savecache!=None:
    #     np.save(savecache, closeDis)

    dist = getattr(stats, 'norm')
    parameters = dist.fit(closeDis)

    if thres=='m1s':
        th = parameters[0]-math.sqrt(parameters[1])
    if thres=='m2s':
        th = parameters[0]-2*math.sqrt(parameters[1])
    if thres=='min':
        th = min(closeDis)

    if th<0:
        th=0

    return th


# TODO: extracte the feature of the "teacher model"

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

# TODO: implement the contrastive fixing function

def fixTraining(net, data_loader, ref_Loader,train_optimizer, device,TH):
    net.train()
    #net.eval()
    global batch_size
    # cost=torch.nn.MSELoss()
    
    # cost = L2Loss()
    cost = nn.TripletMarginLoss(margin=TH, p=2)
    cost_ref = nn.MSELoss()

    # total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    total_loss, total_num = 0.0, 0

    jishu=0

    tmp=[]

    alpha = 0.4

    accumulation_steps= 8

    if len(data_loader)>=len(ref_Loader):
        ENU = enumerate(zip(tqdm(data_loader), cycle(ref_Loader)))
    else:
        ENU = enumerate(zip(cycle(data_loader), tqdm(ref_Loader)))

    for i,((input_ids, token_type_ids,attention_mask), (input_ids2, token_type_ids2,attention_mask2, groundemb) ) in ENU:

        # print(input_ids.shape)
        # print(token_type_ids.shape)
        # print(attention_mask.shape)

        # print(input_ids2.shape)
        # print(token_type_ids2.shape)
        # print(attention_mask2.shape)
        # print(groundemb.shape)
        # print(groundemb.shape[0])

        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        s = {'input_ids':input_ids[:,0,:], 'token_type_ids':token_type_ids[:,0,:], 'attention_mask':attention_mask[:,0,:]}
        s_pos = {'input_ids':input_ids[:,1,:], 'token_type_ids':token_type_ids[:,1,:], 'attention_mask':attention_mask[:,1,:]}
        s_neg = {'input_ids':input_ids[:,2,:], 'token_type_ids':token_type_ids[:,2,:], 'attention_mask':attention_mask[:,2,:]}

        #s= s.to(device)
        outputs_0 = net(**s)
        #s_pos= s_pos.to(device)
        outputs_1 = net(**s_pos)
        #s_neg= s_neg.to(device)
        outputs_2 = net(**s_neg)

        pooler_output_0 = outputs_0.pooler_output
        pooler_output_1 = outputs_1.pooler_output
        pooler_output_2 = outputs_2.pooler_output

        # print(pooler_output_0.shape)
        # print(pooler_output_1.shape)
        # print(pooler_output_2.shape)

        loss1 = cost(pooler_output_0,pooler_output_1,pooler_output_2)/accumulation_steps

        input_ids2 = input_ids2.to(device)
        token_type_ids2 = token_type_ids2.to(device)
        attention_mask2 = attention_mask2.to(device)
        groundemb = groundemb.to(device)

        # sref = {'input_ids':input_ids2.reshape, 'token_type_ids':token_type_ids2,  'attention_mask':attention_mask2}

        sref = {'input_ids':torch.reshape(input_ids2,(-1,input_ids2.shape[2]) ), 'token_type_ids':torch.reshape(token_type_ids2,(-1,token_type_ids2.shape[2])),  'attention_mask':torch.reshape(attention_mask2,(-1,attention_mask2.shape[2]))}

        #s= s.to(device)
        outputs_ref = net(**sref)
        

        pooler_output_ref = outputs_ref.pooler_output

        groundemb = torch.reshape(groundemb,(-1,groundemb.shape[2]))


        loss2 = cost_ref(pooler_output_ref,groundemb)/accumulation_steps

        loss = alpha*loss1 + (1-alpha)*loss2

        # print("loss1: {}".format(loss1))
        # print("loss2: {}".format(loss2))
        # print("loss: {}".format(loss))
        # print(loss)

        # print(torch.norm(pooler_output_0-pooler_output_1, p=2, dim=1))
        # print(torch.norm(pooler_output_0-pooler_output_2, p=2, dim=1))
        
        # if float(loss.detach().cpu())>0:
        #     jishu+=1
        # print(jishu,float(loss.detach().cpu()))
        loss.backward()
        total_num += len(input_ids)
        total_loss += loss.item()

        if (i + 1) % accumulation_steps == 0:
            train_optimizer.step()
            train_optimizer.zero_grad()



    return total_loss

def test(net, device, Th):
    #net.train()
    net.eval()
    # cost=torch.nn.MSELoss()
    
    # cost = L2Loss()

    cost = nn.TripletMarginLoss(margin=TH, p=2)

    data_loader=DataLoader(traindata,batch_size=1,shuffle=False)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    jishu=0
    for input_ids, token_type_ids,attention_mask in train_bar:
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        s = {'input_ids':input_ids[:,0,:], 'token_type_ids':token_type_ids[:,0,:], 'attention_mask':attention_mask[:,0,:]}
        s_pos = {'input_ids':input_ids[:,1,:], 'token_type_ids':token_type_ids[:,1,:], 'attention_mask':attention_mask[:,1,:]}
        s_neg = {'input_ids':input_ids[:,2,:], 'token_type_ids':token_type_ids[:,2,:], 'attention_mask':attention_mask[:,2,:]}

        #s= s.to(device)
        outputs_0 = net(**s)
        #s_pos= s_pos.to(device)
        outputs_1 = net(**s_pos)
        #s_neg= s_neg.to(device)
        outputs_2 = net(**s_neg)

        try:

            pooler_output_0 = outputs_0.pooler_output
            pooler_output_1 = outputs_1.pooler_output
            pooler_output_2 = outputs_2.pooler_output
        
        except:

            pooler_output_0 = outputs_0.last_hidden_state
            pooler_output_1 = outputs_1.last_hidden_state
            pooler_output_2 = outputs_2.last_hidden_state

            pooler_output_0 = pooler_output_0[:,0,:]
            pooler_output_1 = pooler_output_1[:,0,:]
            pooler_output_2 = pooler_output_2[:,0,:]
        # print(pooler_output_0.shape)
        # print(pooler_output_1.shape)
        # print(pooler_output_2.shape)

        

        #TODO fix the defination of LOSS
        loss = cost(pooler_output_0,pooler_output_1,pooler_output_2)

        #print(loss)

        if float(loss.detach().cpu())>0:
            jishu+=1
        tmp.append(float(loss.detach().cpu()))
        # print(jishu,float(loss.detach().cpu()))
        
#         train_optimizer.zero_grad()
#         loss.backward()
        
#         train_optimizer.step()

#         total_num += batch_size
#         total_loss += loss.item() * batch_size
        # break
    print(jishu)
    return jishu


parser = argparse.ArgumentParser()
parser.add_argument('--contrastset',type=str,help='path of input files under initial_data')
parser.add_argument('--bugs',type=str,help='path of input files under initial_data')
parser.add_argument('--plm',type=str,help='name of pretrained languague model to test')
parser.add_argument('--cache',type=str,default='/data/jwp/Models/huggingface/',help='dict of huggingface cache')
parser.add_argument('--gpu',type=str,default='',help='gpu id, if value is default then use cpu')
parser.add_argument('--customodel',type=str,default='None',help='name of customodel')
parser.add_argument('--customcache',type=str,default='../mutatedPLMs',help='path of customodel, if here, the plm is replaced..')
parser.add_argument('--output',type=str,help='path of repaired models')
parser.add_argument('--thres',type=str,default='m2s',choices=['zero','min','m1s','m2s'],help='threshold, m1s: mean-standard, m2s:mean-2standard')

args = parser.parse_args()


if args.customodel =='None':
    plmname = str(args.plm)
else:
    plmname = str(args.customodel)

plmname= plmname.replace('/','-')



# dclf_dir=os.path.join(args.dclf_dir,plmname.replace('/','-'))


device_id = args.gpu
device=("cuda:"+str(device_id)) if torch.cuda.is_available() else "cpu"

bug_path = args.bugs

fixed_model_path = args.output



# TODO: load buggy contrast triples and deal with them into Dataloader

BuggySet = LoadJson(bug_path)
AllData = LoadJson(args.contrastset)

Data = []

tokenizer = AutoTokenizer.from_pretrained(args.plm,cache_dir=args.cache,model_max_length=512)

if args.customodel =='None':
    model = AutoModel.from_pretrained(args.plm,cache_dir=args.cache)

else:
    # model = AutoModel.from_pretrained(os.path.join('../mutatedPLMs',args.customodel))
    model = AutoModel.from_pretrained(os.path.join(args.customcache, args.customodel))


model.to(device)
model.eval()

refInput = []
refOutput = []


TH = GetThreshold(args.thres)

for Mutate_Type in list(BuggySet.keys()):

    tmp = BuggySet[Mutate_Type]

    iflag = np.zeros(len(AllData[Mutate_Type]))

    for ele in tmp:
        Data.append(tokenizer(ele[1], return_tensors="pt",truncation=True,max_length=512,pad_to_max_length = True))
        iflag[ele[0]]=1
        #Data.append(tokenizer(ele[1], return_tensors="pt",truncation=True,max_length=512,padding=True))
    
    choice = []
    for i in range(len(iflag)):
        if iflag[i]==0:
            choice.append(i)
        
    #choice = random.sample(choice, min(len(choice),len(tmp)))

    # choice = random.sample(choice,10)

    for i in choice:
        refInput.append(tokenizer(AllData[Mutate_Type][i], return_tensors="pt",truncation=True,max_length=512,pad_to_max_length = True))
        refOutput.append(feature_extraction(AllData[Mutate_Type][i]))


del(model)

print(len(Data))

traindata = ContrastSet(Data)
refdata = RefSet(refInput,refOutput)

batch_size = 4

trainLoader = DataLoader(traindata,batch_size=batch_size,shuffle=True)
refLoader = DataLoader(refdata,batch_size=int(batch_size),shuffle=True)

minLoss=10000000000
# a,b,c = traindata.__getitem__(0)

# print(a.shape,b.shape,c.shape)

# a,b,c = traindata.__getitem__(1)

# print(a.shape,b.shape,c.shape)




for i in range(1):

    if args.customodel =='None':
        model = AutoModel.from_pretrained(args.plm,cache_dir=args.cache)

    else:
        # model = AutoModel.from_pretrained(os.path.join('../mutatedPLMs',args.customodel))
        model = AutoModel.from_pretrained(os.path.join(args.customcache, args.customodel))


    
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-6)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)

    for epoch in range(8):
        loss= fixTraining(model, trainLoader, refLoader ,optimizer, device, TH)
        exp_lr_scheduler.step()
        num = test(model, device, 0)
        
        if num<minLoss:
            model.save_pretrained(fixed_model_path)
            minLoss = num
        
        print("Epoch {} : loss for this epoch is {},  the best loss until now is {}.".format(
            epoch, loss, minLoss
        ))