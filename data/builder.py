import os
Basepath=os.path.dirname(__file__)
import json
from datasets import load_dataset
import numpy

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



sst = load_dataset("sst",cache_dir='/data/jwp/Datasets/huggingface')
Results={}

print(len(sst['validation']))
print(len(sst['train']))
print(sst.keys())

for part in ['test']:
    jishu = 0
    for itm in range(len(sst[part])):
        # if sst[part][itm]['label']>=0.3 and sst[part][itm]['label']<=0.7:
        #     continue
        label=0
        if sst[part][itm]['label']>0.5:
            label=1
        jishu+=1
        Results[itm]=(sst[part][itm]['sentence'],label)

# WriteJson(Results,Basepath+'/sst_train.json')

print(jishu)


# imdb = load_dataset("imdb",cache_dir='/data/jwp/Datasets/huggingface/')
# Results={}
# for part in ['train']:
#     for itm in range(len(imdb[part])):
#         label=int(imdb[part][itm]['label'])
#         #label = 'Negative' if label<1 else 'Positive'
#         x=imdb[part][itm]['text']
#         Results[itm]=(x,label)
# WriteJson(Results,Basepath+'/imdb_train.json')