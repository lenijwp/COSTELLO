import os
import sys
import json
import argparse
import os.path as osp
import sys
import time
import random
BaseDir=os.path.dirname(__file__)
from Generator.SentenceTrans import SentenceGenerator
# from Generator.SentencePairTrans import SentencePairGenerator

os.path.append('/data/jwp/codes/Tools/NL-Augmenter')

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

parser = argparse.ArgumentParser()
parser.add_argument('--trans',type=str, default='SA.json',help='config of selected transformations')
parser.add_argument('--input',type=str, default='sst_train.json',help='path of input files under initial_data')
parser.add_argument('--output',type=str,default='trysee.json',help='path of output contrastsets under initial_data')

args = parser.parse_args()
config=LoadJson(osp.join(BaseDir,'data/inputconfig',args.trans))
data=LoadJson(osp.join(BaseDir,'data/initial_data',args.input))

results={}

if config['generate']['type']=='SentenceTrans':
    Generator=SentenceGenerator()
# else:
#     Generator=SentencePairGenerator()

if len(config['generate']['singleProperty'])>0:
    for trans in config['generate']['singleProperty']:
        BeginTime=time.time()
        transName=trans
        results[transName]=[]
        for idx in data.keys():
            
            s=data[idx][0]
            try:
                tmp=eval('Generator.'+str(trans)+'(s)')
            except:
                continue
            if len(tmp)!=0:
                # print(len(tmp))
                for ele in tmp:
                    results[transName].append(ele)
            
        
        print("Finish {} : output {} unit triplets during {} seconds.".format(transName,len(results[transName]),time.time()-BeginTime))

    
if len(config['generate']['crossProperty'])>0:
    for trans in config['generate']['crossProperty']:
        BeginTime=time.time()
        transName=trans[0]+'_'+trans[1]
        results[transName]=[]
        for idx in data.keys():
            s=data[idx][0]
            try:
                cSet=eval('Generator.'+str(trans[0])+'(s)')
                fSet=eval('Generator.'+str(trans[1])+'(s)')
            except:
                continue
            if (cSet != None) and (fSet != None):

                # for c in cSet:
                #     for f in fSet:
                c=random.choice(cSet)
                f=random.choice(fSet)
                results[transName].append((s,c,f))

        print("Finish {} : output {} unit triplets during {} seconds.".format(transName,len(results[transName]),time.time()-BeginTime))



outputPath=osp.join(BaseDir,'data/contrast_set',args.output)
WriteJson(results,outputPath)