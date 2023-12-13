import os
from urllib import response
import nlpcloud
import json
import numpy as np
from tqdm import trange,tqdm
import time
import requests

def LoadJson(path):
    '''
    '''
    res=[]
    with open(path,mode='r',encoding='utf-8') as f:
        dicts = json.load(f)
        res=dicts
    return res

def WriteJson(data,path):
    '''
    '''
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=4)


import dashscope
from http import HTTPStatus
from dashscope import TextEmbedding

def queryOPENAI(text):
    # Define the URL and request headers
    url = "https://gateway.ai.cloudflare.com/v1/2ebae02146f70e97e4eb8dd39b9b1682/lenicodes/openai/embeddings"
    headers = {
        "Authorization": "{your key}",
        "Content-Type": "application/json"
    }

    # Define the request payload
    payload = {
        "model": "text-embedding-ada-002",
        "input": text
    }

    # Convert the payload to JSON format
    payload_json = json.dumps(payload)
    # for i in range(20):
    while True:
    # Send the POST request
        

        # Check the response
        
        try:
            # response = requests.post(url, headers=headers, data=payload_json, timeout=30, proxies={'http': 'http://127.0.0.1:20171'})
            response = requests.post(url, headers=headers, data=payload_json, timeout=30)
            if response.status_code == 200:
                # Request was successful
                result = response.json()
                return result['data']
            else:
                # Request failed
                # return "Error: " + str(response.status_code)
                time.sleep(20)
        except requests.exceptions.Timeout:
            # Request timed out
            time.sleep(3)
        except requests.exceptions.RequestException as e:
            time.sleep(20)
    # if resp.status_code == HTTPStatus.OK:
    #     print(resp)
    # else:
    #     print(resp)






Dataset = LoadJson('../../data/initial_data/sst_tokens.json')

tmp = []

Embs=[]

for key in tqdm(list(Dataset.keys()),'Collecting'):
    tmp.append(str(key))

    if len(tmp)==20:
        response = queryOPENAI(tmp)
        for ele in response:
            Embs.append(ele['embedding'])
        time.sleep(1)
        tmp=[]
    



        
if len(tmp)!=0:
    response = queryOPENAI(tmp)
    for ele in response:
        Embs.append(ele['embedding'])
    time.sleep(1)
    tmp=[]


saveEmb = np.array(Embs)
print(saveEmb.shape)

np.save('./tokens.npy',saveEmb)


Dataset = LoadJson('../../data/initial_data/sst_train.json')

tmp = []

Embs=[]

for key in tqdm(list(Dataset.keys()),'Collecting'):
    tmp.append(Dataset[key][0])

    if len(tmp)==10:
        response = queryOPENAI(tmp)
        for ele in response:
            Embs.append(ele['embedding'])
        time.sleep(1)
        tmp=[]
    
        
if len(tmp)!=0:
    response = queryOPENAI(tmp)
    for ele in response:
        Embs.append(ele['embedding'])
    time.sleep(1)
    tmp=[]

saveEmb = np.array(Embs)
print(saveEmb.shape)

np.save('./sst2-train.npy',saveEmb)


TestSuite = LoadJson('../../data/contrast_set/ctset1.json')

for MuType in TestSuite.keys():

    # if str(MuType)=='synon_contr':
    #     continue
    print(MuType)
    #print(len(TestSuite[MuType]))

    Data = TestSuite[MuType]

    Embs=[]



    for i in trange(len(Data)):

        response = queryOPENAI(Data[i])
        Embs.append([response[0]['embedding'], response[1]['embedding'], response[2]['embedding']])

        time.sleep(1.1)

    saveEmb = np.array(Embs)
    print(saveEmb.shape)

    np.save('./'+str(MuType)+'.npy',saveEmb)



