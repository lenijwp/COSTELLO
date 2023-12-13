# %%
import torch
import os
import json
from tqdm import tqdm
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
import torch.nn as nn
import pickle
import gc
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()
import torch.nn.functional as F
import transformers.generation.logits_process as logits_process
import transformers.generation.stopping_criteria as stopping_criteria
from torch.nn import MSELoss,CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import trange

# %%
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

# %%
CACHE_DIR ='/home/lenijwp/datacache/huggingface/'
device = 'cuda:1'
tokenizer = AutoTokenizer.from_pretrained("/home/lenijwp/datacache/Model/costello/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("/home/lenijwp/datacache/Model/costello/flan-t5-base",low_cpu_mem_usage=True,device_map=device).eval()


# %%
model

# %%
hook_embeddings = []
# hook_outputs = []

def remove_hooks(model):
    for layer in model.children():
        if isinstance(layer, nn.Module):
            remove_hooks(layer)
        if hasattr(layer, '_forward_hooks'):
            layer._forward_hooks.clear()
        if hasattr(layer, '_backward_hooks'):
            layer._backward_hooks.clear()
remove_hooks(model)

def hook(module, fea_in, fea_out):
    hook_embeddings.append(fea_in)
    # hook_outputs.append(fea_out)
    return None

layer_name = 'lm_head'
for (name, module) in model.named_modules():
    if name == layer_name:
        # module.requires_grad_(True)
        module.register_forward_hook(hook=hook)

def clear_hooks():
    global hook_embeddings
    # global hook_outputs
    hook_embeddings.clear()
    # hook_outputs.clear()
    torch.cuda.empty_cache()



# %%
def get_embeddings(text):
    global hook_embeddings
    
    # input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
    encoded_input = tokenizer(
        text,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    decoder_input = tokenizer(
        text,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    model(input_ids=encoded_input["input_ids"], decoder_input_ids=decoder_input["input_ids"])
    embeddings = hook_embeddings[-1][0][:,-1,:]
    ret = []
    for i in range(embeddings.shape[0]):
        ret.append(embeddings[i].cpu().detach().numpy())
    
    clear_hooks()
    
    return ret

# %%


# %%


Dataset = LoadJson('/home/lenijwp/codes/costello/data/initial_data/sst_tokens.json')

tmp = []

Embs=[]

for key in tqdm(list(Dataset.keys()),'Collecting'):
    tmp.append(str(key))

    if len(tmp)==20:
        response = get_embeddings(tmp)
        for ele in response:
            Embs.append(ele)
    
        tmp=[]
    



        
if len(tmp)!=0:
    response = get_embeddings(tmp)
    for ele in response:
        Embs.append(ele)
    time.sleep(1)
    tmp=[]


saveEmb = np.array(Embs)
print(saveEmb.shape)

np.save('./tokens.npy',saveEmb)


Dataset = LoadJson('/home/lenijwp/codes/costello/data/initial_data/sst_train.json')

tmp = []

Embs=[]

for key in tqdm(list(Dataset.keys()),'Collecting'):
    tmp.append(Dataset[key][0])

    if len(tmp)==10:
        response = get_embeddings(tmp)
        for ele in response:
            Embs.append(ele)
   
        tmp=[]
    
        
if len(tmp)!=0:
    response = get_embeddings(tmp)
    for ele in response:
        Embs.append(ele)
    time.sleep(1)
    tmp=[]

saveEmb = np.array(Embs)
print(saveEmb.shape)

np.save('./sst2-train.npy',saveEmb)


TestSuite = LoadJson('/home/lenijwp/codes/costello/data/contrast_set/ctset1.json')

for MuType in TestSuite.keys():

    # if str(MuType)=='synon_contr':
    #     continue
    print(MuType)
    #print(len(TestSuite[MuType]))

    Data = TestSuite[MuType]

    Embs=[]



    for i in trange(len(Data)):

        response = get_embeddings(Data[i])
        Embs.append([response[0], response[1], response[2]])

 

    saveEmb = np.array(Embs)
    print(saveEmb.shape)

    np.save('./'+str(MuType)+'.npy',saveEmb)





# %%



