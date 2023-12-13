import os
import json
import argparse
from tqdm import trange,tqdm
import logging


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

def WriteJson(data,path):
    '''
    '''
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=4)

def makesure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def makesure_json(path):
    if not os.path.exists(path):
        tmp = {}
        WriteJson(tmp,path)

file = 'plm.list'

plms = readfile(file)

# norm_list = ['l2','l1','cos']
thres_list = ['min','1sigma','2sigma','zero']
# thres_list = ['zero','1sigma','2sigma']


root_dir = './results'
contrast_set_path = './data/contrast_set/contrastset.json'

train_data_sst = './data/initial_data/sst_train.json'
valid_data_sst = './data/initial_data/sst_test.json'

train_data_mrpc = './data/initial_data/mrpc_train.json'
valid_data_mrpc = './data/initial_data/mrpc_validation.json'




parser = argparse.ArgumentParser()
parser.add_argument('--norm',choices=['l2','l1','cos'],default='l2')
parser.add_argument('--gpu',type=str,default='0')
parser.add_argument('--thres',type=str,default='min',choices=['1sigma','2sigma','zero','min'])

args = parser.parse_args()


logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("./log/norm_{}_thres_{}_gpu_{}_sst.txt".format(args.norm,args.thres,args.gpu))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
 
logger.info("Start print  for norm:{}, gpu:{}, thres:{}".format(args.norm,args.gpu,args.thres))





for thres in [args.thres]:


    bug_dirs = os.path.join(root_dir, 'testout','ori_{}_{}_sstdict'.format(args.norm,thres))
    new_bug_dirs = os.path.join(root_dir, 'testout','fix_{}_{}'.format(args.norm,thres))

    makesure_dir(bug_dirs)
    makesure_dir(new_bug_dirs)

    # dc_dirs_sst = os.path.join(root_dir, 'DCs','ori','sst','{}_{}'.format(args.norm,thres))
    dc_dirs_sst = os.path.join(root_dir, 'DCs','ori','sst','l2_2sigma')
    new_dc_dirs_sst = os.path.join(root_dir, 'DCs','afterfix','sst','{}_{}'.format(args.norm,thres))
    # dc_dirs_mrpc = os.path.join(root_dir, 'DCs','ori','mrpc','{}_{}'.format(args.norm,thres))
    dc_dirs_mrpc = os.path.join(root_dir, 'DCs','ori','mrpc','l2_2sigma'.format(args.norm,thres))
    new_dc_dirs_mrpc = os.path.join(root_dir, 'DCs','afterfix','mrpc','{}_{}'.format(args.norm,thres))



    makesure_dir(dc_dirs_sst)
    makesure_dir(new_dc_dirs_sst)
    makesure_dir(dc_dirs_mrpc)
    makesure_dir(new_dc_dirs_mrpc)


    fixmodel_dir = os.path.join(root_dir, 'fixmodel','{}_{}'.format(args.norm,thres))
    makesure_dir(fixmodel_dir)

    bug_evals_path = os.path.join(root_dir, 'evaluation','ori_{}_{}_sstdict.json'.format(args.norm,thres))
    new_bug_evals_path = os.path.join(root_dir, 'evaluation','afterfix_{}_{}.json'.format(args.norm,thres))

    makesure_dir(os.path.join(root_dir, 'evaluation'))
    makesure_json(bug_evals_path)
    makesure_json(new_bug_evals_path)

    #TODO: change the cache dir before release
    dis_cache_dir = os.path.join('./data/cache', 'cache_wdis'+args.norm)



    for i in range(len(plms)):

        logger.info("Begin to deal with No.{} plm ( {} )".format(i,plms[i]))

        # if i>=1:
        #     break

        print('****************************')
        print('Begin to deal with No.{} plm ( {} )'.format(i,plms[i]))
        print('****************************')

        plm = plms[i]
        
        
        gpu_id = args.gpu


        

        mm = os.system("python Testing.py --contrastset {} --plm {} --gpu {} --outputdir {} --norm {} --thres {} --tokencache {}".format(contrast_set_path,plm, gpu_id, bug_dirs, args.norm, thres, dis_cache_dir))
        logger.info('Test ori bugs : {}'.format(mm>>8))

        
        mm = os.system("python v4-TrainDCs.py --dataset {} --validation {} --gpu {} --plm {} --output_dir {} ".format(train_data_sst,valid_data_sst,gpu_id, plms[i], dc_dirs_sst))
        logger.info('Train ori dcs on sst : {}'.format(mm>>8))

        mm = os.system("python v4-TrainDCs.py --dataset {} --validation {} --gpu {} --plm {} --output_dir {} ".format(train_data_mrpc,valid_data_mrpc,gpu_id, plms[i], dc_dirs_mrpc))
        logger.info('Train ori dcs on mrpc : {}'.format(mm>>8))



        
        mm = os.system("python Evaluation-DCs.py --contrastset {} --plm {} --gpu {} --bugs {}  --results {}  --dclf_dir {} --type {}".format(contrast_set_path,plm, gpu_id, os.path.join(bug_dirs,plm.replace('/','-')) ,bug_evals_path,dc_dirs_sst, 'sst'))
        logger.info('Eval ori dcs on sst : {}'.format(mm>>8))
        mm = os.system("python Evaluation-DCs.py --contrastset {} --plm {} --gpu {} --bugs {}  --results {}  --dclf_dir {} --type {}".format(contrast_set_path,plm, gpu_id, os.path.join(bug_dirs,plm.replace('/','-')) ,bug_evals_path,dc_dirs_mrpc, 'mrpc'))
        logger.info('Eval ori dcs on mrpc : {}'.format(mm>>8))


        continue
        if args.norm!='l2':
            continue
        
        if thres=='min':
            continue

        # if thres == 'zero':
        #     continue

        mm = os.system("python v4-Fixing.py --alldata {} --bugs {} --plm {} --gpu {} --output {} --thres {} --tokencache {}".format(contrast_set_path, os.path.join(bug_dirs,plm.replace('/','-')),plm,gpu_id, os.path.join(fixmodel_dir,plm.replace('/','-')),thres,dis_cache_dir))
        logger.info('Fixing bugs : {}'.format(mm>>8))

        
        mm = os.system("python v4-Testing-general.py --contrastset {} --plm {} --gpu {} --outputdir {} --norm {} --thres {} --tokencache {} --customodel {} --customcache {}".format(contrast_set_path,plm, gpu_id, new_bug_dirs, args.norm, thres, dis_cache_dir, plm.replace('/','-'), fixmodel_dir))
        logger.info('Test fixed bugs : {}'.format(mm>>8))


        
        mm = os.system("python v4-TrainDCs.py --dataset {} --validation {} --gpu {} --plm {} --output_dir {} --customodel {} --customcache {}".format(train_data_sst,valid_data_sst, gpu_id, plms[i], new_dc_dirs_sst,plm.replace('/','-'), fixmodel_dir))
        logger.info('Train fixed dcs on sst :{}'.format(mm>>8))

        mm = os.system("python v4-TrainDCs.py --dataset {} --validation {} --gpu {} --plm {} --output_dir {} --customodel {} --customcache {}".format(train_data_mrpc,valid_data_mrpc, gpu_id, plms[i], new_dc_dirs_mrpc,plm.replace('/','-'), fixmodel_dir))
        logger.info('Train fixed dcs on mrpc :{}'.format(mm>>8))

        
        mm =os.system("python v4-Evaluation-DCs.py --contrastset {} --plm {} --gpu {} --bugs {}  --results {}  --dclf_dir {} --type {} --customodel {} --customcache {}".format(contrast_set_path,plm, gpu_id, os.path.join(new_bug_dirs,plm.replace('/','-')) ,new_bug_evals_path, new_dc_dirs_sst,'sst',plm.replace('/','-'), fixmodel_dir))
        logger.info('Eval fixed dcs on sst :{}'.format(mm>>8))
        mm = os.system("python v4-Evaluation-DCs.py --contrastset {} --plm {} --gpu {} --bugs {}  --results {}  --dclf_dir {} --type {} --customodel {} --customcache {}".format(contrast_set_path,plm, gpu_id, os.path.join(new_bug_dirs,plm.replace('/','-')) ,new_bug_evals_path, new_dc_dirs_sst,'mrpc',plm.replace('/','-'), fixmodel_dir))
        logger.info('Eval fixed dcs on mrpc :{}'.format(mm>>8))

    
    #     break
    # break
    

    
