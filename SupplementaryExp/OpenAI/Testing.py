import os
from urllib import response
import nlpcloud
import json
import numpy as np
from tqdm import trange,tqdm
import time
from scipy.spatial.distance import cosine
from numba import jit
from scipy import stats
import math

# @jit
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
    return dis**0.5

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



Threslist = ['0','th0','th1','th2']

for thres_type in Threslist:
    print("Testing Under thres_type: ",thres_type)

    TestSuite = LoadJson('/data/jwp/codes/nlptest/ct4plm/data/contrast_set/ctset1.json')


    tmpdis = np.load('./tokens.npy')

    wordEmb = tmpdis
    print("wordEmb.shape:",wordEmb.shape)

    worddis = EuclideanDistance(wordEmb,wordEmb)
    for i in range(worddis.shape[0]):
        worddis[i][i]=10000000


    closeDis=np.zeros(worddis.shape[0])

    for i in range(worddis.shape[0]):
        closeDis[i] = worddis[i].min()

    dist = getattr(stats, 'norm')
    parameters = dist.fit(closeDis)

    if thres_type =='th0':
        th = min(closeDis)
    elif thres_type =='th1':
        th = parameters[0]-2*math.sqrt(parameters[1])
    elif thres_type =='th2':
        th = parameters[0]-math.sqrt(parameters[1])
    elif thres_type =='0':
        th = 0
    
    if th<0:
        th=0

    print("th:",th)

    Bugs = {}

    norm = 'l2'

    cnt = 0

    for MuType in TestSuite.keys():

        # if str(MuType)=='synon_contr':
        #     continue

        Bugs[MuType] = []

        embs = np.load('./'+str(MuType)+'.npy')


        Data = TestSuite[MuType]

        for i in range(len(Data)):
            dis_c = Calculate_distance(embs[i][0],embs[i][1],norm)
            dis_f = Calculate_distance(embs[i][0],embs[i][2],norm)


            if dis_c-dis_f>th:
                Bugs[MuType].append([i,Data[i]])

    WriteJson(Bugs, f'./test_result_{thres_type}.json')
        