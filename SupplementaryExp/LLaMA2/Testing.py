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

@jit
def pairwise_euclidean_distance(matrix1, matrix2):

    # 计算第一个矩阵每个向量的平方和
    matrix1_squared = np.sum(matrix1 ** 2, axis=1, keepdims=True)

    # 计算第二个矩阵每个向量的平方和
    matrix2_squared = np.sum(matrix2 ** 2, axis=1, keepdims=True)

    # 计算两个矩阵每对向量之间的内积
    dot_product = np.dot(matrix1, matrix2.T)

    # 使用欧氏距离公式计算距离矩阵
    distance_matrix = np.sqrt(matrix1_squared + matrix2_squared.T - 2 * dot_product)
    
    return distance_matrix


def EuclideanDistance(x, y, batch_size=1000):
    """
    Calculate Euclidean Distance between two large matrices by splitting into smaller batches
    :param x: Matrix x
    :param y: Matrix y
    :param batch_size: Size of each batch
    :return: Euclidean distance matrix
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')

    dis = np.zeros((rowx, rowy))

    # Split matrices into batches based on row size
    for i in trange(0, rowx, batch_size):
        for j in range(0, rowy, batch_size):
            x_batch = x[i:min(i+batch_size, rowx)]
            y_batch = y[j:min(j+batch_size, rowy)]

            # Calculate Euclidean distance for the current batch
            xy = np.einsum('ij,kj->ik', x_batch, y_batch)
            x2 = np.einsum('ij,ij->i', x_batch, x_batch)
            y2 = np.einsum('ij,ij->i', y_batch, y_batch)
            batch_dis = x2[:, None] + y2 - 2 * xy
            batch_dis = np.sqrt(batch_dis)

            # Assign the calculated distances to the corresponding indices in the result matrix
            dis[i:min(i+batch_size, rowx), j:min(j+batch_size, rowy)] = batch_dis

    return dis

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
def ManhattanDistance(x, y):
    
    dis = np.zeros((x.shape[0],y.shape[0]))

    for i in range(x.shape[0]):
        dis[i] = np.sum(np.abs(x[i] - y), axis=1)

    return dis

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

tmpdis = np.load('./tokens.npy')

wordEmb = tmpdis
print("wordEmb.shape:",wordEmb.shape)

worddis = EuclideanDistance(wordEmb,wordEmb)
for i in range(worddis.shape[0]):
    worddis[i][i]=10000000

Threslist = ['0','th0','th1','th2']

for thres_type in Threslist:
    print("Testing Under thres_type: ",thres_type)

    TestSuite = LoadJson('/data/jwp/codes/nlptest/ct4plm/data/contrast_set/ctset1.json')


    


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

    norm = 'l1'

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
        