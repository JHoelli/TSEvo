'''
This section and calculations are based on :
Pawelczyk, Martin, et al. "Carla: a python library to benchmark algorithmic recourse and counterfactual explanation algorithms." arXiv preprint arXiv:2108.00783 (2021).
'''
from cProfile import label
import os
from tslearn.neighbors import KNeighborsTimeSeries
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial import distance
import pickle
from typing import List
from deap import base
from deap import creator
import torch


def d1_distance(factual: np.ndarray,counterfactual: np.ndarray) -> List[float]:
    """
    Computes D1 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """
    le=factual.shape[-1]*factual.shape[-2]
    delta=factual.reshape(-1)-counterfactual.reshape(-1)
    # compute elements which are greater than 0
    return np.sum(delta != 0, dtype=np.float)/le


def d2_distance(factual: np.ndarray,counterfactual: np.ndarray) -> List[float]:
    """
    Computes D2 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """
    le=factual.shape[-1]*factual.shape[-2]
    delta=factual.reshape(-1)-counterfactual.reshape(-1)
    return np.sum(np.abs(delta), dtype=np.float)/le


def d3_distance(factual: np.ndarray,counterfactual: np.ndarray) -> List[float]:
    """
    Computes D3 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """
    delta=factual.reshape(-1)-counterfactual.reshape(-1)
    return np.sum(np.square(np.abs(delta)), dtype=np.float).tolist()


def d4_distance(factual: np.ndarray,counterfactual: np.ndarray) -> List[float]:
    """
    Computes D4 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """
    delta=factual.reshape(-1)-counterfactual.reshape(-1)
    return np.max(np.abs(delta)).tolist()

def yNN(
    counterfactuals,
    mlmodel,data,
    y,labels=None
) :
    """
    ----------
    counterfactuals: Generated counterfactual examples
    recourse_method: Method we want to benchmark
    y: Number of
    Returns
    -------
    float
    
    """
    number_of_diff_labels = 0
    if labels==None:
        labels = [np.argmax(cf.output) for cf in counterfactuals]
    


    counterfactuals = [np.array(cf) for cf in counterfactuals]

    N = np.array(counterfactuals).shape[-1]
    M = np.array(counterfactuals).shape[-2]

    data = np.concatenate( (data.reshape(-1,M*N),np.array(counterfactuals).reshape(-1,M* N)))
    nbrs = NearestNeighbors(n_neighbors=y).fit(np.array(data))

 
    calc=[]
    for i, row in enumerate(counterfactuals):
        #print(row)
        row=row.reshape(-1,M* N)
        knn = nbrs.kneighbors(row, y, return_distance=False)[0]
        cf_label = labels[i] 

        for idx in knn:
            neighbour = data[idx] 
            neighbour = neighbour.reshape((1,M, N))

            individual = np.array(neighbour.tolist(), dtype=np.float64)
            input_ = torch.from_numpy(individual).float()

            output = torch.nn.functional.softmax(mlmodel(input_)).detach().numpy()
            neighbour_label = np.argmax(output)

            if not cf_label == neighbour_label:
                number_of_diff_labels += 1
        calc.append([1 - (1 / ( y)) * number_of_diff_labels])

    return np.array(calc)

def yNN_timeseries(
    counterfactuals,
    mlmodel,data,
    y,labels=None
) :
    """
    Parameters
    ----------
    counterfactuals: Generated counterfactual examples
    recourse_method: Method we want to benchmark
    y: Number of
    Returns
    -------
    float

    """
    number_of_diff_labels = 0
    if labels is None:
        labels = [np.argmax(cf.output) for cf in counterfactuals]

    counterfactuals = [np.array(cf) for cf in counterfactuals]

    N = np.array(counterfactuals).shape[-1]
    M= np.array(counterfactuals).shape[-2]
    data = np.concatenate( (data.reshape(-1, M,N),np.array(counterfactuals).reshape(-1, M,N)))

    data=data.reshape(-1, N,M)

    nbrs = KNeighborsTimeSeries(n_neighbors=y, metric = 'dtw')
    nbrs.fit(np.array(data))
    
    calc=[]
    p=len(counterfactuals)
    for i, row in enumerate(counterfactuals):

        knn = nbrs.kneighbors(np.array(row).reshape(1,N,M), return_distance=False)

        
        cf_label = labels[i] 

        for idx in knn:
            neighbour = data[idx]
            neighbour = neighbour.reshape((1, -1))
            individual = np.array(neighbour.tolist(), dtype=np.float64)
            input_ = torch.from_numpy(individual).float().reshape(-1,M,N)
            output = torch.nn.functional.softmax(mlmodel(input_)).detach().numpy()
            neighbour_label = np.argmax(output)
            if not cf_label == neighbour_label:
                number_of_diff_labels += 1
    
        calc.append([1 - (1 /(p* y)) * number_of_diff_labels])
    if p==1: 
        return np.array(calc)
    return  1 - (1 / (N * y)) * number_of_diff_labels


def compute_redundancy(
    fact: np.ndarray, cf: np.ndarray, mlmodel, label_value: int
) -> int:
    red = 0
    
    if len(fact.shape) ==1:
        shape=(1,1,fact.shape[0])
    else:
        shape= fact.shape

    fact=fact.reshape(-1)
    cf=cf.reshape(-1)
    for col_idx in range(cf.shape[0]):  # input array has one-dimensional shape

        if fact[col_idx] != cf[col_idx]:
            temp_cf = np.copy(cf)

            temp_cf[col_idx] = fact[col_idx]

            individual = np.array(temp_cf.tolist(), dtype=np.float64)
            input_ = torch.from_numpy(individual).float().reshape(1,shape[-2],shape[-1])
            output = torch.nn.functional.softmax(mlmodel(input_)).detach().numpy()


            temp_pred = np.argmax(output)

            if temp_pred == label_value:
                red += 1

    return red


def redundancy(original, counterfactuals, mlmodel, labels= None) :
    """
    Computes Redundancy measure for every counterfactual
    Parameters
    ----------
    factuals: Encoded and normalized factual samples
    counterfactuals: Encoded and normalized counterfactual samples
    mlmodel: Black-box-model we want to discover
    Returns
    -------
    List with redundancy values per counterfactual sample
    """
    if labels == None:
        labels = [np.argmax(cf.output) for cf in counterfactuals]
    df_cfs = np.array(counterfactuals)
    redun=[]
    for i in range (0,len(df_cfs)):
        redun.append(compute_redundancy(original,np.array(df_cfs[i]),mlmodel,labels[i]))
    return redun

def supression_test(input, item, path):
    #TODO for Today
    '''Fong, Ruth, Mandela Patrick, and Andrea Vedaldi. "Understanding deep networks via extremal perturbations and smooth masks." Proceedings of the IEEE/CVF international conference on computer vision. 2019.'''
    
    pass

def pointing_game(input, item, path):
    '''Fong, Ruth, Mandela Patrick, and Andrea Vedaldi. "Understanding deep networks via extremal perturbations and smooth masks." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
    On synthetic datsets of Ismail et. al . 
    '''
    
    pass

def calculate_all():
    pass

