import os
from ossaudiodev import openmixer
from tkinter import PIESLICE 
import numpy as np
import pandas as pd
from evaluation.metrics import d1_distance,d2_distance
import pickle
from deap import creator, base, algorithms, tools
from deap.benchmarks.tools import hypervolume, diversity, convergence
from tslearn.datasets import UCR_UEA_datasets
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin,window=0)
normelize = False
def normelize(data, min, max):
    return (data - min)/(max - min)
d1=[]
d2=[]
d1_w=[]
d2_w=[]
d1_i=[]
d2_i=[]
d1_c=[]
d2_c=[]
d1_a=[]
d2_a=[]
df={}
val=0
line_d1=''
line_d2=''
for dataset in ['GunPoint','CharacterTrajectories','Coffee','CBF','ElectricDevices','ECG5000','FordA','Heartbeat', 'UWaveGestureLibrary','NATOPS']:#os.listdir('./Results/Benchmarking'):
    d1=[]
    d2=[]
    d1_w=[]
    d2_w=[]
    d1_i=[]
    d2_i=[]
    d1_c=[]
    d2_c=[]
    d1_a=[]
    d2_a=[]
   
    if not dataset.endswith('.png') and not dataset.endswith('.csv'):
        df[dataset]={}
        X_train,train_y,X_test,test_y=UCR_UEA_datasets().load_dataset(dataset)
        train_x=X_train.reshape(-1,X_train.shape[-1],X_train.shape[-2])
        test_x=X_test.reshape(-1,X_train.shape[-1],X_train.shape[-2]) 
        train_x=np.nan_to_num(train_x, copy=True, nan=0.0)
        test_x=np.nan_to_num(test_x, copy=True, nan=0.0)
        mi = np.min(X_test)
        ma = np.max(X_test)
        #if normelize:
        #    test_x=normelize(test_x, mi, ma)
        i=0
        for item in pickle.load(open(f'./Results/Benchmarking/{dataset}/Wachter_cf.pkl','rb')):

            if not item is None:
                #if normelize:
                #    item = normelize(item, mi, ma)
                #print(item)
                d1_w.append(d1_distance(item, test_x[i]))
                d2_w.append(d2_distance(item, test_x[i]))
                i=i+1
                val=val+1
        i=0
        if os.path.exists(f'./Results/Benchmarking/{dataset}/ib_cf.pkl'):
            for item in pickle.load(open(f'./Results/Benchmarking/{dataset}/ib_cf.pkl','rb')):
                if not item is None:
                    #if normelize:
                    #    item = normelize(item, mi, ma)
                    d1_i.append(d1_distance(item, test_x[i]))
                    d2_i.append(d2_distance(item, test_x[i]))
                    i=i+1
        i=0
        if os.path.exists(f'./Results/Benchmarking/{dataset}/cfg_cf.pkl'):
            for item in pickle.load(open(f'./Results/Benchmarking/{dataset}/cfg_cf.pkl','rb')):
                if not item is None:
                    #if normelize:
                    #    item = normelize(item, mi, ma)
                    d1_c.append(d1_distance(item, test_x[i]))
                    d2_c.append(d2_distance(item, test_x[i]))
                    i=i+1
        i=0
        if os.path.exists(f'./Results/Benchmarking/{dataset}/ates_cf.pkl'):
            for item in pickle.load(open(f'./Results/Benchmarking/{dataset}/ates_cf.pkl','rb')):
                if not item is None:
                    #if normelize:
                     #   item = normelize(item, mi, ma)
                    d1_a.append(d1_distance(item, test_x[i]))
                    d2_a.append(d2_distance(item, test_x[i]))
                    i=i+1
        i=0
        while i<=19:
            item= pickle.load(open(f'./Results/mutate_both/{dataset}/Counterfactuals_{i}.pkl','rb'))
            #if normelize:
            #    item = normelize(item, mi, ma)
            d1.append(d1_distance(np.array(item[0]), test_x[i]))
            d2.append(d2_distance(np.array(item[0]), test_x[i]))
            i=i+1

        line_d1=line_d1+dataset+'&$'+str(round(np.mean(d1),2))+'\pm'+str(round(np.std(d1),2)) +'$&$' +str(round(np.mean(d1_w),2))+'\pm'+str(round(np.std(d1_w) ,2))+'$&$' +str(round(np.mean(d1_i),2))+'\pm'+str(round(np.std(d1_i),2))+'$&$' +str(round(np.mean(d1_c),2))+'\pm'+str(round(np.std(d1_c),2))+'$&$' +str(round(np.mean(d1_a),2))+'\pm'+str(round(np.std(d1_a),2))+'$ \\'+'\\ \hline '
        #Normelize the Ds
        
        line_d2=line_d2+dataset+'&$'+str(round(np.mean(d2),2))+'\pm'+str(round(np.std(d2),2)) +'$&$' +str(round(np.mean(d2_w),2))+'\pm'+str(round(np.std(d2_w) ,2))+'$&$' +str(round(np.mean(d2_i),2))+'\pm'+str(round(np.std(d2_i),2))+'$&$' +str(round(np.mean(d2_c),2))+'\pm'+str(round(np.std(d2_c),2))+'$&$' +str(round(np.mean(d2_a),2))+'\pm'+str(round(np.std(d2_a),2))+'$ \\'+'\\ \hline '          

        
print(line_d1)
print(line_d2)