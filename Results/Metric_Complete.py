import os
from tkinter import Y 
import pandas as pd
import numpy as np 

red_mean=[]
red_std=[]
yNN_mean=[]
yNN_std=[]
yNN_timeseries_mean=[]
yNN_timeseries_std=[]
spa_org=[]
spa_std=[]
spa_clo=[]
spa_clo_std=[]
datas=[]
print(os.getcwd())
paths=['./authentic_opposing_information','./frequency_band_mapping','./mutate_mean','./mutate_both']
for path in paths:
    for dataset in os.listdir(path):
        if os.path.isdir(f'./{path}/{dataset}'):
            data= pd.read_csv(f'./{path}/{dataset}/Metrics.csv')
            print(dataset)
            datas.append(dataset)
            #red_mean.append(np.mean(data['red']))
            #red_std.append(np.std(data['red']))
            #yNN_mean.append(np.mean(data['ynn']))
            #yNN_std.append(np.std(data['ynn']))
            yNN_timeseries_mean.append(np.mean(data['ynn_timeseries']))
            yNN_timeseries_std.append(np.std(data['ynn_timeseries']))
            #spa_org.append(np.mean(data['sparsity']))
            #spa_std.append(np.std(data['sparsity']))
            #spa_clo.append(np.mean(data['closest']))
            #spa_clo_std.append(np.std(data['closest']))
    frame=pd.DataFrame([])
    frame['dataset']=datas
    #frame['redundancy']=red_mean
    #frame['redundancy_std']=red_std
    #frame['yNN']=yNN_mean
    #frame['yNN_std']=yNN_std
    frame['yNN_timeseries']=yNN_timeseries_mean
    frame['yNN_timeseries_std']=yNN_timeseries_std
    #frame['sparsity']=spa_org
    #frame['sparsity_std']=spa_std
    #frame['closest']=spa_clo
    #frame['closest_std']=spa_clo_std
    frame.to_csv(f'./{path}/Summary.csv')


