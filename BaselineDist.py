import os
from ossaudiodev import openmixer 
import numpy as np
import pandas as pd
from data.DataLoader import load_UEA_dataset
from evaluation.metrics import d1_distance,d2_distance
import pickle
from deap import creator, base, algorithms, tools
from deap.benchmarks.tools import hypervolume, diversity, convergence
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin,window=0)
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
for dataset in os.listdir('./Results/Benchmarking'):
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
        if os.path.exists(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TRAIN.tsv'):
            train = pd.read_csv(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TRAIN.tsv', sep='\t', header=None)
            test = pd.read_csv(os.path.abspath(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TEST.tsv'), sep='\t', header=None)
 
            train_y, train_x = train.loc[:, 0].apply(lambda x: x-1).to_numpy(), train.loc[:, 1:].to_numpy()
            test_y, test_x = test.loc[:, 0].apply(lambda x: x-1).to_numpy(), test.loc[:, 1:].to_numpy()
        else:
            train_x,train_y,test_x,test_y=load_UEA_dataset(dataset)
        i=0
        for item in pickle.load(open(f'./Results/Benchmarking/{dataset}/Wachter_cf.pkl','rb')):
            if not item is None:
                #print(item)
                d1_w.append(d1_distance(item, test_x[i]))
                d2_w.append(d2_distance(item, test_x[i]))
                i=i+1
                val=val+1
        i=0
        if os.path.exists(f'./Results/Benchmarking/{dataset}/ib_cf.pkl'):
            for item in pickle.load(open(f'./Results/Benchmarking/{dataset}/ib_cf.pkl','rb')):
                if not item is None:
                    d1_i.append(d1_distance(item, test_x[i]))
                    d2_i.append(d2_distance(item, test_x[i]))
                    i=i+1
        i=0
        if os.path.exists(f'./Results/Benchmarking/{dataset}/cfg_cf.pkl'):
            for item in pickle.load(open(f'./Results/Benchmarking/{dataset}/cfg_cf.pkl','rb')):
                if not item is None:
                    d1_c.append(d1_distance(item, test_x[i]))
                    d2_c.append(d2_distance(item, test_x[i]))
                    i=i+1
        i=0
        if os.path.exists(f'./Results/Benchmarking/{dataset}/ates_cf.pkl'):
            for item in pickle.load(open(f'./Results/Benchmarking/{dataset}/ates_cf.pkl','rb')):
                if not item is None:
                    d1_a.append(d1_distance(item, test_x[i]))
                    d2_a.append(d2_distance(item, test_x[i]))
                    i=i+1
        i=0
        while i<=19:
            item= pickle.load(open(f'./Results/mutate_both/{dataset}/Counterfactuals_{i}.pkl','rb'))
            d1.append(d1_distance(np.array(item[0]), test_x[i]))
            d2.append(d2_distance(np.array(item[0]), test_x[i]))
            i=i+1

        line_d1=line_d1+dataset+'&$'+str(round(np.mean(d1),2))+'\pm'+str(round(np.std(d1),2)) +'$&$' +str(round(np.mean(d1_w),2))+'\pm'+str(round(np.std(d1_w) ,2))+'$&$' +str(round(np.mean(d1_i),2))+'\pm'+str(round(np.std(d1_i),2))+'$&$' +str(round(np.mean(d1_c),2))+'\pm'+str(round(np.std(d1_c),2))+'$&$' +str(round(np.mean(d1_a),2))+'\pm'+str(round(np.std(d1_a),2))+'$ \\'+'\\ \hline'
        line_d2=line_d2+dataset+'&$'+str(round(np.mean(d2),2))+'\pm'+str(round(np.std(d2),2)) +'$&$' +str(round(np.mean(d2_w),2))+'\pm'+str(round(np.std(d2_w) ,2))+'$&$' +str(round(np.mean(d2_i),2))+'\pm'+str(round(np.std(d2_i),2))+'$&$' +str(round(np.mean(d2_c),2))+'\pm'+str(round(np.std(d2_c),2))+'$&$' +str(round(np.mean(d2_a),2))+'\pm'+str(round(np.std(d2_a),2))+'$ \\'+'\\ \hline'          

        
print(line_d1)
print(line_d2)