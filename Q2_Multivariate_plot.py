from cProfile import label
from turtle import ontimer
from tslearn.datasets import UCR_UEA_datasets
import pickle 
import matplotlib.pyplot as plt
import numpy as np
from deap import creator, base, algorithms, tools
from deap.benchmarks.tools import hypervolume, diversity, convergence
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin,window=0)

run_on =['NATOPS'] #['Heartbeat','NATOPS','CharacterTrajectories','UWaveGestureLibrary']
#TODO Load model for classificaton
# TODO Titel
num_changed_timeseries=0
num_changed_timeseries_a=0
dataset=[]
draw_a=[]
draw_our=[]
draw_a_org=[]
draw_our_org=[]
for dataset in run_on: 
    X_train,train_y,X_test,test_y=UCR_UEA_datasets().load_dataset(dataset)
    original= X_test[0].reshape(X_test.shape[-1],X_test.shape[-2])
    original_y=test_y[0]
    cf_our = np.array(pickle.load(open(f'./Results/mutate_both/{dataset}/Counterfactuals_0.pkl','rb'))).reshape(original.shape[-2],original.shape[-1])
    ates=np.array(pickle.load(open(f'./Results/Benchmarking/{dataset}/ates_cf.pkl','rb')))[0].reshape(original.shape[-2],original.shape[-1])
    #wachter=np.array(pickle.load(open(f'./Results/Benchmarking/{dataset}/Wachter_cf.pkl','rb')))
    
    #if wachter is not None:
    #    wachter=wachter[0].reshape(original.shape[-2],original.shape[-1])
    fig,axs=plt.subplots(len(original),1,sharex=True)
    i=0
    for item in cf_our:
        print(i)
        axs[i].plot(item,color='y',label='CF')
        axs[i].plot(original[i], color='b', label='Original')
        
        #axs[i].set_ylabel(f'Feature {i}')
    
        i=i+1
    plt.legend()
    plt.show()
    fig,axs=plt.subplots(len(original),1, sharex=True)
    
    i=0
    for item in ates:
        print(i)
        
        axs[i].plot(item,color='y',label='CF')
        axs[i].plot(original[i], color='b', label='Original')
        
        #axs[i].set_ylabel(f'Feature {i}')
        i=i+1
    plt.legend()
    plt.show()
    i=0
    #if wachter is not None:
       # for item in wachter:
       #     print(i)
        
          #  axs[i].plot(item,color='y',label='CF')
         #   axs[i].plot(original[i], color='b', label='Original')
        #
         #   axs[i].set_ylabel(f'Feature {i}')
         #   i=i+1
        #plt.legend()
        #plt.show()