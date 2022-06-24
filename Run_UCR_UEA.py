from evaluation.metrics import redundancy, yNN 
from cProfile import label
from datetime import datetime
import pandas as pd
import os
from evaluation.metrics import yNN_timeseries
import numpy as np
import platform
import torch
import numpy as np
from models.ResNet import ResNetBaseline, get_all_preds
from TSEvo.CounterfactualExplanation import Explanation
import pickle
from evaluation.Plots import plot_CF, plot_CF_Original, plot_CF_Original_Closest
from tslearn.datasets import UCR_UEA_datasets

run_on = ['GunPoint','Coffee','CBF','ElectricDevices','ECG5000','FordA','Heartbeat','PenDigits', 'UWaveGestureLibrary','NATOPS']
os_type= platform.system()
os.environ["CUDA_VISIBLE_DEVICES"]=""
mutation_type=['authentic_opposing_information','frequency_band_mapping','mutate_mean','mutate_both']

for dataset in run_on: 
    '''Get Data'''
    X_train,train_y,X_test,test_y=UCR_UEA_datasets().load_dataset(dataset)
    train_x=X_train.reshape(-1,X_train.shape[-1],X_train.shape[-2])
    test_x=X_test.reshape(-1,X_train.shape[-1],X_train.shape[-2])
    enc1=pickle.load(open(f'./models/{dataset}/OneHotEncoder.pkl','rb'))
    test_y=enc1.transform(test_y.reshape(-1,1))
    n_classes = test_y.shape[1]
    print(X_train.shape[-1])
    print(n_classes)

    '''Load Model'''
    model = ResNetBaseline(in_channels=X_train.shape[-1], num_pred_classes=n_classes)
    model.load_state_dict(torch.load(f'./models/{dataset}/ResNet'))
    model.eval()
    print(test_x.shape)
    y_pred= model(torch.from_numpy(test_x).float()).detach().numpy()
    '''Run Algo'''
    exp =Explanation(model= model,data=(test_x,np.argmax(y_pred,axis=-1)),backend='PYT',verbose=2)
    ynn=[]
    ynn_timeseries=[]
    red=[]
    sal_01=[]
    sal_02=[] 
    max_iteration=len(test_y)
    for mut in mutation_type:
        if mut in os.listdir('./Results/'):
            pass
        else:
            os.mkdir(f'./Results/{mut}')
        if dataset in os.listdir(f'./Results/{mut}'):
            pass
        else:
            os.mkdir(f'./Results/{mut}/{dataset}')
        for i, item in enumerate(test_x):
            print(f'Dataset {dataset} Iteration {i}/{max_iteration}')
            observation_01=item
            label_01=np.array([y_pred[i]])#test_y[0]
            print(label_01)
            start_time=datetime.now()
            pop,logbook, window, mutation=exp.explain_instance(observation_01,label_01,transformer=mut)
            end_time=datetime.now()

            
            mlmodel = model 
            counterfactuals = pop
            pickle.dump(end_time-start_time,open( f'./Results/{mut}/{dataset}/Time_{i}.pkl', "wb" ) )
            pickle.dump( counterfactuals, open( f'./Results/{mut}/{dataset}/Counterfactuals_{i}.pkl', "wb" ) )
            pickle.dump( logbook, open( f'./Results/{mut}/{dataset}/Logbook_{i}.pkl', "wb" ) )
            pickle.dump(window,open( f'./Results/{mut}/{dataset}/Window_{i}.pkl', "wb" ))
            if mut == 'mutate_both':
                pickle.dump(mutation,open( f'./Results/{mut}/{dataset}/Mut_{i}.pkl', "wb" ))

            
            original = observation_01 
            ynn.append(yNN(counterfactuals, mlmodel,train_x,5)[0][0])
            ynn_timeseries.append(yNN_timeseries(counterfactuals, mlmodel,train_x,5)[0][0])
            red.append(redundancy(original, counterfactuals, mlmodel)[0])
            sal_01.append(np.count_nonzero(np.abs(observation_01.reshape(-1)-np.array(pop)[0].reshape(-1)).reshape(1,-1)))

            # Closest from opprosite class

            data= test_x[np.argmax(y_pred,axis=1) != np.argmax(label_01,axis= 1) ]
            l= y_pred[np.argmax(y_pred,axis=1) != np.argmax(label_01,axis= 1)]
            timeline_max=[]
            mi_max=5
            j= 0
            i_max=0
            for timeline in data: 
                mi = np.sum(np.abs(timeline.reshape(-1)- observation_01.reshape(-1) ))/150
                if mi <mi_max: 
                    mi_max=mi
                    timeline_max= timeline
                    i_max=j
                j = j+1
            sal_02.append(np.count_nonzero(np.abs(timeline_max.reshape(-1)-np.array(pop)[0].reshape(-1)).reshape(1,-1)))

            #if draw_plot:
            #    plot_CF(pop,path=f'./Results/{mut}/{dataset}/Only_CF_{i}.png')
            #    plot_CF_Original_Closest(pop,observation_01,label_01,timeline_max, l[i_max], path=f'./Results/{mut}/{dataset}Original_CF_Closest_{i}.png')
            #    pop=np.array(pop)[0][0]
            #    plot_CF_Original(pop, observation_01,label_01, path=f'./Results/{mut}/{dataset}/Original_CF_{i}.png')
                
            if i==19:
                print('Stop Run')
                break

        results = pd.DataFrame([])
        results['ynn']=ynn
        results['ynn_timeseries']=ynn_timeseries
        results['red']=red
        results['sparsity']=sal_01
        results['closest']=sal_02
        results.to_csv(f'./Results/{mut}/{dataset}/Metrics.csv')