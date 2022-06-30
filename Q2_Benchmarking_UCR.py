import imp
from evaluation.metrics import redundancy, yNN, d1_distance , d2_distance
from cProfile import label
from datetime import datetime
import pandas as pd
import os
from evaluation.Plots import plot_basic_dataset
from evaluation.metrics import yNN_timeseries
import numpy as np
from pathlib import Path
import platform
import sklearn
import torch
from models.CNN_TSNet import UCRDataset, train
from models.ResNet import ResNetBaseline, fit, get_all_preds
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
from TSEvo.CounterfactualExplanation import Explanation
from evaluation import WachterEtAl
import pickle
from evaluation.Plots import plot_CF, plot_CF_Original, plot_CF_Original_Closest
from tslearn.datasets import UCR_UEA_datasets
import warnings
from evaluation.Instance_BasedCF_NativeGuide import NativeGuidCF
from deap import creator, base, algorithms, tools
from deap.benchmarks.tools import hypervolume, diversity, convergence
from tslearn.datasets import UCR_UEA_datasets
from evaluation.Wachter_CF import Wachter
from evaluation.Nun_CF import NativeGuideCF

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin,window=0)

mutation='mutate_both'
run_on = ['Coffee','CBF','ElectricDevices','ECG5000','GunPoint','FordA'] 
draw_plot=False
os_type= platform.system()
os.environ["CUDA_VISIBLE_DEVICES"]=""
#mutation_type=['mutate_both']

for dataset in run_on: 
    if not os.path.isdir(f'./Results/Benchmarking/{dataset}'):
        os.mkdir(f'./Results/Benchmarking/{dataset}')
        
        if dataset in os.listdir('./Results/'):
            pass
        else:
            os.mkdir(f'./Results/{dataset}')
    X_train,train_y,X_test,test_y=UCR_UEA_datasets().load_dataset(dataset)
    train_x=X_train.reshape(-1,X_train.shape[-1],X_train.shape[-2])
    test_x=X_test.reshape(-1,X_train.shape[-1],X_train.shape[-2]) 
    enc1=pickle.load(open(f'./models/{dataset}/OneHotEncoder.pkl','rb'))
    test_y=enc1.transform(test_y.reshape(-1,1))
    n_classes = test_y.shape[1]
    if len(train_x.shape)==2:
        train_x=train_x.reshape(-1,train_x.shape[-2],train_x.shape[-1])
        test_x=test_x.reshape(-1,train_x.shape[-2],train_x.shape[-1])
    '''Load Model'''
    model = ResNetBaseline(in_channels=train_x.shape[-2], num_pred_classes=n_classes)
    model.load_state_dict(torch.load(f'./models/{dataset}/ResNet'))
    model.eval()
    
    y_pred= model(torch.from_numpy(test_x).float()).detach().numpy()
    test_y=y_pred
    '''Explanation Method'''
  
    '''Initialize Methods'''
    nguide_cf=NativeGuideCF(model,np.array(train_x).shape, (test_x,test_y))

    '''Calculate'''
    ynn=[]
    ynn_timeseries=[]
    red=[]
    sal_01=[]
    sal_02=[]
    not_valid=0
    wachter_cf=[]
    ynn_wachter=[]
    ynn_timeseries_wachter=[]
    red_wachter=[]
    sal_01_wachter=[]
    sal_02_wachter=[]
    not_valid_wachter=0
    ynn_cfg=[]
    ynn_timeseries_cfg=[]
    red_cfg=[]
    sal_01_cfg=[]
    sal_02_cfg=[]
    cfg_cf=[]
    not_valid_cfg=0
    ynn_ib=[]
    ynn_timeseries_ib=[]
    red_ib=[]
    sal_01_ib=[]
    sal_02_ib=[]
    ib_cf=[]
    not_valid_ib=0
    max_iteration=len(test_y)
    #TODO add time Measure
    for i, item in enumerate(test_x):
        print('Image Number ',{i})
        observation_01=item
        label_01=np.array([test_y[i]])#test_y[0]
        if os.path.exists( f'./Results/{mutation}/{dataset}/Counterfactuals_{i}.pkl'):
            pop=pickle.load(open( f'./Results/{mutation}/{dataset}/Counterfactuals_{i}.pkl', "rb" ))
        else:
            break
        #input_ = torch.from_numpy(np.array(pop)).float()
        #output = torch.nn.functional.softmax(model(input_)).detach().numpy()
        y_target =np.argmax(test_y[i]) #output.argmax()
        mlmodel = model 
        counterfactuals = pop
        original = observation_01
        #print(y_target)
        #print(counterfactuals[0].output) 
        if y_target == np.argmax(counterfactuals[0].output):
            not_valid=not_valid+1
        ynn.append(yNN(counterfactuals, mlmodel,train_x,5)[0][0])
        ynn_timeseries.append(yNN_timeseries(counterfactuals, mlmodel,train_x,5)[0][0])
        red.append(redundancy(original, counterfactuals, mlmodel)[0])
        sal_01.append(d1_distance(observation_01,np.array(pop)))
        sal_02.append(d2_distance(observation_01,np.array(pop)))
    
        # Wachter et al . 
        #print(item.shape)
        #print(observation_01.shape)
        item = item.reshape(1,item.shape[-2],item.shape[-1])
        #print(type(item))
        print(item is None)
        #print(WachterEtAl)
        wachter_counterfactual,laberl_w=WachterEtAl.wachter_recourse(mlmodel, item)
        #w=Wachter(model,(test_x,test_y))
        #wachter_counterfactual, laberl_w=w.explain(item)
        #wachter_counterfactual, laberl_w=WachterEtAl.wachter_recourse(mlmodel, item, y_target)
        wachter_cf.append(wachter_counterfactual)
        if not wachter_counterfactual is None:
            wachter_couterfactual=wachter_counterfactual.reshape(np.array(pop).shape[0],np.array(pop).shape[1],np.array(pop).shape[2])
            ynn_wachter.append(yNN(wachter_counterfactual, mlmodel,train_x,5,labels=np.array([laberl_w]))[0][0])
            ynn_timeseries_wachter.append(yNN_timeseries(wachter_counterfactual, mlmodel,train_x,5,labels=np.array([laberl_w]))[0][0])
            red_wachter.append(redundancy(original, wachter_counterfactual, mlmodel,labels=np.array([laberl_w]))[0])
            sal_01_wachter.append(d1_distance(observation_01,np.array(wachter_counterfactual)))
            sal_02_wachter.append(d2_distance(observation_01,np.array(wachter_counterfactual)))
            if laberl_w == np.argmax(label_01,axis=1):
                not_valid_wachter=not_valid_wachter+1
        else: 
            print('Wachter not a valid CF!')
            not_valid_wachter=not_valid_wachter+1
            #Other Approach 

        item = item.reshape(1,item.shape[-2],item.shape[-1])
        cfg_counterfactual,label_cfg=nguide_cf.explain(item,  y_target)#(mlmodel, item, y_target).reshape(np.array(pop).shape[0],np.array(pop).shape[1],np.array(pop).shape[2])
        cfg_cf.append(cfg_counterfactual)
        if not cfg_counterfactual is None:
            print(cfg_counterfactual.shape)
            ynn_cfg.append(yNN(cfg_counterfactual, mlmodel,train_x,5,labels=np.array([label_cfg]))[0][0])
            ynn_timeseries_cfg.append(yNN_timeseries(cfg_counterfactual, mlmodel,train_x,5,labels=np.array([label_cfg]))[0][0])
            red_cfg.append(redundancy(original, cfg_counterfactual, mlmodel,labels=np.array([label_cfg]))[0])
            sal_01_cfg.append(d1_distance(observation_01,np.array(cfg_counterfactual)))
            sal_02_cfg.append(d2_distance(observation_01,np.array(cfg_counterfactual)))
            if label_cfg == np.argmax(label_01,axis=1):
                not_valid_cfg=not_valid_cfg+1

        else: 
            print('GradCam not a valid CF!')
            not_valid_cfg=not_valid_cfg+1

        #Other 2 
        item = item.reshape(1,item.shape[-2],item.shape[-1])
        reference_set=(train_x,train_y)
        ib_counterfactual,label_ib=nguide_cf.explain(item,  y_target,method='dtw_bary_center')#(mlmodel, item, y_target).reshape(np.array(pop).shape[0],np.array(pop).shape[1],np.array(pop).shape[2])
        ib_cf.append(ib_counterfactual)
    
        if not ib_counterfactual is None:
            ib_counterfactual=ib_counterfactual.reshape(1,1,-1)
            ynn_ib.append(yNN(ib_counterfactual,mlmodel,train_x,5,labels=np.array([label_ib]))[0][0])
            ynn_timeseries_ib.append(yNN_timeseries(ib_counterfactual, mlmodel,train_x,5,labels=np.array([label_ib]))[0][0])
            red_ib.append(redundancy(original, ib_counterfactual, mlmodel,labels=np.array([label_ib]))[0])
            sal_01_ib.append(d1_distance(observation_01,np.array(ib_counterfactual)))
            sal_02_ib.append(d2_distance(observation_01,np.array(ib_counterfactual)))
            if label_ib == np.argmax(label_01,axis=1):
                not_valid_ib=not_valid_ib+1
        else: 
            print('Instance-Base not a valid CF!')
            not_valid_ib=not_valid_ib+1
        
        if i ==19:
            break


    pickle.dump(wachter_cf,open(f'./Results/Benchmarking/{dataset}/Wachter_cf.pkl','wb'))
    pickle.dump(ib_cf,open(f'./Results/Benchmarking/{dataset}/ib_cf.pkl','wb'))
    pickle.dump(cfg_cf,open(f'./Results/Benchmarking/{dataset}/cfg_cf.pkl','wb'))
    dis1={}
    dis1['Wachter']=sal_01_wachter
    dis1['our']=sal_01
    dis1['ib']=sal_01_ib
    dis1['cfg']=sal_01_cfg
    pickle.dump(dis1, open(f'./Results/Benchmarking/{dataset}/Dis1.pkl','wb'))
    dis2= {}
    dis2['Wachter']=sal_02_wachter
    dis2['our']=sal_02
    dis2['ib']=sal_02_ib
    dis2['cfg']=sal_02_cfg
    pickle.dump(dis2, open(f'./Results/Benchmarking/{dataset}/Dis2.pkl','wb'))
    #TODO Problems with CF Output
    results = pd.DataFrame([])
    results['method']=['TS_Evo', 'Wachter', 'NG_DBN', 'NG_GradCam']
    #results['ynn']=[np.mean(ynn),np.mean(ynn_wachter),np.mean(ynn_ib),np.mean(ynn_cfg)]
    #results['ynn_std']=[np.std(ynn),np.std(ynn_wachter),np.std(ynn_ib),np.std(ynn_cfg)]
    results['validity']=[1-not_valid/20, 1-not_valid_wachter/20,1-not_valid_ib/20,1-not_valid_cfg/20]
    results['ynn_timeseries']=[np.mean(ynn_timeseries),np.mean(ynn_timeseries_wachter),np.mean(ynn_timeseries_ib),np.mean(ynn_timeseries_cfg)]
    results['ynn_timeseries_std']=[np.std(ynn_timeseries),np.std(ynn_timeseries_wachter),np.std(ynn_timeseries_ib),np.std(ynn_timeseries_cfg)]
    #results['red']=[np.mean(red),np.mean(red_wachter),np.mean(red_ib),np.mean(red_cfg)]
    #results['red_std']=[np.std(red),np.std(red_wachter),np.std(red_ib),np.std(red_cfg)]
    results['sparsity']=[np.mean(sal_01),np.mean(sal_01_wachter),np.mean(sal_01_ib),np.mean(sal_01_cfg)]
    results['sparsity_std']=[np.std(sal_01),np.std(sal_01_wachter),np.std(sal_01_ib),np.std(sal_01_cfg)]
    results['dis']=[np.mean(sal_02),np.mean(sal_02_wachter),np.mean(sal_02_ib),np.mean(sal_02_cfg)]
    results['dis_std']=[np.std(sal_02),np.std(sal_02_wachter),np.std(sal_02_ib),np.std(sal_02_cfg)]
    #results['closest']=sal_02
    results.to_csv(f'./Results/Benchmarking/{dataset}/BenchmarkMetrics.csv')

    # Calculat Full yNN
    #d.append(dataset)
    #ap.append('Wachter')
    #ap.append('TSEvo')
    #ap.append('cfg')
    #ap.append('ib')
    #cf_full.append()





