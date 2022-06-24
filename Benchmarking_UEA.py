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
from evaluation.AtesEtAl import OptimizedSearch
from evaluation.Plots import plot_CF, plot_CF_Original, plot_CF_Original_Closest
from tslearn.datasets import UCR_UEA_datasets
import warnings
from evaluation.Instance_BasedCF_NativeGuide import NativeGuidCF
from deap import creator, base, algorithms, tools
from deap.benchmarks.tools import hypervolume, diversity, convergence
from tslearn.datasets import UCR_UEA_datasets
from models.ResNet import ResNetBaseline
from data.DataLoader import load_UEA_dataset
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin,window=0)

run_on = ['Heartbeat','NATOPS','PenDigits','UWaveGestureLibrary']
draw_plot=False
os_type= platform.system()
os.environ["CUDA_VISIBLE_DEVICES"]=""
mutation_type=['mutate_both']
full_method=[]
full_dataset=[]
full_ynn=[]

for dataset in run_on: 
    if not os.path.isdir(f'./Results/Benchmarking/{dataset}'):
        os.mkdir(f'./Results/Benchmarking/{dataset}')
    '''Get Data'''
    os_type= platform.system()

    
    X_train,train_y,X_test,test_y=UCR_UEA_datasets().load_dataset(dataset)
    train_x=X_train.reshape(-1,X_train.shape[-1],X_train.shape[-2])
    test_x=X_test.reshape(-1,X_train.shape[-1],X_train.shape[-2])

    enc1=pickle.load(open(f'./models/{dataset}/OneHotEncoder.pkl','rb'))
    test_y=enc1.transform(y_test.reshape(-1,1))
    train_y=enc1.transform(y_train.reshape(-1,1))
    n_classes = test_y.shape[1]
    print(n_classes)


    '''Load Model'''    

    model = ResNetBaseline(in_channels=train_x.shape[-2], num_pred_classes=n_classes)
    model.load_state_dict(torch.load(f'./models/{dataset}/ResNet'))
    model.eval()
    mlmodel=model
    '''Predict'''
    y_pred= model(torch.from_numpy(test_x).float()).detach().numpy()
    test_y=y_pred

    '''Explanation Method'''    
    comte = OptimizedSearch(mlmodel, train_x, np.argmax(train_y,axis=1), silent=False, threads=1,num_distractors=2)

    '''Calculate'''
    #CF=[]
    #log=[]
    
    ynn=[]
    ynn_timeseries=[]
    red=[]
    sal_01=[]
    sal_02=[]
    wachter_cf=[]
    ynn_wachter=[]
    ynn_timeseries_wachter=[]
    red_wachter=[]
    sal_01_wachter=[]
    sal_02_wachter=[]
    ynn_full=[]
    d=[]
    app=[]
    not_valid_wachter=0
    ates_cf=[]
    ynn_ates=[]
    ynn_timeseries_ates=[]
    red_ates=[]
    sal_01_ates=[]
    sal_02_ates=[]
    not_valid_ates=0
    max_iteration=len(test_y)
    wachter_cf_s=[]
    ates_cf_s=[]
    cfs=[]
    ys=[]
    wachter_y=[]
    ates_y=[]
    sh=test_x.shape
    for i, item in enumerate(test_x):
        print('Image Number ',{i})
        observation_01=item
        label_01=np.array([test_y[i]])#test_y[0]
        print('Label', label_01)
        if os.path.exists( f'./Results/mutate_both/{dataset}/Counterfactuals_{i}.pkl'):
            pop=pickle.load(open( f'./Results/mutate_both/{dataset}/Counterfactuals_{i}.pkl', "rb" ))
        else:
            break
        input_ = torch.from_numpy(np.array(pop)).float()
        output = torch.nn.functional.softmax(model(input_)).detach().numpy()
        y_target = output.argmax()
        t=output.argmin()
        print('Y_Target',y_target)
        mlmodel = model 
        counterfactuals = pop
        original = observation_01 
        ynn.append(yNN(counterfactuals, mlmodel,train_x,5)[0][0])
        ynn_timeseries.append(yNN_timeseries(counterfactuals, mlmodel,train_x,5)[0][0])
        red.append(redundancy(original, counterfactuals, mlmodel)[0])
        sal_01.append(d1_distance(observation_01,np.array(pop)))
        sal_02.append(d2_distance(observation_01,np.array(pop)))
        ys.append(np.argmax(pop[0].output))
        cfs.append(pop)
    
        # Wachter et al . 
        item = item.reshape(1,sh[-2],sh[-1])
        wachter_counterfactual=WachterEtAl.wachter_recourse(mlmodel, item, y_target)
        #wachter_counterfactual=wachter_counterfactual.reshape(1,sh[-2],sh[-1])
        wachter_cf.append(wachter_counterfactual)
        if not wachter_counterfactual is None:
            print(wachter_counterfactual.shape)
            wachter_cf_s.append(wachter_counterfactual)
            input_ = torch.from_numpy(np.array(wachter_counterfactual)).float().reshape(1,sh[-2],sh[-1])
    
            output = torch.nn.functional.softmax(model(input_)).detach().numpy()
            wachter_y.append(np.argmax(output))
            wachter_counterfactual=wachter_counterfactual.reshape(1,sh[-2],sh[-1])
            #wachter_couterfactual=wachter_counterfactual.reshape(np.array(pop).shape[0],np.array(pop).shape[1],np.array(pop).shape[2])
            ynn_wachter.append(yNN(wachter_counterfactual, mlmodel,train_x,5,labels=np.array([y_target]))[0][0])
            ynn_timeseries_wachter.append(yNN_timeseries(wachter_counterfactual, mlmodel,train_x,5,labels=np.array([y_target]))[0][0])
            red_wachter.append(redundancy(original, wachter_counterfactual, mlmodel,labels=np.array([y_target]))[0])
            sal_01_wachter.append(d1_distance(observation_01,np.array(wachter_counterfactual)))
            sal_02_wachter.append(d2_distance(observation_01,np.array(wachter_counterfactual)))
        else: 
            not_valid_wachter=not_valid_wachter+1

        item = item.reshape(sh[-2],sh[-1])
        explanation = comte.explain(item,to_maximize=t,savefig=False) 
        print(explanation)
        if explanation is None:
            ates_cf.append(None)
        
        if not explanation is None and not explanation==[]:
            #print(explanation)
            _,modifies=explanation
            
            #print(modifies.shape)
            #print(modifies)
            #if modifies is tuple:
            try:    
                _,modifies=modifies
            except:
                pass
            modifies=modifies.reshape(1,sh[-2],sh[-1])
            ates_cf_s.append(modifies)
            input_ = torch.from_numpy(np.array(modifies)).float().reshape(1,sh[-2],sh[-1])
    
            output = torch.nn.functional.softmax(model(input_)).detach().numpy()
            ates_y.append(np.argmax(output))
            ates_cf.append(modifies)
            #print(modifies.shape)
            ates_couterfactual=modifies.reshape(np.array(pop).shape[0],np.array(pop).shape[1],np.array(pop).shape[2])
            #ynn_ates.append(yNN(modifies, mlmodel,train_x,5,labels=np.array([y_target]))[0][0])
            ynn_timeseries_ates.append(yNN_timeseries(modifies, mlmodel,train_x,5,labels=np.array([y_target]))[0][0])
            red_ates.append(redundancy(original, modifies, mlmodel,labels=np.array([y_target]))[0])
            sal_01_ates.append(d1_distance(observation_01,np.array(modifies)))
            sal_02_ates.append(d2_distance(observation_01,np.array(modifies)))
        else: 
            not_valid_wachter=not_valid_ates+1





    full_dataset.append(dataset)
    full_dataset.append(dataset)
    full_dataset.append(dataset)
    full_method.append('Wachter')
    full_method.append('TSEvo')
    full_method.append('Ates')
    full_ynn.append(yNN_timeseries(wachter_cf_s, mlmodel,train_x,5,labels=np.array(wachter_y)))
    full_ynn.append(yNN_timeseries(cfs, mlmodel,train_x,5,labels=np.array(ys)))
    full_ynn.append(yNN_timeseries(ates_cf_s, mlmodel,train_x,5,labels=np.array(ates_y)))


    pickle.dump(wachter_cf,open(f'./Results/Benchmarking/{dataset}/Wachter_cf.pkl','wb'))
    pickle.dump(ates_cf,open(f'./Results/Benchmarking/{dataset}/ates_cf.pkl','wb'))
    #pickle.dump(cfg_counterfactual,open(f'./Results/Benchmarking/{dataset}/cfg_cf.pkl','wb'))
    dis1={}
    dis1['Wachter']=sal_01_wachter
    dis1['our']=sal_01
    dis1['Ates']=sal_01_ates
    pickle.dump(dis1, open(f'./Results/Benchmarking/{dataset}/Dis1.pkl','wb'))
    dis2= {}
    dis2['Wachter']=sal_02_wachter
    dis2['our']=sal_02
    dis2['Ates']=sal_02_ates
    #dis2['cfg']=sal_01_cfg
    pickle.dump(dis2, open(f'./Results/Benchmarking/{dataset}/Dis2.pkl','wb'))
    #TODO Problems with CF Output
    #TODO Problems with CF Output
    results = pd.DataFrame([])
    results['method']=['TS_Evo', 'Wachter', 'Ates']
    #results['ynn']=[np.mean(ynn),np.mean(ynn_wachter),np.mean(ynn_ib),np.mean(ynn_cfg)]
    #results['ynn_std']=[np.std(ynn),np.std(ynn_wachter),np.std(ynn_ib),np.std(ynn_cfg)]
    results['validity']=['Not implemented', 1-not_valid_wachter/20,1-not_valid_ates/20]
    results['ynn_timeseries']=[np.mean(ynn_timeseries),np.mean(ynn_timeseries_wachter),np.mean(ynn_timeseries_ates)]
    results['ynn_timeseries_std']=[np.std(ynn_timeseries),np.std(ynn_timeseries_wachter),np.std(ynn_timeseries_ates)]
    results['red']=[np.mean(red),np.mean(red_wachter),np.mean(red_ates)]
    results['red_std']=[np.std(red),np.std(red_wachter),np.std(red_ates)]
    results['sparsity']=[np.mean(sal_01),np.mean(sal_01_wachter),np.mean(sal_01_ates)]
    results['sparsity_std']=[np.std(sal_01),np.std(sal_01_wachter),np.std(sal_01_ates)]
    results['dis']=[np.mean(sal_02),np.mean(sal_02_wachter),np.mean(sal_02_ates)]
    results['dis_std']=[np.std(sal_02),np.std(sal_02_wachter),np.std(sal_02_ates)]
#results['closest']=sal_02
#results.to_csv(f'./Results/{dataset}/BenchmarkMetrics.csv')
    #results['closest']=sal_02
    results.to_csv(f'./Results/Benchmarking/{dataset}/BenchmarkMetrics.csv')
frame=pd.DataFrame([])
frame['Dataset']=full_dataset
frame['Method']=full_method
frame['ynn']=full_ynn

frame.to_csv('Full_ynn_UEA.csv')