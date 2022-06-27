import pickle
from tty import CFLAG
import matplotlib.pyplot as plt
from evaluation.WachterEtAl import wachter_recourse
from evaluation.metrics import yNN_timeseries
from models.CNN_TSNet import train
import seaborn as sns 
import os
import pandas as pd 
import numpy as np
import torch
from models.ResNet import ResNetBaseline
from tslearn.datasets import UCR_UEA_datasets
from deap import creator, base, algorithms, tools
from deap.benchmarks.tools import hypervolume, diversity, convergence
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin,window=0)

def calculate_total_ynn():
    data=[]
    method=[]
    value=[]
    for file in ['Coffee','CBF','ElectricDevices','ECG5000','GunPoint','FordA']:
        if not file.endswith('csv') and not file.endswith('png'):
            pop=[]
            y=[]
            cf=[]
            for a in os.listdir(f'./Results/mutate_both/{file}'):
                #print(a)
                if a.startswith('Counter'):
                    cf=pickle.load(open(f'./Results/mutate_both/{file}/{a}','rb'))
                    y.append(np.argmax(cf[0].output))
                    #print(np.array(cf[0][0]).shape)
                    pop.append(np.array(cf[0]))
            
            shape= np.array(cf).shape
            #print(shape)
            if shape[-2]==1:
                train_x,train_y,_,_=load_UCR_dataset(file)
            else: 
                train_x,train_y,_,_=load_UEA_dataset(file)
                pass
            
            wachter=pickle.load(open(f'./Results/Benchmarking/{file}/Wachter_cf.pkl','rb'))
            n=[]
            for a in wachter:
                    if not a is None:
                        n.append(a)
                    else:
                        print('None')
            wachter=n
            
            model = ResNetBaseline(in_channels=shape[-2], num_pred_classes=len(np.unique(train_y)))
            model.load_state_dict(torch.load(f'./models/{file}/ResNet'))
            model.eval()
            input_ = torch.from_numpy(np.array(wachter)).float().reshape(-1,shape[-2],shape[-1])
            output = torch.nn.functional.softmax(model(input_)).detach().numpy()
            y_wachter = np.argmax(output,axis=1)
            
            if shape[-2]==1:
                cfg=pickle.load(open(f'./Results/Benchmarking/{file}/cfg_cf.pkl','rb'))
                #print(len(cfg))
                n=[]
                for a in cfg:
                    if not a is None:
                        n.append(a)
                    else:
                        print('None')
                cfg=n
                ib=pickle.load(open(f'./Results/Benchmarking/{file}/ib_cf.pkl','rb'))
                n=[]
                for a in ib:
                    if not a is None:
                        n.append(a)
                print(np.array(pop).shape)
                ib=n
                #print(len(cfg))
                input_ = torch.from_numpy(np.array(cfg,dtype=np.float32)).float().reshape(-1,shape[-2],shape[-1])
                output = torch.nn.functional.softmax(model(input_)).detach().numpy()
                cfg_y = np.argmax(output,axis=1)
                input_ = torch.from_numpy(np.array(ib)).float().reshape(-1,shape[-2],shape[-1])
                output = torch.nn.functional.softmax(model(input_)).detach().numpy()
                ib_y = np.argmax(output,axis=1)
                data.append(file)
                data.append(file)
                method.append('cfg')
                method.append('ib')
                print(np.array(cfg).shape)
                if len(np.array(cfg).shape)==2:
                    cfg=np.array(cfg).reshape(1,shape[-2],shape[-1])
                if len(np.array(ib).shape)==2:
                    ib=np.array(ib).reshape(1,shape[-2],shape[-1])
                else:
                    ib=np.array(ib).reshape(-11,shape[-2],shape[-1])
                value.append(yNN_timeseries(cfg, model,train_x,5,labels=cfg_y))
                value.append(yNN_timeseries(ib, model,train_x,5,labels=ib_y))

            data.append(file)
            data.append(file)
            method.append('TSEvo')
            method.append('wachter')
            value.append(yNN_timeseries(pop, model,train_x,5,labels=y))
            if len(np.array(wachter).shape)==2:
                    cfg=np.array(wachter).reshape(1,shape[-2],shape[-1])
            value.append(yNN_timeseries(wachter, model,train_x,5,labels=y_wachter))
            
    results=pd.DataFrame([])
    results['Dataset']=data
    results['Method']=method
    results['yNN']=value
    results.to_csv('./Results/Benchmarking/yNN_base_UCR.csv')
            

def calculate_total_dis():
    data=[]
    value_std=[]
    method=[]
    value=[]
    data2=[]
    value_std2=[]
    method2=[]
    value2=[]
    for file in os.listdir('./Results/Benchmarking'):
        if not file.endswith('csv') and not file.endswith('png'):
            d1= pickle.load(open(f'./Results/Benchmarking/{file}/Dis1.pkl','rb'))
            d2= pickle.load(open(f'./Results/Benchmarking/{file}/Dis2.pkl','rb'))
            for c in d1.keys():
                value.append(np.mean(d1[c]))
                value_std.append(np.std(d1[c]))
                data.append(file)
                method.append(c)
                value2.append(np.mean(d2[c]))
                value_std2.append(np.std(d2[c]))
                data2.append(file)
                method2.append(c)
    results=pd.DataFrame([])
    results['Dataset']=data
    results['Method']=method
    results['Sparsity']=value
    results['Sparsity_STD']=value_std
    results['Proximity']=value2
    results['Proximity_STD']=value_std2
    results.to_csv('./Results/Benchmarking/dis_base.csv')

def dis_to_latex():
    rem=pd.read_csv('./Results/Benchmarking/dis_base.csv')
    for d in np.unique(rem['Dataset']):
        line=d
        data=rem[rem['Dataset']==d]
        line=line +'&$'+str(data[data['Method']=='our']['Sparsity'].round(4).values[0]) +' \pm ' +str(data[data['Method']=='our']['Sparsity_STD'].round(4).values[0])+'$&$'+str(data[data['Method']=='our']['Proximity'].round(4).values[0]) +' \pm ' +str(data[data['Method']=='our']['Proximity_STD'].round(4).values[0])+'$&'
        if not len(data[data['Method']=='Wachter']['Sparsity'].values)==0:
            line=line +'$' +str(data[data['Method']=='Wachter']['Sparsity'].round(4).values[0]) +' \pm ' +str(data[data['Method']=='Wachter']['Sparsity_STD'].round(4).values[0])+'$&$'+str(data[data['Method']=='Wachter']['Proximity'].round(4).values[0]) +' \pm ' +str(data[data['Method']=='Wachter']['Proximity_STD'].round(4).values[0])+'$&'
        else:
            line=line+'&'+'&'
        if not len(data[data['Method']=='ib']['Sparsity'].values)==0:
            line=line +'$' +str(data[data['Method']=='ib']['Sparsity'].round(4).values[0]) +' \pm ' +str(data[data['Method']=='ib']['Sparsity_STD'].round(4).values[0])+'$&$'+str(data[data['Method']=='ib']['Proximity'].round(4).values[0]) +' \pm ' +str(data[data['Method']=='ib']['Proximity_STD'].round(4).values[0])+'$&'
        else:
            line=line+'&'+'&'
        if not len(data[data['Method']=='cfg']['Sparsity'].values)==0:
            line=line +'$' +str(data[data['Method']=='cfg']['Sparsity'].round(4).values[0])+' \pm ' +str(data[data['Method']=='cfg']['Sparsity_STD'].round(4).values[0])+'$&$'+str(data[data['Method']=='cfg']['Proximity'].round(4).values[0]) +' \pm ' +str(data[data['Method']=='cfg']['Proximity_STD'].round(4).values[0])+'$&'
        else:
            line=line+'&'+'&'
        if not len(data[data['Method']=='Ates']['Sparsity'].values)==0:
            line=line  +'$'+str(data[data['Method']=='Ates']['Sparsity'].round(4).values[0]) +' \pm ' +str(data[data['Method']=='Ates']['Sparsity_STD'].round(4).values[0])+'$&$'+str(data[data['Method']=='Ates']['Proximity'].round(4).values[0]) +' \pm ' +str(data[data['Method']=='Ates']['Proximity_STD'].round(4).values[0])+'$ \\'+'\\'+' \hline'
        else:
            line=line+'&'+'\\'+'\\'+' \hline'
        print(line)   

def dis_to_latex_2_Tables():
    rem=pd.read_csv('./Results/Benchmarking/dis_base.csv')
    print('Distance')
    for d in np.unique(rem['Dataset']):
        line=d
        data=rem[rem['Dataset']==d]
        line=line +'&$'+str(data[data['Method']=='our']['Proximity'].round(2).values[0]) +' \pm ' +str(data[data['Method']=='our']['Proximity_STD'].round(2).values[0])+'$&'
        if not len(data[data['Method']=='Wachter']['Sparsity'].values)==0:
            line=line +'$' +str(data[data['Method']=='Wachter']['Proximity'].round(2).values[0]) +' \pm ' +str(data[data['Method']=='Wachter']['Proximity_STD'].round(2).values[0])+'$&'
        else:
            line=line+'&'
        if not len(data[data['Method']=='ib']['Sparsity'].values)==0:
            line=line +'$' +str(data[data['Method']=='ib']['Proximity'].round(2).values[0]) +' \pm ' +str(data[data['Method']=='ib']['Proximity_STD'].round(2).values[0])+'$&'
        else:
            line=line+'&'
        if not len(data[data['Method']=='cfg']['Sparsity'].values)==0:
            line=line +'$' +str(data[data['Method']=='cfg']['Proximity'].round(2).values[0]) +' \pm ' +str(data[data['Method']=='cfg']['Proximity_STD'].round(2).values[0])+'$&'
        else:
            line=line+'&'
        if not len(data[data['Method']=='Ates']['Sparsity'].values)==0:
            line=line  +'$'+str(data[data['Method']=='Ates']['Proximity'].round(2).values[0]) +' \pm ' +str(data[data['Method']=='Ates']['Proximity_STD'].round(2).values[0])+'$ \\'+'\\'+' \hline'
        else:
            line=line+'\\'+'\\'+' \hline'
        print(line)  
    print('Sparsity')
    for d in np.unique(rem['Dataset']):
        line=d
        data=rem[rem['Dataset']==d]
        line=line +'&$'+str(data[data['Method']=='our']['Sparsity'].round(2).values[0]) +' \pm ' +str(data[data['Method']=='our']['Sparsity_STD'].round(2).values[0])+'$&'
        if not len(data[data['Method']=='Wachter']['Sparsity'].values)==0:
            line=line +'$' +str(data[data['Method']=='Wachter']['Sparsity'].round(2).values[0]) +' \pm ' +str(data[data['Method']=='Wachter']['Sparsity_STD'].round(2).values[0])+'$&'
        else:
            line=line+'&'
        if not len(data[data['Method']=='ib']['Sparsity'].values)==0:
            line=line +'$' +str(data[data['Method']=='ib']['Sparsity'].round(2).values[0]) +' \pm ' +str(data[data['Method']=='ib']['Sparsity_STD'].round(2).values[0])+'$&'
        else:
            line=line+'&'
        if not len(data[data['Method']=='cfg']['Sparsity'].values)==0:
            line=line +'$' +str(data[data['Method']=='cfg']['Sparsity'].round(2).values[0])+' \pm ' +str(data[data['Method']=='cfg']['Sparsity_STD'].round(2).values[0])+'$&'
        else:
            line=line+'&'
        if not len(data[data['Method']=='Ates']['Sparsity'].values)==0:
            line=line  +'$'+str(data[data['Method']=='Ates']['Sparsity'].round(2).values[0]) +' \pm ' +str(data[data['Method']=='Ates']['Sparsity_STD'].round(2).values[0])+'$ \\'+'\\'+' \hline'
        else:
            line=line+'\\'+'\\'+' \hline'
        print(line)  
    
def dis_to_latex_t():
    rem=pd.read_csv('./Results/Benchmarking/dis_base.csv')
    for d in np.unique(rem['Method']):
        line2=d
        line=d
        data=rem[rem['Method']==d]
        for name in sorted(np.unique(rem['Dataset'])):
            if not len(data[data['Dataset']==name]['Sparsity'].values)==0:
                line=line+'&$'+ str(data[data['Dataset']==name]['Sparsity'].round(4).values[0]) +' \pm ' +str(data[data['Dataset']==name]['Sparsity_STD'].round(4).values[0]) +'$'
                line2=line2+'&$'+ str(data[data['Dataset']==name]['Proximity'].round(4).values[0]) +' \pm ' +str(data[data['Dataset']==name]['Proximity_STD'].round(4).values[0]) +'$'
            else:
                line=line+'&'
                line2=line2+'&'
        line=line+'&'+'\\'+'\\'+' \hline'
        line2=line2+'&'+'\\'+'\\'+' \hline'
        print(line)
        print(line2)



def make_table():
    #TODO still to do 
    dataFrame=pd.DataFrame([])
    for data in os.listdir('./Results/Benchmarking'):
        if not data.endswith('.png'):
            bench=pd.read_csv(f'./Results/Benchmarking/{data}/BenchmarkMetrics.csv')
            line=data 
            for i in range(0,4):
                line =  line +'& $' +str(bench['ynn_timeseries'][i]) + ' \pm ' +str(bench['ynn_timeseries_std'][i])+'$ & $'+str(bench['red'][i]) + ' \pm ' +str(bench['red_std'][i])+ '$&$'+str(bench['sparsity'][i]) + ' \pm ' +str(bench['sparsity_std'][i])+'$&$'+str(bench['dis'][i]) + ' \pm ' +str(bench['dis_std'][i])+'$ & $'+str(bench['validity'][i])
            line=line+'\\'
            print(line)
    return 

def make_table_split():
    #TODO still to do 
    dataFrame=pd.DataFrame([])
    print('yNN')
    for data in os.listdir('./Results/Benchmarking'):
        if not data.endswith('.png'):
            bench=pd.read_csv(f'./Results/Benchmarking/{data}/BenchmarkMetrics.csv')
            line_a=data 
            for i in range(0,4):
                line_a =  line_a +'& $' +str(bench['ynn_timeseries'][i].round(4)) + ' \pm ' +str(bench['ynn_timeseries_std'][i].round(4))+'$'
            line_a=line_a+'\\'+'\\'
            print(line_a)

def build_figure(k=0):
    j=1
    #k=0
    for dataset in ['GunPoint']:#os.listdir('./Benchmarking'):#'CBF','Coffee','ECG5000','ElectricDevices','FordA',
        
        mi=5
        ma=-5
        X_train,train_y,X_test,test_y=UCR_UEA_datasets().load_dataset(dataset)
        train_x=X_train.reshape(-1,X_train.shape[-1],X_train.shape[-2])
        test_x=X_test.reshape(-1,X_train.shape[-1],X_train.shape[-2]) 
        #test = pd.read_csv(os.path.abspath(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TEST.tsv'), sep='\t', header=None)
        #test_y, test_x = test.loc[:, 0].apply(lambda x: x-1).to_numpy(), test.loc[:, 1:].to_numpy()
        original=test_x[k]
        model = ResNetBaseline(in_channels=1, num_pred_classes=len(np.unique(test_y)))
        model.load_state_dict(torch.load(f'./models/{dataset}/ResNet'))
        model.eval()
        l=original.shape[-1]
        enc1=pickle.load(open(f'./models/{dataset}/OneHotEncoder.pkl','rb'))
        test_y=enc1.transform(test_y.reshape(-1,1))
        label=np.argmax( test_y[k])
        #input_ = torch.from_numpy(original).float().reshape(1,-1,l)
        #output = torch.nn.functional.softmax(model(input_)).detach().numpy()
        #label = output.argmax()
        test_y=np.argmax(test_y,axis=1)
        highlight_differences = True

        '''Get Data'''
        pop=pickle.load(open(f'./Results/mutate_both/{dataset}/Counterfactuals_{k}.pkl','rb'))
        if mi>np.min(np.array(pop).reshape(-1),axis=0):
            mi=np.min(np.array(pop).reshape(-1),axis=0)
        if ma<np.max(np.array(pop).reshape(-1),axis=0):
            ma=np.max(np.array(pop).reshape(-1),axis=0)
        try:
            wa = pickle.load(open(f'./Results/Benchmarking/{dataset}/Wachter_cf.pkl','rb'))[0]
            if wa is None:
                wa=np.zeros_like(original)
        except:
            wa=np.zeros_like(original)
        
        if mi>np.min(wa.reshape(-1),axis=0):
            mi=np.min(wa.reshape(-1),axis=0)
        if ma<np.max(wa.reshape(-1),axis=0):
            ma=np.max(wa.reshape(-1),axis=0)
        try:
            cfg= pickle.load(open(f'./Results/Benchmarking/{dataset}/cfg_cf.pkl','rb'))[0]
            if cfg is None:
                cfg=np.zeros_like(original)

        except:
            cfg=np.zeros_like(original)
        if mi>np.min(cfg.reshape(-1),axis=0):
            mi=np.min(cfg.reshape(-1))
        if ma<np.max(cfg.reshape(-1),axis=0):
            ma=np.max(cfg.reshape(-1))
        try:
            ib = pickle.load(open(f'./Results/Benchmarking/{dataset}/ib_cf.pkl','rb'))[0]
            if ib is None:
                ib=np.zeros_like(original)
        except:
            
            ib=np.zeros_like(original)
        if mi> np.min(ib.reshape(-1),axis=0):
            mi=np.min(ib.reshape(-1),axis=0)
        if ma<np.max(ib.reshape(-1),axis=0):
            ma=np.max(ib.reshape(-1),axis=0)
        print('Test_Y ',test_y)
        print('Label',label)
        print(test_y[test_y != label])
        data= test_x[np.where(test_y != label)]
        timeline_max=[]
        y=test_y[test_y != label]
        mi_max=5
        i= 0
        i_max=0
        for timeline in data: 
            mi = np.sum(np.abs(timeline.reshape(-1)- original.reshape(-1) ))/150
            if mi <mi_max: 
                mi_max=mi
                timeline_max= timeline
                i_max=i
            i = i+1
        if mi> np.min(timeline_max.reshape(-1),axis=0):
            mi=np.min(timeline_max.reshape(-1),axis=0)
        if ma<np.max(timeline_max.reshape(-1),axis=0):
            ma=np.max(timeline_max.reshape(-1),axis=0)
        
        

        print('MI',mi)
        print(np.min(original.reshape(-1),axis=0))
        if mi> np.min(original.reshape(-1),axis=0):
            mi=np.min(original.reshape(-1),axis=0)
        if ma<np.max(original.reshape(-1),axis=0):
            ma=np.max(original.reshape(-1),axis=0)
        #print(mi)
        #print(ma)
        mi =mi -0.2
        ma=ma+0.2

        ax011 = plt.subplot(6,1,1,)
        ax012 = ax011.twinx()
        ax011.set_ylim(mi,ma)
        ax012.set_ylim(mi,ma)
        sns.heatmap(np.zeros_like(original.reshape(1,-1)), fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
        p=sns.lineplot(x=range(l), y=original.flatten(), label= str(label), color='white',ax=ax012,legend=False)
        p.set_ylabel("Original")
        
        
        plt.legend(loc='upper right')
       

        ax021 = plt.subplot(6,1,3)
        ax022 = ax021.twinx()
        ax021.set_ylim(mi,ma)
        ax022.set_ylim(mi,ma)
        if highlight_differences:
            sal_01= np.abs(original.reshape(-1)-np.array(pop)[0][0].reshape(-1)).reshape(1,-1)
            sns.heatmap(sal_01, fmt="g", cmap='viridis', cbar=False, ax=ax021, yticklabels=False)
        p=sns.lineplot(x=range(l), y=np.array(pop)[0][0].flatten(), label= str(np.argmax(pop[0].output)), color='white', ax=ax022,legend=False)
        ax022.set(xlabel=None)
        p.set_ylabel("TSEvo")
        #ax021.set(xlabel='TSEvo')
        plt.legend(loc='upper right')
        
        input_ = torch.from_numpy(wa).float().reshape(1,-1,l)
        output = torch.nn.functional.softmax(model(input_)).detach().numpy()
        idx = output.argmax()
        
        ax031 = plt.subplot(6,1,4)
        ax032 = ax031.twinx()
        ax031.set_ylim(mi,ma)
        ax032.set_ylim(mi,ma)
        if highlight_differences:
            sal_02= np.abs(original.reshape(-1)-np.array(wa).reshape(-1)).reshape(1,-1)
            sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax031, yticklabels=False)
        p=sns.lineplot(x=range(l), y=wa.flatten(), label=str(idx), color='white', ax=ax032,legend=False)
        p.set_ylabel("Wachter")
        plt.legend(loc='upper right')
        
        input_ = torch.from_numpy(cfg).float().reshape(1,-1,l)
        output = torch.nn.functional.softmax(model(input_)).detach().numpy()
        idx = output.argmax()
        
        ax041 = plt.subplot(6,1,5)
        ax042 = ax041.twinx()
        ax041.set_ylim(mi,ma)
        ax042.set_ylim(mi,ma)
        if highlight_differences:
            sal_02= np.abs(original.reshape(-1)-np.array(cfg).reshape(-1)).reshape(1,-1)
            sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax041, yticklabels=False)
        p=sns.lineplot(x=range(l), y=cfg.flatten(), label= str(idx), color='white', ax=ax042,legend=False)
        p.set_ylabel("Nun-Cf")
        plt.legend(loc='upper right')
        
        input_ = torch.from_numpy(ib).float().reshape(1,-1,l)
        output = torch.nn.functional.softmax(model(input_)).detach().numpy()
        idx = output.argmax()
        
        ax051 = plt.subplot(6,1,6)
        ax052 = ax051.twinx()
        ax051.set_ylim(mi,ma)
        ax052.set_ylim(mi,ma)
        if highlight_differences:
            sal_02= np.abs(original.reshape(-1)-np.array(ib).reshape(-1)).reshape(1,-1)
            sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax051, yticklabels=False)
        p=sns.lineplot(x=range(l), y=ib.flatten(), label=str(idx), color='white', ax=ax052,legend=False)
        p.set_ylabel("NUN-Grad") 
        plt.legend(loc='upper right')
        
        ax061 = plt.subplot(6,1,2)
        ax062 = ax061.twinx()
        ax061.set_ylim(mi,ma)
        ax062.set_ylim(mi,ma)
        
        if highlight_differences:
            sal_02= np.abs(original.reshape(-1)-np.array(timeline_max).reshape(-1)).reshape(1,-1)
            sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax061, yticklabels=False)
        p=sns.lineplot(x=range(l), y=timeline_max.flatten(), label=str(y[i_max]), color='white', ax=ax062,legend=False)
        plt.legend(loc='upper right')
        p.set_ylabel("Sample")

        print(mi)
        print(ma)

        ax011.set(xlabel=None)
        ax012.set(xlabel=None)
        ax021.set(xlabel=None)
        ax022.set(xlabel=None)
        ax031.set(xlabel=None)
        ax032.set(xlabel=None)
        ax041.set(xlabel=None)
        ax042.set(xlabel=None)
        ax051.set(xlabel=None)
        ax052.set(xlabel=None)
        ax011.set(xticklabels=[])
        ax012.set(xticklabels=[])
        ax021.set(xticklabels=[])
        ax022.set(xticklabels=[])
        ax031.set(xticklabels=[])
        ax032.set(xticklabels=[])
        ax041.set(xticklabels=[])
        ax042.set(xticklabels=[])
        ax061.set(xticklabels=[])
        ax062.set(xticklabels=[])
        plt.tight_layout()
        plt.savefig(f'./Results/Benchmarking/{dataset}_summary_{k}.png',transparent=True)
        plt.close()

def plot_dis(k):
    '''#TODO'''
    print('Start')
    df_plot=pd.DataFrame([])
    d=[]
    dis=[]
    meth=[]
    df_plot2=pd.DataFrame([])
    d2=[]
    dis2=[]
    meth2=[]
    for dataset in ['CBF','Coffee','ECG5000','ElectricDevices','FordA','GunPoint']:
        df= pickle.load(open(f'./Results/Benchmarking/{dataset}/Dis1.pkl','rb'))
        df2= pickle.load(open(f'./Results/Benchmarking/{dataset}/Dis2.pkl','rb'))
        print(dataset)
        print(df)
        
        for a in df.keys():
            print(a)
            dis.extend(df[a])
            meth.extend(np.repeat(a, len(df[a])))
            #print(len(dis))
            #print(len(meth))
            d.extend( np.full(shape=len(df[a]),fill_value=dataset))
        for a in df2.keys():
            print(a)
            dis2.extend(df2[a])
            meth2.extend(np.repeat(a, len(df2[a])))
            #print(len(dis))
            #print(len(meth))
            d2.extend( np.full(shape=len(df2[a]),fill_value=dataset))
    #print(len(d))
    #print(len(dis))
    #print(len(meth))
    #print(d)
    #print('dataframe')
    df_plot2['proximity']=dis2
    df_plot2['dataset']=d2
    df_plot2['method']=meth2
    df_plot['sparsity']=dis
    df_plot['dataset']=d
    df_plot['method']=meth
    print(np.unique(df_plot2['dataset']))
    df_plot2['method']=df_plot2['method'].str.replace('our','TsEvo')
    df_plot2['method']=df_plot2['method'].str.replace('wachter','Wachter')
    df_plot2['method']=df_plot2['method'].str.replace('ib','NUN-CF')
    df_plot2['method']=df_plot2['method'].str.replace('cfg','NUN-Grad')
    df_plot['method']=df_plot['method'].str.replace('our','TsEvo')
    df_plot['method']=df_plot['method'].str.replace('wachter','Wachter')
    df_plot['method']=df_plot['method'].str.replace('ib','NUN-CF')
    df_plot['method']=df_plot['method'].str.replace('cfg','NUN-Grad')
    #print('finished')
    #sns.boxplot(x="dis", y="meth", hue="dataset", data=df_plot, palette="Pastel1")
    #sns.violinplot(x="dis", y="meth", hue="dataset", data=df_plot, palette="Pastel1")
    #sns.violinplot(x=dis, y=meth, hue=dataset, data=df_plot, palette="Pastel1")
    #print('C')
    #plt.savefig(f'./Results/Benchmarking/D_{k}.png',transparent=True)

    f, axes = plt.subplots(1, 2)
    sns.boxplot(x="method", y="sparsity", hue="dataset", data=df_plot, palette="Pastel1",ax=axes[0])
    sns.boxplot(x="method", y="proximity", hue="dataset", data=df_plot2, palette="Pastel1",ax=axes[1])
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper right', ncol=5, bbox_to_anchor=(.75, 0.98))
    plt.tight_layout()
        #plt.savefig('myimage.svg', format='svg', dpi=1200)
    plt.show()
    #plt.close()
    #plt.show()
    #plt.close()

#def calculate_validity():
#    data = []
#    wachter=[]
#    ib=[]
#    cfg=[]
#    ates=[]
#    for file in os.listdir('./Results/Benchmarking'):      
#        if not file.endswith('.csv') and not file.endswith('.png') and file != '.':
#            if 'ates_cf.pkl'in os.listdir(f'./Results/Benchmarking/{file}'):
#                w=pickle.load(open(f'./Results/Benchmarking/{file}/ates_cf.pkl','rb'))
#                da= []
#                for a in w: 
#                    if a is None:
#                        pass
#                    else:
#                        da.append(a)
#                    print(a)
#                ates.append(len(w))
#            if 'ib_cf.pkl'in os.listdir(f'./Results/Benchmarking/{file}'):
#                w=pickle.load(open(f'./Results/Benchmarking/{file}/ib_cf.pkl','rb'))
#                da= []
#                for a in w: 
#                    if a is None:
#                        pass
#                    else:
#                        da.append(a)
#                #print(da)
#                ib.append(len(da))
#            if 'cfg_cf.pkl'in os.listdir(f'./Results/Benchmarking/{file}'):
#                w=pickle.load(open(f'./Results/Benchmarking/{file}/cfg_cf.pkl','rb'))
#                da= []
#                for a in w: 
#                    if a is None:
#                        pass
#                    else:
#                        da.append(a)
#                #print(da)

#                cfg.append(len(da))
#            if 'Wachter_cf.pkl'in os.listdir(f'./Results/Benchmarking/{file}'):
#                w=pickle.load(open(f'./Results/Benchmarking/{file}/Wachter_cf.pkl','rb'))
#                da= []
#                for a in w: 
#                    if a is None:
#                        pass
#                    else:
#                        da.append(a)
#                #print(da)
#                wachter.append(len(da))
#    print(ates)
#    print(ib)
#    print(cfg)
#    print(wachter)
    #print(np.mean(data,axis=1))
    #print(np.std(data,axis=1))

def validity():
    counter=0
    sum_wachter=0
    sum_ates=0
    sum_cfg=0
    sum_out=0
    sum_ng=0

    for a in['GunPoint','Coffee','CBF','ElectricDevices','ECG5000','FordA']:#,'Heartbeat','PenDigits', 'UWaveGestureLibrary','NATOPS']: #os.listdir('./Results/Benchmarking'):
        if os.path.isdir(f'./Results/Benchmarking/{a}'):
            print(a)
            counter += counter
            print('Counter', counter)
            data = pd.read_csv(f'./Results/Benchmarking/{a}/BenchmarkMetrics.csv')
            print(data[data['method']== 'TS_Evo']['validity'].values[0])
            sum_out+=data[data['method']== 'TS_Evo']['validity'].values[0]
            sum_wachter+=data[data['method']== 'Wachter']['validity'].values[0]
            sum_cfg+=data[data['method']== 'NG_DBN']['validity'].values[0]
            sum_ng+=data[data['method']== 'NG_GradCam']['validity'].values[0]
    print(f'Our {sum_out/counter} , Wachter {sum_wachter/counter}, NG_DBM {sum_cfg/counter}, NG_GradCAm {sum_ng/counter}' )
            #TODO Multivariate is still TODO 
            #sum_ates+=data[data['methods']== 'TS_Evo']['validity'].values[0]
            
    pass

if __name__=='__main__':
    validity()
    #build_figure()
    #make_table_split()
    #dis_to_latex_2_Tables()
    #plot_dis(str(1))
    #calculate_total_ynn()
    #calculate_total_dis()
    #dis_to_latex_t()
    #calculate_validity()