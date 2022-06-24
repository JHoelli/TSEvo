import pandas as pd
import matplotlib.pyplot as plt
import os 
import pickle
import torch
import numpy as np 
from evaluation.metrics import redundancy, yNN, d1_distance, d2_distance, yNN_timeseries
from models.ResNet import ResNetBaseline, get_all_preds
from tslearn.datasets import UCR_UEA_datasets
from tslearn.datasets import UCR_UEA_datasets
from deap import creator, base, algorithms, tools
from deap.benchmarks.tools import hypervolume, diversity, convergence
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin,window=0)
files=['authentic_opposing_information','frequency_band_mapping','mutate_mean','mutate_both']

def calculate_full_ynn(new_calculation=False, sum = False):
    ynn_timeseries=[]
    mu=[]
    data=[]
    for mut in files:

        if not new_calculation:

            break
        for dataset in os.listdir(f'./Results/{mut}'):
        #for dataset in os.listdir(f'./Results/{mut}'): 
            
            '''Get Data'''
            if os.path.isdir(f'./Results/{mut}/{dataset}'):
                print(f'./Results/{mut}/{dataset}')
                X_train,train_y, X_test, test_y=UCR_UEA_datasets().load_dataset(dataset)
                train_x = X_train.reshape(X_train.shape[0], X_train.shape[-1], X_train.shape[-2])
                test_x = X_test.reshape(X_test.shape[0], X_train.shape[-1], X_test.shape[-2])
                enc1=pickle.load(open(f'./models/{dataset}/OneHotEncoder.pkl','rb'))
                test_y=enc1.transform(test_y.reshape(-1,1))
                n_classes = test_y.shape[1]


                '''Load Model'''
                model = ResNetBaseline(in_channels=train_x.shape[-2], num_pred_classes=n_classes)
                model.load_state_dict(torch.load(f'./models/{dataset}/ResNet'))
                
                labels=[]
                li=[]
                max_iteration=len(test_y)
                for i, item in enumerate(test_x):
                    print(f'Dataset {dataset} Iteration {i}/{max_iteration}')
                    observation_01=item
                    label_01=np.array([test_y[i]])#test_y[0]
                    print(label_01)
                    pop=pickle.load(open( f'./Results/{mut}/{dataset}/Counterfactuals_{i}.pkl', "rb" ))
                    mlmodel = model 
                    counterfactuals = pop
                    labels.append(np.argmax(counterfactuals[0].output))
                    li.append(np.array(counterfactuals).reshape(observation_01.shape[0],observation_01.shape[1]))
                    if i==19:
                        print('Stop Run')
                        break
                
                ynn_timeseries.append(yNN_timeseries(li, mlmodel,train_x,5,labels=labels))
                data.append(dataset)
                mu.append(mut)
    results = pd.DataFrame([])
    results['Dataset']=data
    results['ynn']=ynn_timeseries
    results['mut']=mu
    results.to_csv('./Results/ALL_ynn.csv')

    

def calculate_single_ynn_redundancy(new_calculation=False, sum = False):
    for mut in files:
        if not new_calculation:

            break
        for dataset in os.listdir(f'./Results/{mut}'): 
            
            '''Get Data'''
            if os.path.isdir(f'./Results/{mut}/{dataset}'):
                print(f'./Results/{mut}/{dataset}')
                if dataset in ['Heartbeat','PenDigits','NATOPS','UWaveGestureLibrary']:
                    X_train,train_y, X_test, test_y=UCR_UEA_datasets().load_dataset(dataset)
                    train_x = X_train.reshape(X_train.shape[0], X_train.shape[-1], X_train.shape[-2])
                    test_x = X_test.reshape(X_test.shape[0], X_train.shape[-1], X_test.shape[-2])
                else:
                    #print(Path('/media/jacqueline/Data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv'))
                    train = pd.read_csv(os.path.abspath(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TRAIN.tsv'), sep='\t', header=None)
                    test = pd.read_csv(os.path.abspath(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TEST.tsv'), sep='\t', header=None)
    
                    train_y, train_x = train.loc[:, 0].apply(lambda x: x-1).to_numpy(), train.loc[:, 1:].to_numpy()
                    test_y, test_x = test.loc[:, 0].apply(lambda x: x-1).to_numpy(), test.loc[:, 1:].to_numpy()
                    train_x = train_x.reshape(train_x.shape[0],1, train_x.shape[-1])
                    test_x = train_x.reshape(train_x.shape[0],1, train_x.shape[-1])
                enc1=pickle.load(open(f'./models/{dataset}/OneHotEncoder.pkl','rb'))
                test_y=enc1.transform(test_y.reshape(-1,1))
                n_classes = test_y.shape[1]


                '''Load Model'''
                print(train_x.shape)
                model = ResNetBaseline(in_channels=train_x.shape[-2], num_pred_classes=n_classes)
                model.load_state_dict(torch.load(f'./models/{dataset}/ResNet'))
                #CF=[]
                #log=[]
                ynn=[]
                ynn_timeseries=[]
                red=[]
                max_iteration=len(test_y)
                for i, item in enumerate(test_x):
                    print(f'Dataset {dataset} Iteration {i}/{max_iteration}')
                    observation_01=item
                    label_01=np.array([test_y[i]])#test_y[0]
                    print(label_01)
                    pop=pickle.load(open( f'./Results/{mut}/{dataset}/Counterfactuals_{i}.pkl', "rb" ))


        
                    mlmodel = model 
                    counterfactuals = pop
          
                    original = observation_01 
                    #ynn.append(yNN(counterfactuals, mlmodel,train_x,5)[0][0])
                    ynn_timeseries.append(yNN_timeseries(counterfactuals, mlmodel,train_x,5)[0][0])
                    #red.append(redundancy(original, counterfactuals, mlmodel)[0])
                    if i==19:
                        print('Stop Run')
                        break
                results = pd.DataFrame([])
                #results['ynn']=ynn
                #results['red']=red
                results['ynn_timeseries']=ynn_timeseries
                results.to_csv(f'./Results/{mut}/{dataset}/Metrics.csv')
    
    print(os.getcwd())

    for path in files:
        if not sum:
            break
        datas=[]
        red_mean=[]
        red_std=[]
        yNN_mean=[]
        yNN_std=[]
        yNN_timeseries_mean=[]
        yNN_timeseries_std=[]
        for dataset in os.listdir('./Results/'+path):
            if os.path.isdir(f'./Results/{path}/{dataset}'):
                data= pd.read_csv(f'./Results/{path}/{dataset}/Metrics.csv')
                datas.append(dataset)
                #red_mean.append(np.mean(data['red']))
                #red_std.append(np.std(data['red']))
                #yNN_mean.append(np.mean(data['ynn']))
                #yNN_std.append(np.std(data['ynn']))
                yNN_timeseries_mean.append(np.mean(data['ynn_timeseries']))
                yNN_timeseries_std.append(np.std(data['ynn_timeseries']))
        frame=pd.DataFrame([])
        print(len(datas))
        print(len(red_mean))
        frame['dataset']=datas
        #frame['redundancy']=red_mean
        #frame['redundancy_std']=red_std
        #frame['yNN']=yNN_mean
        #frame['yNN_std']=yNN_std
        frame['yNN_timeseries']=yNN_timeseries_mean
        frame['yNN_timeseries_std']=yNN_timeseries_std
        frame.to_csv(f'./Results/{path}/Summary.csv')




def latex_table():
    f=open('latex.txt','w')
    st=''
    for file in files:
        data=pd.read_csv('./Results/'+file+'/Summary.csv')
        for d in data['dataset']:
            st=st+d 
            for mut in files: 
                da=pd.read_csv('./Results/'+mut+'/Summary.csv')
                print(mut)
                print(d)
                da=da[da['dataset']==d]
                print(da)
                st =st+'$&$'+str(da['yNN_timeseries'].round(4).values[0])+'\pm'+str(da['yNN_timeseries_std'].round(4).values[0])
                #+str(da['redundancy'].round(4).values[0])+'\pm'+str(da['redundancy_std'].round(4).values[0])+'$&$'
            st=st+'\\'+'\\'+' \hline'
        break

        #st= data['dataset']+'&'+str(data['redundancy'].round(6).values)+'\pm'+str(data['redundancy_std'].round(6).values)+'&'+str(data['yNN'].round(6).values)+'\pm'+str(data['yNN_std'].round(6).values)+'\\'+' \hline'
    print(st)
    f.write(st)

def rerun_l1_l2():
    dis = {}
    for mut in files: 
        dis[f'{mut}']={}#f'{dataset}':{}}
        for dataset in os.listdir(f'./Results/{mut}'):
            if not dataset.endswith('.csv') and not dataset.endswith('.png'):
                print(f'{mut}/{dataset}')
                
                X_train,y_train, X_test, y_test=UCR_UEA_datasets().load_dataset(dataset)
                train_x = X_train.reshape(X_train.shape[0], X_train.shape[-1], X_train.shape[-2])
                test_x = X_test.reshape(X_test.shape[0], X_train.shape[-1], X_test.shape[-2])
                
                if True:
                    enc1=pickle.load(open(f'./models/{dataset}/OneHotEncoder.pkl','rb'))
                    test_y=enc1.transform(y_test.reshape(-1,1))
                    n_classes = test_y.shape[1]
                    #print(n_classes)
                    sal_01=[]
                    sal_02=[]
                    #TODO Not Converging ! 
                    max_iteration=min(len(test_y),19)
                    for i, item in enumerate(test_x):
                        #print(f'Dataset {dataset} Iteration {i}/{max_iteration}')
                        observation_01=item
                        label_01=np.array([test_y[i]])#test_y[0]
                        #print(label_01)
                        pop=pickle.load(open( f'./Results/{mut}/{dataset}/Counterfactuals_{i}.pkl', "rb" ))


                
                        counterfactuals = np.array(pop)[0]
                        #print(np.array(pop[0]).shape)
                        #print(observation_01.shape)
                        if len(observation_01.shape)==1:
                            observation_01=observation_01.reshape(1,-1)
                            counterfactuals=np.array(pop)[0].reshape(1,-1)
     
                        original = observation_01
                        sal_01.append(d1_distance(observation_01,counterfactuals))
                        sal_02.append(d2_distance(observation_01,counterfactuals))
                        if i == max_iteration:
                            break
                
                #print(f'saved mut,{mut}')
                #print(f'saved data,{dataset}')
                dis[f'{mut}'][f'{dataset}']={f'dis_1':sal_01,'dis_2':sal_02 }
                #print(dis)
                
    #print(dis)

    return dis

def evaluate_benchmarks_according_to_o():
    pass
def plot_l1_l2(rerun = True, flip= True):
    '''
    One Graphik in dis 1 and one for dis 2... 
    
    Group datasets according to muation?
    '''
    if rerun:
        dis = rerun_l1_l2()
    #print(dis)
    # Claculation F1 / F2 first to be implemented
    import seaborn as sns
    #sns.set_style('whitegrid')
    #ax = sns.violinplot(x='Survived', y='Age', data=df)
    #ax = sns.stripplot(x="Survived", y="Age", data=df)
    group1= []
    group2=[]
    dataset=[]
    mutation=[]
    print (dis.keys())
    for mut in files:
        print(dis[f'{mut}'].keys())
        for a in sorted(dis[f'{mut}'].keys()):
            
            group1.extend(dis[f'{mut}'][f'{a}']['dis_1'])
            group2.extend(dis[f'{mut}'][f'{a}']['dis_2'])
            for item in dis[f'{mut}'][f'{a}']['dis_1']:
                mutation.append(mut)
                dataset.append(a)
    df=pd.DataFrame([])
    df['mutation']=mutation
    df['mutation']=df['mutation'].str.replace('mutate_mean','Gaussian')
    df['mutation']=df['mutation'].str.replace('mutate_both','Combination')
    df['mutation']=df['mutation'].str.replace('authentic_opposing_information','Opposing')
    df['mutation']=df['mutation'].str.replace('frequency_band_mapping','Frequency')
    df['sparsity']=group1
    df['proximity']=group2
    df['dataset']=dataset
    if flip:
        sns.violinplot(x="l_1", y="mutation", hue="dataset", data=df, palette="Pastel1")
        plt.show()
        plt.close()
        sns.violinplot(x="l_2", y="mutation", hue="dataset", data=df, palette="Pastel1")
        plt.show()
        plt.close()
    else:
        f, axes = plt.subplots(1, 2)
        sns.boxplot(x="mutation", y="sparsity", hue="dataset", data=df, palette="Pastel1",ax=axes[0])
        sns.boxplot(x="mutation", y="proximity", hue="dataset", data=df, palette="Pastel1",ax=axes[1])
        axes[0].get_legend().remove()
        axes[1].get_legend().remove()
        handles, labels = axes[0].get_legend_handles_labels()
        f.legend(handles, labels, loc='upper right', ncol=5, bbox_to_anchor=(.75, 0.98))
        plt.tight_layout()
        plt.savefig('myimage.png', format='png', dpi=1200)
        plt.show()
        plt.close()
def flatex_full_ynn():
    data = pd.read_csv('./Results/ALL_ynn.csv')
    for name in np.unique(data['Dataset']):
        line= name
        d=data[data['Dataset']==name]
        for i,row in d.iterrows():
            #print(i)
            #print(row)
            line=line + '&$'+str(round(row['ynn'],4)) +'$'
        line = line + '\\'+'\\'+' \hline'
        print(line)
import seaborn as sns
import matplotlib.pyplot as plt
def build_figure_mut(k=0):
    print('Start Build Figure')
    j=1
    #k=0
    for dataset in ['CBF','Coffee','ECG5000','ElectricDevices','FordA','GunPoint']:#os.listdir('./Benchmarking'):
        
        mi=5
        ma=-5

        #test = pd.read_csv(os.path.abspath(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TEST.tsv'), sep='\t', header=None)
        #test_y, test_x = test.loc[:, 0].apply(lambda x: x-1).to_numpy(), test.loc[:, 1:].to_numpy()
        
        X_train,train_y,X_test,test_y=UCR_UEA_datasets().load_dataset(dataset)
        train_x=X_train.reshape(-1,X_train.shape[-1],X_train.shape[-2])
        test_x=X_test.reshape(-1,X_train.shape[-1],X_train.shape[-2]) 
        original=test_x[k]
        
        model = ResNetBaseline(in_channels=1, num_pred_classes=len(np.unique(test_y)))
        model.load_state_dict(torch.load(f'./models/{dataset}/ResNet'))
        model.eval()
        l=original.shape[-1]
        enc1=pickle.load(open(f'./models/{dataset}/OneHotEncoder.pkl','rb'))
        #test_y=enc1.transform(test_y.reshape(-1,1))
        #label=np.argmax( test_y[k])
        input_ = torch.from_numpy(original).float().reshape(1,-1,l)
        output = torch.nn.functional.softmax(model(input_)).detach().numpy()
        label = output.argmax()
        
        test_y=np.argmax(model(torch.from_numpy(test_x).float()).detach().numpy(),axis=1)
        highlight_differences = True

        '''Get Data'''
        pop=pickle.load(open(f'./Results/authentic_opposing_information/{dataset}/Counterfactuals_{k}.pkl','rb'))
        if mi>np.min(np.array(pop).reshape(-1),axis=0):
            mi=np.min(np.array(pop).reshape(-1),axis=0)
        if ma<np.max(np.array(pop).reshape(-1),axis=0):
            ma=np.max(np.array(pop).reshape(-1),axis=0)
       
        wa = pickle.load(open(f'./Results/frequency_band_mapping/{dataset}/Counterfactuals_{k}.pkl','rb'))[0] 
        print(type(wa))
            
        
        if mi>np.min(np.array(wa).reshape(-1),axis=0):
            mi=np.min(np.array(wa).reshape(-1),axis=0)
        if ma<np.max(np.array(wa).reshape(-1),axis=0):
            ma=np.max(np.array(wa).reshape(-1),axis=0)
        cfg= pickle.load(open(f'./Results/mutate_mean/{dataset}/Counterfactuals_{k}.pkl','rb'))[0]
            
        if mi>np.min(np.array(cfg).reshape(-1),axis=0):
            mi=np.min(np.array(cfg).reshape(-1))
        if ma<np.max(np.array(cfg).reshape(-1),axis=0):
            ma=np.max(np.array(cfg).reshape(-1))
        try:
            ib = pickle.load(open(f'./Results/mutate_both/{dataset}/Counterfactuals_{k}.pkl','rb'))[0]
            if ib is None:
                ib=np.zeros_like(original)
        except:
            
            ib=np.zeros_like(original)
        if mi> np.min(np.array(ib).reshape(-1),axis=0):
            mi=np.min(np.array(ib).reshape(-1),axis=0)
        if ma<np.max(np.array(ib).reshape(-1),axis=0):
            ma=np.max(np.array(ib).reshape(-1),axis=0)
        print(test_y)
        #print(label)
        data= test_x[np.where(test_y != label)]
        y=test_y[test_y != label]
        timeline_max=[]
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
        
        

        
        if mi> np.min(original.reshape(-1),axis=0):
            mi=np.min(original.reshape(-1),axis=0)
        if ma<np.max(original.reshape(-1),axis=0):
            ma=np.max(original.reshape(-1),axis=0)
        print(mi)
        print(ma)
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
        p.set_ylabel("Authentic")
        #ax021.set(xlabel='TSEvo')
        plt.legend(loc='upper right')
        
        
        ax031 = plt.subplot(6,1,4)
        ax032 = ax031.twinx()
        ax031.set_ylim(mi,ma)
        ax032.set_ylim(mi,ma)
        if highlight_differences:
            sal_02= np.abs(original.reshape(-1)-np.array(wa)[0].reshape(-1)).reshape(1,-1)
            sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax031, yticklabels=False)
        p=sns.lineplot(x=range(l), y=np.array(wa)[0].flatten(), label=str(np.argmax(wa.output)), color='white', ax=ax032,legend=False)
        p.set_ylabel("Frequency")
        plt.legend(loc='upper right')
        
        
        ax041 = plt.subplot(6,1,5)
        ax042 = ax041.twinx()
        ax041.set_ylim(mi,ma)
        ax042.set_ylim(mi,ma)
        print(type(cfg))
        if highlight_differences:
            sal_02= np.abs(original.reshape(-1)-np.array(cfg)[0].reshape(-1)).reshape(1,-1)
            sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax041, yticklabels=False)
        p=sns.lineplot(x=range(l), y=np.array(cfg)[0].flatten(), label= str(np.argmax(cfg.output)), color='white', ax=ax042,legend=False)
        p.set_ylabel("Gaussian")
        plt.legend(loc='upper right')
        
        
        ax051 = plt.subplot(6,1,6)
        ax052 = ax051.twinx()
        ax051.set_ylim(mi,ma)
        ax052.set_ylim(mi,ma)
        if highlight_differences:
            sal_02= np.abs(original.reshape(-1)-np.array(ib)[0].reshape(-1)).reshape(1,-1)
            sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax051, yticklabels=False)
        p=sns.lineplot(x=range(l), y=np.array(ib)[0].flatten(), label=str(np.argmax(ib.output)), color='white', ax=ax052,legend=False)
        p.set_ylabel("Combination") 
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
        plt.savefig(f'./Results/Benchmarking/mut_{dataset}_summary_{k}.png',transparent=True)
        #plt.show()
        plt.close()

if __name__ == 'main':

    plot_l1_l2(True, False)
    calculate_full_ynn(True, True)
    #build_figure_mut()