import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchcam.methods import SmoothGradCAMpp, GradCAM
def plot_basic_dataset(train, test, path=None):
    #TODO This needs to flexibilised
    df = train.rename(columns={0:'label'})
    df = df.rename_axis('id')
    df = df.reset_index()
    df = pd.melt(df, id_vars=['label', 'id'], var_name='time', value_name='value')
    df_test = test.rename(columns={0:'label'})
    df_test = df_test.rename_axis('id')
    df_test = df_test.reset_index()
    df_test = pd.melt(df_test, id_vars=['label', 'id'], var_name='time', value_name='value')
    sns.set(rc={'figure.figsize':(20,8)})
    fig, axs = plt.subplots(ncols=len(np.unique(df['label'])))
    mi=df.min().value
    ma=df.max().value
    #print(mi)
    #print(ma)
    axs[0].set_ylim(mi, ma)
    axs[1].set_ylim(mi, ma)
    i=0
    for a in np.unique(df['label']):
        #print(a)
        #print(df.label)
        sns.lineplot(data=df[df.label==a], x='time', y='value', hue='id', ax=axs[i]).set_title(a)
        i=i+1
    fig.align_ylabels(axs=axs[0])
    if path == None: 
        plt.show()
    else: 
        plt.savefig(path+'/All_Data.png',transparent=True)
        plt.close()
    sns.set(rc={'figure.figsize':(10,5)})
    sns.lineplot(data=df, x='time', y='value', hue='label')
    if path == None: 
        plt.show()
    else: 
        plt.savefig(path+'/Mean_Data.png',transparent=True)
        plt.close()

def plot_CF(pop,path= None): 
    sns.set(rc={'figure.figsize':(10,5)})
    sns.lineplot(y=pop[0][0], x=range(np.array(pop).shape[-1]))
    if path == None: 
        plt.show()
    else: 
        plt.savefig(path,transparent=True)
    plt.close()
 
def plot_CF_Original(pop,original, original_y, highlight_differences= True ,path=None):
    '''
    TODO: 
    * Eliminated [0][0] from pop
    * write classes
    * Play Back to Evaluation
    '''
    l=original.shape[-1]
    #print(np.array(pop).shape)
    #print(original.shape)
   

    sns.set(rc={'figure.figsize':(15,6)})

    ax011 = plt.subplot(211)
    ax012 = ax011.twinx()
    ax021 = plt.subplot(212)
    ax022 = ax021.twinx()
    #print(min(pop))
    #print(min(original))
    mi=min(min(pop.reshape(-1)), min(original.reshape(-1)))
    ma=max(max(pop.reshape(-1)), max(original.reshape(-1)))
    #print(mi)
    #print(ma)
    ax012.set_ylim(mi, ma)
    ax022.set_ylim(mi, ma)
    if highlight_differences:
        sal_01= np.abs(original.reshape(-1)-np.array(pop).reshape(-1)).reshape(1,-1)
        sns.heatmap(sal_01, fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
    sns.lineplot(x=range(l), y=original.flatten(), label='Observation class ' + str(original_y), color='white', ax=ax012)
    if highlight_differences:
        sns.heatmap(sal_01, fmt="g", cmap='viridis', cbar=False, ax=ax021, yticklabels=False)
    sns.lineplot(x=range(l), y=np.array(pop).flatten(), label='Explanation class ' , color='white', ax=ax022) #
    if path == None: 
        plt.show()
    else: 
        plt.savefig(path,transparent=True)
    plt.close()

def plot_CF_Original_Closest(pop,original,original_y,timeline_max, timeline_max_y, highlight_differences= True ,path=None):
    l=original.shape[-1]

    ax011 = plt.subplot(311)
    ax012 = ax011.twinx()
    if highlight_differences:
        sal_01= np.abs(original.reshape(-1)-np.array(pop)[0][0].reshape(-1)).reshape(1,-1)
        sns.heatmap(sal_01, fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
    sns.lineplot(x=range(l), y=original.flatten(), label='Observation class ' + str(original_y), color='white', ax=ax012)

    ax021 = plt.subplot(312)
    ax022 = ax021.twinx()
    if highlight_differences:
        sal_02= np.abs(original.reshape(-1)-timeline_max.reshape(-1)).reshape(1,-1)
        sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax021, yticklabels=False)
    sns.lineplot(x=range(l), y=timeline_max.flatten(), label='Observation class ' + str(timeline_max_y), color='white', ax=ax022)

    ax031 = plt.subplot(313)
    ax032 = ax031.twinx()
    if highlight_differences:
        sal_01= np.abs(original.reshape(-1)-np.array(pop)[0][0].reshape(-1)).reshape(1,-1)
        sns.heatmap(sal_01, fmt="g", cmap='viridis', cbar=False, ax=ax031, yticklabels=False)
    sns.lineplot(x=range(l), y=np.array(pop)[0][0].flatten(), label='counterfactual class ' + str(np.argmax(pop[0].output)), color='white', ax=ax032)
    if path == None: 
        plt.show()
    else: 
        plt.savefig(path,transparent=True)
    plt.close()

def gadient_vis(pop,original,original_y,timeline_max, timeline_max_y, model ,path=None):
    l=original.shape[-1]
    #print(original.shape)
    #print(np.array(pop).shape)
    cam_extractor =SmoothGradCAMpp(model,input_shape=(np.array(pop).shape[-2],np.array(pop).shape[-1]))


    input_ = torch.from_numpy(timeline_max).float().reshape(1,1,-1)
    out = torch.nn.functional.softmax(model(input_)).detach().numpy()

    activation_map3 = cam_extractor(out.squeeze(0).argmax().item(), out)

    input_ = torch.from_numpy(np.array(pop[0][0])).float().reshape(1,1,-1)
    out = torch.nn.functional.softmax(model(input_)).detach().numpy()
    activation_map2 = cam_extractor(out.squeeze(0).argmax().item(), out)
    input_ = torch.from_numpy(original).float().reshape(1,1,-1)
    out = torch.nn.functional.softmax(model(input_)).detach().numpy()
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)


    sns.set(rc={'figure.figsize':(15,6)})

    ax011 = plt.subplot(311)
    ax012 = ax011.twinx()
    ax021 = plt.subplot(312)
    ax022 = ax021.twinx()
    ax031 = plt.subplot(313)
    ax032 = ax031.twinx()

    sal_01= activation_map[0].detach().numpy().reshape(1,-1)
    sns.heatmap(sal_01, fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
    sns.lineplot(x=range(l), y=original.flatten(), label=f'Observation class {original_y}', color='white', ax=ax012)

    sal_03= activation_map3[0].detach().numpy().reshape(1,-1)
    sns.heatmap(sal_03, fmt="g", cmap='viridis', cbar=False, ax=ax031, yticklabels=False)
    sns.lineplot(x=range(l), y= timeline_max.flatten(), label=f'Explanation class ' , color='white', ax=ax032)

    sal_02= activation_map2[0].detach().numpy().reshape(1,-1)
    sns.heatmap(sal_02, fmt="g", cmap='viridis', cbar=False, ax=ax021, yticklabels=False)
    sns.lineplot(x=range(l), y=np.array(pop)[0][0].flatten(), label='counterfactual class ' + str(np.argmax(pop[0].output)), color='white', ax=ax022)
    if path == None: 
        plt.show()
    else: 
        plt.savefig(path,transparent=True)
    plt.close()

def plot_multi():
    '''TODO Impelement'''
    pass