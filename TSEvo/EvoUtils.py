from re import X
import numpy as np
import random
import pandas as pd 
from pyts.utils import windowed_view
from deap import creator, base, algorithms, tools
import time
from deap.benchmarks.tools import hypervolume, diversity, convergence
from scipy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def eval(x, mop, return_values_of):
    '''
    Help Function.
    Args:
            x (np.array): instance to evaluate.
            mop (pymop.Problem): instance of Multiobjective Problem.
            return_values_of (np.array): Specify array to return.
            
    Returns:
            [np.array]: fitnessvalues
    '''
    #print('eval')
    return mop.evaluate([x], return_values_of)#, mop.prediction



def evaluate_pop(pop,toolbox):
    for ind in pop:
        out= toolbox.evaluate(ind)
        #print(out)
        if type(out) == tuple:
            ind.fitness.values = out
        else:
            ind.fitness.values = tuple(out[0])
        #print('IND Fitness',ind.fitness.values)
    return pop

def recombine(ind1, ind2):
    '''Crossover'''
    
    window_size1=ind1.window
    window_size2=ind2.window
  
    shape=np.array(ind1).shape[1]
  
    num_channels=len(ind1.channels)
    channel1= ind1.channels
    mutation = ind1.mutation
    
    if window_size1==1:
        ind1, ind2 = tools.cxUniform(np.array(ind1).reshape(num_channels,shape), np.array(ind2).reshape(num_channels,shape), indpb=0.1)
    else: 
        
        if (shape/window_size1).is_integer():
        
            ind1 =windowed_view(np.array(ind1).reshape(num_channels,shape), window_size1, window_step= window_size1)
         
            ind2 = windowed_view(np.array(ind2).reshape(num_channels,shape), window_size1, window_step= window_size1)
       

        else: 
            #print('CX else')
            shape_new = window_size1*(int(shape/window_size1)+1)
            padded= np.zeros((num_channels,shape_new))
            padded2= np.zeros((num_channels,shape_new))
            padded[:, :shape]=np.array(ind1).reshape(num_channels,shape)
            padded2[:, :shape]=np.array(ind2).reshape(num_channels,shape)
            ind1 =windowed_view(np.array(padded).reshape(num_channels,shape_new), window_size1, window_step= window_size1)
            ind2 = windowed_view(np.array(padded2).reshape(num_channels,shape_new), window_size1, window_step= window_size1)

        if num_channels==1:        
            ind1[0], ind2[0] = tools.cxUniform(np.array(ind1[0]).tolist(), np.array(ind2[0]).tolist(), indpb=0.1)
            
        else: 
            items= np.where(channel1==1)         
            if(len(items[0])!=0):
                for item in items: 
                    ind1[item], ind2[item] = tools.cxUniform(np.array(ind1[item]).tolist(), np.array(ind2[item]).tolist(), indpb=0.1)
        
      

    
    shape_new=np.array(ind1).reshape(1,-1).shape[1]
    if shape_new>shape:
   
        diff= shape_new-shape 
        ind1= np.array(ind1).reshape(num_channels,-1)[:,0:shape_new-diff]
        ind2 = np.array(ind2).reshape(num_channels,-1)[:,0:shape_new-diff]
    ind1 = creator.Individual(np.array(ind1).reshape(num_channels,-1).tolist())
    ind2= creator.Individual(np.array(ind2).reshape(num_channels,-1).tolist())
    ind1.window=window_size1
    ind2.window=window_size2
    ind1.mutation=mutation
    ind2.mutation=mutation

    ind1.channels=channel1
    ind2.channels=channel1

    return ind1 , ind2 

def mutate(individual,means,sigmas, indpb, uopb):
    '''Gaussian Mutation'''

    window= individual.window
    channels=individual.channels
    items= np.where(channels==1)

    if len(items[0])!=0:
        channel= random.choice(items[0])
        means=means[channel]
        sigmas=sigmas[channel]
        for i, m, s in zip(range(len(individual[int(channel)])), means, sigmas):

            if random.random() < indpb:               
                individual[channel][i]= random.gauss(m, s)
           
    window, channels=mutate_hyperperameter(individual,window, channels,len(channels))
    ind=creator.Individual(individual)
    ind.window=window
    ind.channel=channels
    ind.mutation='mean'
    return ind,

def create_mstats():
    '''Logging the Stats'''
    stats_y_distance = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_x_distance = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_changed_features = tools.Statistics(lambda ind: ind.fitness.values[2])
    mstats = tools.MultiStatistics(
                                   stats_y_distance=stats_y_distance,
                                   stats_x_distance=stats_x_distance,
                                   stats_changed_features=stats_changed_features)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats


def create_logbook():
    logbook = tools.Logbook()
    logbook.header = "gen", "pop", "evals",  "stats_y_distance", "stats_x_distance", "stats_changed_features"
    #logbook.chapters["fitness"].header = "std", "min", "avg", "max"
    logbook.chapters["stats_y_distance"].header = "std", "min", "avg", "max"
    logbook.chapters["stats_x_distance"].header = "std", "min", "avg", "max"
    logbook.chapters["stats_changed_features"].header = "std", "min", "avg", "max"
    return logbook

def pareto_eq(ind1, ind2):
        """Determines whether two individuals are equal on the Pareto front

        Parameters
        ----------
        ind1: DEAP individual from the GP population
            First individual to compare
        ind2: DEAP individual from the GP population
            Second individual to compare

        Returns
        ----------
        individuals_equal: bool
            Boolean indicating whether the two individuals are equal on
            the Pareto front

        """
        return np.all(ind1.fitness.values == ind2.fitness.values)

def authentic_opposing_information(ind1, reference_set):
    #print('Mutate', np.array(ind1).shape)
    #print('Reference', np.array(reference_set).shape)
    #TODO replace same position or other 
    #TODO as crossover or mutation ? 
    #TODO Back to Original Section ? 
    #print('Before Mut ind1', np.array(ind1).shape)
    window=ind1.window
    #window=10
    num_channels=len(ind1.channels)
    channels=ind1.channels
    #print(channels)

    #print(channels)
    #print(window)
    shape= np.array(ind1).shape[-1]
    #print(shape)
    #print(window)
    #print('shape refernece set',np.array(reference_set).shape)
    #print('Reference Set', reference_set.shape)
    sample_series= random.choice(reference_set)
    #print('Sample Series', sample_series.shape)
    #print('Reference Set', reference_set.shape)
    #print('Reference Set line', sample_series.shape)
    if (shape/window).is_integer():
        #print('Mut if')
        #print('Num_Channels',num_channels)
        #print('Window',window)
        #print('a',np.array(ind1).shape)
        ind1 = windowed_view(np.array(ind1).reshape(num_channels,shape), window, window_step= window)#[0]
        #print('a', np.array(ind1).shape)
        #print('Before Mut SampleSeries', np.array(sample_series).shape)
        sample_series=windowed_view(sample_series.reshape(num_channels,shape), window, window_step= window)#[0]
        #print('After Mut SampleSeries', np.array(sample_series).shape)
    else: 
        #print('Mut else')
        #print('Num_Channels',num_channels)
        #print('Window',window)

        shape_new = window*(int(shape/window)+1)
        #print('Shape_new', shape_new)
        padded= np.zeros((num_channels,shape_new))
        sample_padded= np.zeros((num_channels,shape_new))
        #print(np.array(ind1).reshape(num_channels,-1)
        padded[:, :shape]=np.array(ind1).reshape(num_channels,shape)
        sample_padded[:, :shape]=sample_series.reshape(num_channels,shape)
        ind1 =windowed_view(np.array(padded).reshape(num_channels,shape_new), window, window_step= window)
        sample_series=windowed_view(sample_padded, window, window_step= window)#sample_series.reshape(-1,window)


    #ind1 = windowed_view(np.array(ind1).reshape(1,-1), window, window_step= window)[0]
    #print('Before Mut Windowed ind1', np.array(ind1).shape)
   
    #sample_series=windowed_view(sample_series.reshape(1,-1), window, window_step= window)[0]#sample_series.reshape(-1,window)
    #print('Sample Serie', sample_series.shape)
    #ind1=np.array(ind1).reshape(-1,window)
    #According to Guillmee et al and timeXPlain --> same section 
    #print('sample_series',sample_series.shape)
    #print('ind1',ind1.shape)
    items= np.where(channels==1)
    #print('items', items)
    #flag=False
    if len(items[0])!=0:
        #print('Channel activates')
        channel= random.choice(items[0])
        #print('channels', channel)
        #print('ind1',np.array(ind1).shape)
        #print('Sample Series',np.array(sample_series).shape)
        index= random.randint(0,len(ind1[0])-1)
        #print('ind', ind1[channel,index])
        ind1[channel,index]=sample_series[channel,index]
        #print('sample', sample_series[channel,index])
        #print('ind1', ind1[channel,index])
        #flag = True
        
    #print('length')
    #print(len(ind1))
    #print(len(sample_series))
    #TODO Check idential to old time series --> if yes introduce new immformation , if not introduce old  
    #else: 
        #print('channel not activated')
    
    new_shape = ind1.reshape(num_channels,-1).shape[1]
    if new_shape>shape:
        #print('Mut Reshaping')
        diff= shape_new-shape 
        ind1= np.array(ind1).reshape(num_channels,-1)[:,0:shape_new-diff]

    #if flag:
        #print('ind schluss',np.array(ind1)[channel,index])
        #print('ind schluss shape',np.array(ind1).shape)   
    ind1=ind1.reshape(num_channels,-1)
    #print('ind schluss shape',np.array(ind1).shape) 
    #print(ind1.shape)
    #print('After Mut ind1', ind1.shape)
    ind1=creator.Individual(np.array(ind1).reshape(num_channels,-1).tolist())
    
    window, channels=mutate_hyperperameter(ind1,window, channels,num_channels)
    ind1.window=window      
    ind1.channels=channels
    ind1.mutation='auth'
    return ind1, 

def frequency_band_mapping(ind1, reference_set):
    #TODO Test with ECG or Electric Devices
    #print(np.array(ind1).shape)
    #print(reference_set.shape)
    num_channels=len(ind1.channels)
    channels=ind1.channels
    window=ind1.window
    ind1= np.array(ind1).reshape(1,-1,reference_set.shape[-1])
    shape=ind1.shape
    fourier_timeseries = rfft(ind1) #Fourier transformation of timeseries
    #print(fourier_timeseries.shape)
    fourier_reference_set = rfft(np.array(reference_set)) #Fourier transformation reference set
    #print(fourier_reference_set.shape)
    #print('TS',fourier_timeseries.shape)
    #print('Reference',fourier_reference_set.shape)
    len_ts = ind1.shape[-1] #length of timeseries 
    len_fourier = fourier_timeseries.shape[-1] #lentgh of fourier
    #print('len TS',len_ts)
    #print('len TS',len_fourier)
    
    #Define variables
    length = 1
    num_slices = 1
    
    
    #Set up dataframe for slices with start and end value
    slices_start_end_value = pd.DataFrame(columns= ['Slice_number', 'Start', 'End'])
    #Include the first fourier band which should not be perturbed
    new_row = {'Slice_number': 0, 'Start':0, 'End':1}
    #append row to the dataframe
    slices_start_end_value = slices_start_end_value.append(new_row, ignore_index=True)
    #print(slices_start_end_value)
    #Get start and end values of slices and number slices with quadratic scaling
    start_idx = length
    end_idx = length
    while length < len_fourier:
        start_idx = length   #Start value
        end_idx = start_idx + num_slices**2 #End value
        end_idx = min(end_idx, len_fourier)
        
        new_row = {'Slice_number': num_slices, 'Start':start_idx, 'End':end_idx}
        #append row to the dataframe
        slices_start_end_value = slices_start_end_value.append(new_row, ignore_index=True)
        
        length = length + end_idx - start_idx
        num_slices = num_slices + 1
    #print(slices_start_end_value) 
    
    
    #deact_per_sample = np.random.randint(1, num_slices, num_samples - 1) #random draw of inactive slices per sample, has to be maximal num_slices - 1 because of first fourier
    #perturbation_matrix = np.ones((num_samples, num_slices)) #perturbation matrix with ones
    #features_range = range(1,num_slices) #Because of first fourier term
    #original_data = np.array(ind1.copy())[np.newaxis]
    original_fourier_data = np.array(fourier_timeseries.copy())[np.newaxis]
    
    #Perturb num_samples times the timeseries with random deactive slices
    #for i, num_inactive in enumerate(deact_per_sample, start=1):
        
            # choose random slices indexes to deactivate
    #        inactive_idxs = np.random.choice(features_range, num_inactive, replace=False)
            
            
    #        perturbation_matrix[i, inactive_idxs] = 0


    #num_feature=random.randint(0, shape[-2]-1)
    feature= np.where(channels==1)
    #print('items', items)
    if len(feature[0])!=0:
        #Select Feature to be changed 
        num_feature= random.choice(feature[0])

            
        tmp_fourier_series = np.array(fourier_timeseries.copy()) # timeseries.copy()
        max_row_idx = fourier_reference_set.shape[0] 
        rand_idx = np.random.randint(0, max_row_idx)
        idx=random.randint(0, len (slices_start_end_value)-1)
        start_idx = slices_start_end_value['Start'][idx]
        end_idx = slices_start_end_value['End'][idx]
        
        #print('Temp FOurier TS',tmp_fourier_series.shape)
        #print('Fourier Reference',fourier_reference_set.shape)
        #print(tmp_fourier_series.shape)
        tmp_fourier_series[0,num_feature,start_idx:end_idx] = fourier_reference_set[rand_idx,num_feature, start_idx:end_idx].copy()
        #print(tmp_fourier_series.shape)
        perturbed_fourier_retransform = irfft(tmp_fourier_series, n=shape[2])
        #print(perturbed_fourier_retransform.shape)
        ind1=creator.Individual(np.array(perturbed_fourier_retransform).reshape(shape[1],shape[2]).tolist())
    else:
        ind1=creator.Individual(np.array(ind1).reshape(shape[1],shape[2]).tolist())
    window, channels=mutate_hyperperameter(ind1,window, channels,num_channels)
    ind1.channels=channels
    ind1.window=window
    ind1.mutation='freq'
    return ind1, 

def mutate_mean(ind1,reference_set):
    window=ind1.window
    num_channels=len(ind1.channels)
    channels=ind1.channels
    means = reference_set.mean(axis=0)
    sigmas = reference_set.std(axis=0)
    ind1,=mutate(ind1,means=means,sigmas=sigmas,indpb=0.56, uopb=0.32)
    ind1.mutation='mean'
    window, channels=mutate_hyperperameter(ind1,window, channels,num_channels)
    ind1.channels=channels
    ind1.window=window
    return ind1,



def mutate_both (ind1,reference_set):
    '''Still TODO '''
    if ind1.mutation == 'auth':
        #print('Authentication Called')
        ind1, = authentic_opposing_information(ind1, reference_set)
    elif ind1.mutation == 'freq':
        #print('Frequency Called')
        ind1, = frequency_band_mapping(ind1, reference_set)
    if ind1.mutation == 'mean':
        #print('Mean Called')
        means = reference_set.mean(axis=0) #TODO used to be one
        #print(means.shape)
        sigmas = reference_set.std(axis=0) #TODO used to be one
        ind1,=mutate(ind1,means=means,sigmas=sigmas,indpb=0.56, uopb=0.32)#,observation_x=observation_x
        #ind1, = mutate(individual, observation_x,means,sigmas, indpb, uopb)
    
    return ind1, 

def mutate_hyperperameter(ind1,window,channels, num_channels):
    window= window
    channels=channels
    if random.random()<0.5:
        #print(np.array(ind1).shape[-1])
        window=random.randint(1,np.floor(0.5* np.array(ind1).shape[-1]))
    #if random.random()<0.5 and num_channels !=1:
    #    i=random.randint(0,len(channels)-1)
    #    if channels[i]==1:
            #TODO need to be followed by resseting the individual ! ?
    #        channels[i]=0
    #    else:
    #        channels[i]=1 
    return window, channels

def temporal_fusion_transformer():
    #SKTIME
    pass

if __name__ == '__main__':
    from pathlib import Path
    import os
    import platform
    from models.UCRDataset import UCRDataset
    import seaborn as sns
    import matplotlib.pyplot as plt
    from deap import creator, base, algorithms, tools
    from deap.benchmarks.tools import hypervolume, diversity, convergence
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin,window=0)

    os_type= platform.system()

    dataset ='SmallKitchenAppliances' #'ArrowHead'
    if os_type == 'Linux':
        #print(Path('/media/jacqueline/Data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv'))
        train = pd.read_csv(Path(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TRAIN.tsv'), sep='\t', header=None)
        test = pd.read_csv(os.path.abspath(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TEST.tsv'), sep='\t', header=None)
    train_y, train_x = train.loc[:, 0].apply(lambda x: x-1).to_numpy(), train.loc[:, 1:].to_numpy()
    test_y, test_x = test.loc[:, 0].apply(lambda x: x-1).to_numpy(), test.loc[:, 1:].to_numpy()
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1])) 
    #print(test_y)
    
    test_dataset = UCRDataset(test_x,test_y)

    original_x, y = test_dataset[0]
    #print(y)
    if len(original_x.shape) <3:
            original_x=np.array([original_x])
    reference_set = test_x[np.where(test_y!=y)] 

    ind1 , = frequency_band_mapping(original_x, reference_set)
    #print(np.array(ind1).shape)
    #print(np.array(original_x.shape))

    def plot_CF_Original(pop,original, original_y, highlight_differences= True ,path=None):
        l=original.shape[-1]
        #print(np.array(pop).shape)
        #print(original.shape)

        sns.set(rc={'figure.figsize':(15,6)})

        ax011 = plt.subplot(211)
        ax012 = ax011.twinx()
        ax021 = plt.subplot(212)
        ax022 = ax021.twinx()
        if highlight_differences:
            sal_01= np.abs(original.reshape(-1)-np.array(pop)[0][0].reshape(-1)).reshape(1,-1)
            sns.heatmap(sal_01, fmt="g", cmap='viridis', cbar=False, ax=ax011, yticklabels=False)
        sns.lineplot(x=range(l), y=original.flatten(), label='Observation class ' + str(original_y), color='white', ax=ax012)
        if highlight_differences:
            sns.heatmap(sal_01, fmt="g", cmap='viridis', cbar=False, ax=ax021, yticklabels=False)
        sns.lineplot(x=range(l), y=np.array(pop)[0][0].flatten(), label='Explanation class ' , color='white', ax=ax022) #
        if path == None: 
            plt.show()


    plot_CF_Original(ind1,original_x[0], y, highlight_differences= True ,path=None)


