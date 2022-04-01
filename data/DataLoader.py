import pandas as pd 
import os 
import numpy as np 
from tslearn.datasets import UCR_UEA_datasets

def load_UCR_dataset(dataset,path='./data'): # path='./Data/UCRArchive_2018'
    ''' 
    Loads and formats UCR Dataset from specified Path

    Args:
            dataset (str): Datasetname from UCR Dataset.
            path(str): Path to UCR Dataset folder.            
    Returns:
            rain_x,train_y,test_x,test_y: train and test data
    
    '''
    train = pd.read_csv(os.path.abspath(f'{path}/{dataset}/{dataset}_TRAIN.tsv'), sep='\t', header=None)
    test = pd.read_csv(os.path.abspath(f'{path}/{dataset}/{dataset}_TEST.tsv'), sep='\t', header=None)
        
    if not dataset in os.listdir('./Results/'):
        os.mkdir(f'./Results/{dataset}')

    if not dataset in os.listdir('./models/'):
        os.mkdir(f'./models/{dataset}')
    train_y, train_x = train.loc[:, 0].apply(lambda x: x-1).to_numpy(), train.loc[:, 1:].to_numpy()
    test_y, test_x = test.loc[:, 0].apply(lambda x: x-1).to_numpy(), test.loc[:, 1:].to_numpy()

    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

    return train_x,train_y,test_x,test_y


def load_UEA_dataset(dataset):
    X_train,y_train, X_test, y_test=UCR_UEA_datasets().load_dataset(dataset)
    train_x=X_train.reshape(-1,X_train.shape[-1],X_train.shape[-2])
    test_x=X_test.reshape(-1,X_train.shape[-1],X_train.shape[-2])
    return train_x,y_train,test_x,y_test
