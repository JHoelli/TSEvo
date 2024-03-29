'''ResNet Training for the data form the UCR Archieve'''

import pickle
import pandas as pd
import os
from evaluation.Plots import plot_basic_dataset
import numpy as np
from pathlib import Path
import platform
import sklearn
import torch
from models.CNN_TSNet import UCRDataset
from models.ResNet import ResNetBaseline, fit, get_all_preds
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
from tslearn.datasets import UCR_UEA_datasets


os_type= platform.system()

datasets =['CharacterTrajectories','MotorImagery']#['Coffee','CBF','ElectricDevices','ECG5000','GunPoint','FordA','Heartbeat','PenDigits', 'UWaveGestureLibrary','NATOPS']
for dataset in datasets:
    X_train,train_y,X_test,test_y=UCR_UEA_datasets().load_dataset(dataset)
    X_train=np.nan_to_num(X_train, copy=True, nan=0.0)
    X_test=np.nan_to_num(X_test, copy=True, nan=0.0)
    #print(X_train.shape)
    #print(X_train[0])
    train_x=X_train.reshape(-1,X_train.shape[-1],X_train.shape[-2])
    test_x=X_test.reshape(-1,X_train.shape[-1],X_train.shape[-2])
    enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(train_y.reshape(-1,1))
    pickle.dump(enc1,open(f'./models/{dataset}/OneHotEncoder.pkl','wb'))

    train_y=enc1.transform(train_y.reshape(-1,1))
    test_y=enc1.transform(test_y.reshape(-1,1))
    
    n_pred_classes =train_y.shape[1]
    #print(train_y) 
    train_dataset = UCRDataset(train_x.astype(np.float64),train_y.astype(np.int64))
    test_dataset = UCRDataset(test_x.astype(np.float64),test_y.astype(np.int64))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
    model = ResNetBaseline(in_channels=X_train.shape[-1], num_pred_classes=n_pred_classes)
    fit(model,train_loader,test_loader)
    if dataset in os.listdir('./models/'):
        print('Folder exists')
    else: 
        os.mkdir(f'./models/{dataset}')
    torch.save(model.state_dict(), f'./models/{dataset}/ResNet')

    test_preds, ground_truth = get_all_preds(model, test_loader)
    ground_truth=np.argmax(ground_truth,axis=1)

    sns.set(rc={'figure.figsize':(5,4)})
    heatmap=confusion_matrix(ground_truth, test_preds)
    sns.heatmap(heatmap, annot=True)
    plt.savefig(f'./models/{dataset}/ResNet_confusion_matrix.png')
    plt.close()
    acc= accuracy_score(ground_truth, test_preds)
    a = classification_report(ground_truth, test_preds, output_dict=True)
    dataframe = pd.DataFrame.from_dict(a)
    dataframe.to_csv(f'./models/{dataset}/classification_report.csv', index = False)

