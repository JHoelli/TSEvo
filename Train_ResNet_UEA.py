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
from data.DataLoader import load_UEA_dataset

os_type= platform.system()
datasets =[ 'PenDigits','NATOPS', 'UWaveGestureLibrary','Heartbeat']
for dataset in datasets:
    print(dataset)
    train_x,train_y, test_x, test_y=load_UEA_dataset(dataset)
    if not dataset in os.listdir('./Results/'):
        os.mkdir(f'./Results/{dataset}')

    if not dataset in os.listdir('./models/'):
        os.mkdir(f'./models/{dataset}')
            
    enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(train_y.reshape(-1,1))
    pickle.dump(enc1,open(f'./models/{dataset}/OneHotEncoder.pkl','wb'))

    train_y=enc1.transform(train_y.reshape(-1,1))
    test_y=enc1.transform(test_y.reshape(-1,1))

    print('This is relevant',train_x.shape)

    n_pred_classes =train_y.shape[1]
    print('n pred classes',n_pred_classes) 
    train_dataset = UCRDataset(train_x.astype(np.float64),train_y.astype(np.int64))
    test_dataset = UCRDataset(test_x.astype(np.float64),test_y.astype(np.int64))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

    model = ResNetBaseline(in_channels= train_x.shape[-2], num_pred_classes=n_pred_classes)
    fit(model,train_loader,test_loader)
    if dataset in os.listdir('./models/'):
        print('Folder exists')
    else: 
        os.mkdir(f'./models/{dataset}')
    torch.save(model.state_dict(), f'./models/{dataset}/ResNet')

    test_preds, ground_truth = get_all_preds(model, test_loader)
    ground_truth=np.argmax(ground_truth,axis=1)
    #test_preds=np.argmax(test_preds,axis=1)
    sns.set(rc={'figure.figsize':(5,4)})
    heatmap=confusion_matrix(ground_truth, test_preds)
    sns.heatmap(heatmap, annot=True)
    plt.savefig(f'./models/{dataset}/ResNet_confusion_matrix.png')
    plt.close()
    acc= accuracy_score(ground_truth, test_preds)
    a = classification_report(ground_truth, test_preds, output_dict=True)
    dataframe = pd.DataFrame.from_dict(a)
    dataframe.to_csv(f'./models/{dataset}/classification_report.csv', index = False)

