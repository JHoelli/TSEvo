import pandas as pd
import numpy as np
import torch
import gc
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import random
from sklearn import tree
from sklearn.model_selection import cross_val_score
from pymop import Problem
#import UCRDataset

class UCRDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label

class CNN_TSNet(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0,input_size= 0,output= 2, out_channels=2):
        super(CNN_TSNet,self).__init__()
        # in (batch_size, in_channels, length)
        # out (batch_size, out_channels, (lenght - kernel + 1))
        len_in = input_size
        len_out = ((len_in + (2 * padding) - (kernel_size -1) - 1) / stride) + 1
        #print('len',len_out)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # activation
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(int(len_out* out_channels), 50)#len_out * out_channels
        #TODO Flex
        self.fc2 = nn.Linear(50,output)
        
        self.gradients = None
        
    def forward(self,x):
        x = torch.Tensor(x)
            
        x = self.conv1d(x)
        if self.train and x.requires_grad:
            h = x.register_hook(self.activations_hook)
        x = self.relu(x)
        
        x = x.flatten(start_dim=1)
        #x = x.view(-1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
        
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.conv1d(x)
def train():   
    running_loss = .0
    
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)       
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs.float())      
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    return train_loss
    
    
def valid():
    running_loss = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds, labels)
            running_loss += loss
            
        valid_loss = running_loss/len(test_loader)
        return valid_loss
    
    
def get_all_preds(self, loader):
    self.eval()
    with torch.no_grad():
        all_preds = []
        labels = []
        for batch in loader:
            item, label = batch
            preds = self(item.float())
            all_preds = all_preds + preds.argmax(dim=1).tolist()
            labels = labels + label.tolist()
    return all_preds, labels

def objective(trial, train_dataset, test_dataset,output=5):
    kernel_size = trial.suggest_int('kernel_size', 5, 25, 5)
    stride = 1
    padding = kernel_size - 1
    batch_size = trial.suggest_int('batch_size', 4, 32, 4)
    epochs = trial.suggest_int('n_epochs', 300, 1500, 100)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True) 
    x,y= train_dataset[0]
    input_size=x.shape[-1]
    device = torch.device( "cpu")#"cuda:0" if torch.cuda.is_available() else
    model = CNN_TSNet(kernel_size=kernel_size, stride=stride, padding=padding,input_size= input_size, output=output,out_channels=output).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    #train_dataset = UCRDataset(train_x,train_y)
    #test_dataset = UCRDataset(test_x,test_y)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for idx, (inputs,labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
        
            loss = criterion(preds,labels)
            loss.backward()
            optimizer.step()
        gc.collect()
        
    test_preds, ground_truth = get_all_preds(model, test_loader)
    return accuracy_score(ground_truth, test_preds)

if __name__=='__main__':
    '''Parameters'''
    stride = 1
    kernel_size=10
    padding = kernel_size - 1
    batch_size = 4  # 5
    lr = 0.00026          # 1e-6
    epochs = 300    # 1500

    '''Training Data '''
    train = pd.read_csv('/media/jacqueline/Data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv', sep='\t', header=None)
    test = pd.read_csv('/media/jacqueline/Data/UCRArchive_2018/GunPoint/GunPoint_TEST.tsv', sep='\t', header=None)
    train_y, train_x = train.loc[:, 0].apply(lambda x: x-1).to_numpy(), train.loc[:, 1:].to_numpy()
    test_y, test_x = test.loc[:, 0].apply(lambda x: x-1).to_numpy(), test.loc[:, 1:].to_numpy()
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1])) 
    train_dataset = UCRDataset(train_x,train_y)
    test_dataset = UCRDataset(test_x,test_y)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

    '''Model'''

    device = torch.device("cpu")#"cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_TSNet(kernel_size=kernel_size, stride=stride, padding=padding).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    '''Parameter Objective'''
    study = optuna.create_study(direction='maximize')
    study.optimize(model.objective, n_trials=300)
    train_losses = []
    valid_losses = []
    for epoch in tqdm(range(epochs)):
        tl = model.train()
        train_losses.append(tl.detach().numpy())
        vl = model.valid()
        valid_losses.append(vl.detach().numpy())
        gc.collect()

    '''Run Plots'''
    sns.set(rc={'figure.figsize':(5,4)})
    sns.lineplot(y=[x.item() for x in train_losses], x=range(len(train_losses)), label='train')
    sns.lineplot(y=[x.item() for x in valid_losses], x=range(len(valid_losses)), label='test')
    #TODO Save Function
    test_preds, ground_truth = model.get_all_preds(model, test_loader)

    sns.set(rc={'figure.figsize':(5,4)})
    heatmap=confusion_matrix(ground_truth, test_preds)
    sns.heatmap(heatmap, annot=True)
    #TODO Save Function
    accuracy_score(ground_truth, test_preds)
    #TODO Save Function 
    torch.save(model.state_dict(), 'gunpoint_best_state_test')