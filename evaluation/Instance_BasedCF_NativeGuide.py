'''Implementation after Delaney et al . https://github.com/e-delaney/Instance-Based_CFE_TSC'''


from itertools import count
from operator import sub
from tslearn.neighbors import KNeighborsTimeSeries
from torchcam.methods import SmoothGradCAMpp, GradCAM
import numpy as np 
from pathlib import Path
import platform
import os 
import pandas as pd
import numpy as np
import numpy as np
import pickle
from deap import creator, base, algorithms, tools
from deap.benchmarks.tools import hypervolume, diversity, convergence
import torch
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin,window=0)
from tslearn.barycenters import dtw_barycenter_averaging

class NativeGuidCF():
    def __init__(self, model,shape) -> None:
        self.model=model
        self.cam_extractor =SmoothGradCAMpp(self.model,input_shape=(shape) )
        self.ts_length= shape[-1]
        print('TS SHape', shape[-1])


    def native_guide_retrieval(self,query, predicted_label,reference_set, distance, n_neighbors):
        '''
        This gets the nearest unlike neighbors.
        Args:
            query (np.array): The instamce to explain.
            predicted_label (np.array): Label of instance.
            reference_set (np.array): Set of addtional labeled data (could be training or test set)
            distance ():
            num_neighbors (int):number nearest neighbors to return
        Returns:
            [np.array]: Returns K_Nearest_Neighbors of input query with different classification label.
    
        '''
        if type(predicted_label) != int:
            predicted_label=np.argmax(predicted_label)
    
        x_train, y=reference_set
        if len(y.shape)==2:
            y=np.argmax(y,axis=1)
        ts_length=self.ts_length#len(x_train[-1])
        #print(ts_length)
        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric = distance)
        #print(predicted_label)
        #print(y)
        #print(x_train.shape)
        #print(np.where(y != predicted_label))
        #print(x_train[list(np.where(y != predicted_label))].shape)
        #print(x_train.shape)
        #print(query.shape)
        #x_train, inde=np.unique(np.concatenate(x_train,query))
        #print(x_train.shape)
        #print(inde)
       #y=np.delete(y,inde)
        #x_train=x_train[np.where(x_train!=query)].reshape(-1,ts_length,1)
        #y=y[np.where(x_train!=query)]
        knn.fit(x_train[list(np.where(y != predicted_label))].reshape(-1,ts_length,1))
        #print(x_train[list(np.where(y != predicted_label))].shape)
        dist,ind = knn.kneighbors(query.reshape(1,ts_length,1), return_distance=True)
        #print(dist)
        print('Ind_shape',ind.shape)
        print('index',ind)
        x_train.reshape(-1,1,ts_length)
        return dist[0],x_train[np.where(y != predicted_label)][ind[0]]

    def findSubarray(self, a, k): #used to find the maximum contigious subarray of length k in the explanation weight vector
    
        n = len(a)
    
        vec=[] 

        # Iterate to find all the sub-arrays 
        for i in range(n-k+1): 
            temp=[] 

            # Store the sub-array elements in the array 
            for j in range(i,i+k): 
                temp.append(a[j]) 

            # Push the vector in the container 
            vec.append(temp) 

        sum_arr = []
        for v in vec:
            sum_arr.append(np.sum(v))

        return (vec[np.argmax(sum_arr)])

    def counterfactual_generator_swap(self,instance, label,reference_set,train_x,subarray_length=1,max_iter=500):
        _,nun=self.native_guide_retrieval(instance, label,reference_set, 'euclidean', 1)
        if np.count_nonzero(nun.reshape(-1)-instance.reshape(-1))==0:
            print('Starting and nun are Identical !')
        #print('ORIGINAL not identical', np.count_nonzero(nun.reshape(-1)-instance.reshape(-1)))
        #print('ORIGINAL Label',label)
        #if not type(label)==float:
        #    print('Label is one Hot encoded') 
        #    label=np.argmax(label)
        #TODO
        test_x,test_y=reference_set
        test_x=np.array(test_x,dtype=np.float32)
        out=self.model(torch.from_numpy(test_x.reshape(-1,1,instance.shape[-1])))
        y_pred = np.argmax(torch.nn.functional.softmax(out).detach().numpy(), axis=1)
        #Get Activation MAP
        
        #print(nun)
        input_ = torch.from_numpy(np.array(nun)).float().reshape(-1,1,train_x.shape[-1]) #TODO nun or instance
        out = torch.nn.functional.softmax(self.model(input_)).detach().numpy()
        training_weights = self.cam_extractor(out.squeeze(0).argmax().item(), out)[0].detach().numpy()
        input_ = torch.from_numpy(np.array(instance)).float().reshape(-1,1,train_x.shape[-1]) #TODO nun or instance
        out = torch.nn.functional.softmax(self.model(input_)).detach().numpy()
        #print('Weights',training_weights.shape)
        #print('len',subarray_length)
        most_influencial_array=self.findSubarray((training_weights), subarray_length)
    
        starting_point = np.where(training_weights==most_influencial_array[0])[0][0]
    
        X_example = instance.copy().reshape(1,-1)
        #print(instance.shape)
        #print(nun.shape)
        nun=nun.reshape(1,-1)
        #print('Fraction not identical', np.count_nonzero(X_example[0,starting_point:subarray_length+starting_point].reshape(-1)-nun[0,starting_point:subarray_length+starting_point].reshape(-1)))
        X_example[0,starting_point:subarray_length+starting_point] =nun[0,starting_point:subarray_length+starting_point]
        #np.concatenate(instance[:starting_point], nun[starting_point:subarray_length+starting_point], instance[subarray_length+starting_point:])
        #print(np.argmax(label,axis=0))
        input_ = torch.from_numpy(np.array(X_example)).float().reshape(-1,1,train_x.shape[-1]) #TODO nun or instance
        out = torch.nn.functional.softmax(self.model(input_)).detach().numpy()
        prob_target =  out[0][label] #torch.nn.functional.softmax(model(torch.from_numpy(test_x))).detach().numpy()[0][y_pred[instance]]
        #print(out)
        #print('Prob_target',prob_target)
        counter= 0
        while prob_target > 0.5 and counter <max_iter:
        
            subarray_length +=1
            #print(subarray_length)
            #print('Weights',training_weights.shape)
            #print('len',subarray_length)
        
            most_influencial_array=self.findSubarray((training_weights), subarray_length)
            #print(most_influencial_array)
            starting_point = np.where(training_weights==most_influencial_array[0])[0][0]
            #print('starting',starting_point)
            #print(subarray_length)
            X_example = instance.copy().reshape(1,-1)
            #print(instance.shape)
            #print(nun.shape)
            #print('Fraction not identical', np.count_nonzero(X_example[:,starting_point:subarray_length+starting_point].reshape(-1)-nun[:,starting_point:subarray_length+starting_point].reshape(-1)))
            X_example[:,starting_point:subarray_length+starting_point] =nun[:,starting_point:subarray_length+starting_point]
            #print('not identical', np.count_nonzero(X_example.reshape(-1)-instance.reshape(-1)))
            #print('nun',nun.shape)
            #print('X',X_example.shape)
            input_ = torch.from_numpy(np.array(X_example)).float().reshape(-1,1,train_x.shape[-1]) #TODO nun or instance
            out = torch.nn.functional.softmax(self.model(input_)).detach().numpy()
            prob_target = out[0][label]
            #print(out)
            #print('Prob_target',prob_target)
            #model.predict(X_example.reshape(1,1,-1))[0][y_pred[instance]]
            counter=counter+1
            if counter==max_iter or subarray_length==self.ts_length:
                return None
            #print('Prob_Target',prob_target)
        
        return X_example

    def instance_based_cf(self,query,label, reference_set, max_iter=500):
        '''TODO This Calculation still needs to be checked'''
    
        d,nan=self.native_guide_retrieval(query, label,reference_set, 'dtw', 1)
        #print(nan)
        #TODO make use of GRADCAM, Shaping for mutli 
        beta = 0

        insample_cf = nan.reshape(1,1,-1)

        individual = np.array(query.tolist(), dtype=np.float64)
        input_ = torch.from_numpy(individual).float().reshape(1,1,-1)
        output = torch.nn.functional.softmax(self.model(input_)).detach().numpy()
        pred_treshold = 0.5
        target= 0
        print('query',query.shape)
        print('insample',insample_cf.shape)
        query=query.reshape(-1)
        insample_cf=insample_cf.reshape(-1)
        generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([(1-beta), beta]))
        individual = np.array(generated_cf.tolist(), dtype=np.float64)
        input_ = torch.from_numpy(individual).float().reshape(1,1,-1)
        prob_target = torch.nn.functional.softmax(self.model(input_)).detach().numpy()[0][target]
        counter=0

        while prob_target < pred_treshold and counter<max_iter:
            beta +=0.01 
            generated_cf= dtw_barycenter_averaging([query, insample_cf], weights=np.array([(1-beta), beta]))
            individual = np.array(generated_cf.tolist(), dtype=np.float64)
            input_ = torch.from_numpy(individual).float().reshape(1,1,-1)
            prob_target = torch.nn.functional.softmax(self.model(input_)).detach().numpy()[0][target]
            counter=counter+1
        if counter==max_iter:
            return None
    
        #print(generated_cf)
    
        return generated_cf
    
    def explain():
        ''''#TODO This still needs to be implemented'''
        pass


if __name__ == '__main__':


    import sys

    from models.ResNet import ResNetBaseline, get_all_preds
    from CounterfactualExplanation import Explanation
    

    # Load Model Dataset ...
    dataset = 'GunPoint'
    os_type= platform.system()

    if os_type == 'Linux':
        train = pd.read_csv(Path(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TRAIN.tsv'), sep='\t', header=None)
        test = pd.read_csv(os.path.abspath(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TEST.tsv'), sep='\t', header=None)
        if dataset in os.listdir('./Results/'):
            pass
        else:
            os.mkdir(f'./Results/{dataset}')
    
    train_y, train_x = train.loc[:, 0].apply(lambda x: x-1).to_numpy(), train.loc[:, 1:].to_numpy()
    test_y, test_x = test.loc[:, 0].apply(lambda x: x-1).to_numpy(), test.loc[:, 1:].to_numpy()
    enc1=pickle.load(open(f'./models/{dataset}/OneHotEncoder.pkl','rb'))
    test_y=enc1.transform(test_y.reshape(-1,1))
    n_classes = test_y.shape[1]
    print(n_classes)


    '''Load Model'''
    model = ResNetBaseline(in_channels=1, num_pred_classes=n_classes)
    model.load_state_dict(torch.load(f'./models/{dataset}/ResNet'))
    instance_based_cf(test_x[0], test_y[0],model,(test_x,test_y))
    #counterfactual_generator_swap(test_x[0], test_y[0],model,(test_x,test_y),train_x,subarray_length=1)
    #print(native_guide_retrieval(test_x[0], test_y[0],(train_x,train_y), 'dtw', 1))
    #query = train_x[0]
    #beta = 0
    #insample_cf = train_x[1]

    #TODO Classification How to 
    #target = model(query)
    #individual = np.array(query.tolist(), dtype=np.float64)
    #input_ = torch.from_numpy(individual).float().reshape(1,1,-1)
    #output = torch.nn.functional.softmax(model(input_)).detach().numpy()
    #print(output)
    #pred_treshold = 0.5
    #target= 0
    #generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([(1-beta), beta]))
    #individual = np.array(generated_cf.tolist(), dtype=np.float64)
    #input_ = torch.from_numpy(individual).float().reshape(1,1,-1)
    #prob_target = torch.nn.functional.softmax(model(input_)).detach().numpy()[0][target]

    #prob_target = model.predict_proba(generated_cf.reshape(1,-1))[0][target]

    #while prob_target < pred_treshold:
    #    beta +=0.01 
    #    generated_cf= dtw_barycenter_averaging([query, insample_cf], weights=np.array([(1-beta), beta]))
    #    individual = np.array(generated_cf.tolist(), dtype=np.float64)
    #    input_ = torch.from_numpy(individual).float().reshape(1,1,-1)
    #    prob_target = torch.nn.functional.softmax(model(input_)).detach().numpy()[0][target]

