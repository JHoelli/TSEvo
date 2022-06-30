import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import torch
import time

class Wachter():
    def __init__(self,model, reference_data, threshold=0.5,lamda=0.1) -> None:
        self.model=model
        self.lamda=lamda
        self.X_train, _= reference_data
        self.pred_threshold=threshold
        self.predict=self.get_prediction_torch
        self.mad=stats.median_absolute_deviation(self.X_train)

    def _target_(self,instance):
        #TODO custom predictfunctiom
        #Let's Make CF class the second most probable class according to original prediction
        #print(4)
        target = np.argsort((self.predict(instance.reshape(1,self.shape[0],self.shape[1]))))[0][-2:-1][0] 
        return target
    
    def get_prediction_torch(self, individual):
        individual = np.array(individual.tolist(), dtype=np.float64)
        input_ = torch.from_numpy(individual).float()#.reshape(1,-1,self.window)
  
        with torch.no_grad():
            output = torch.nn.functional.softmax(self.model(input_)).detach().numpy()

        return output


    

    def _dist_mad(self,query, cf):
        #print(7)
        manhat = np.abs(query-cf)
        #mad = stats.median_absolute_deviation(self.X_train)
        return np.sum((manhat/self.mad).flatten())

    def _loss_function_mad(self,x_dash):
        #print(3)
        #target = self._target_(self.to_be_explained_instance)
        #print(5)
        #print('Target', self.target)
        #print('XDash',x_dash)
        #print('query',self.query)
        L = self.lamda*(self.predict(x_dash.reshape(1,self.shape[0],self.shape[1]))[0][self.target] - 1)**2+self._dist_mad(x_dash.reshape(1,self.shape[0],self.shape[1]), self.query)
        #print(8)
        return L

    def explain(self,instance):
        #print(1)
        start_time = time.time()
        self.to_be_explained_instance=instance
        min_edit_cf, undefined_cf_instance = [],[]
        self.shape=(instance.shape[-2],instance.shape[-1])
        x0 = instance.reshape(1,self.shape[0],self.shape[1]) # initial guess for cf
        self.query = instance.reshape(1,self.shape[0],self.shape[1])
        self.target = self._target_(self.query.reshape(1,self.shape[0],self.shape[1]))
        #print(2)
        res = minimize(self._loss_function_mad, x0.reshape(1,-1), method='nelder-mead', options={'maxiter':10, 'xatol': 50, 'adaptive': True})
        cf = res.x.reshape(1,self.shape[0],self.shape[1])
        #print(9)
        
        prob_target = self.predict(cf)[0][self.target]


        i=0
        while prob_target < self.pred_threshold:
            #print('i',i)
            self.lamda = self.lamda*(1+0.5)**i
            x0 = cf.reshape(1,-1,1) # starting point is current cf. In our case we use the native-guide or nun
            res = minimize(self._loss_function_mad, x0.reshape(1,-1), method='nelder-mead', options={'maxiter':10, 'xatol': 50, 'adaptive': True})
            cf = res.x.reshape(1,-1,1)#.reshape(1,self.shape[0],self.shape[1])
            prob_target = self.predict(cf)[0][self.target]
            t=self.predict(cf)
            i += 1
            if time.time()-start_time >60:
                print('Time is up')
                print(str(instance))
                undefined_cf_instance.append(instance)
                return None, None

            if i == 500:
                print('Error condition not met after',i,'iterations')
                print(str(instance))
                undefined_cf_instance.append(instance)
                return None, None
        cf=np.array(cf).reshape(1,self.shape[0],self.shape[1])
        min_edit_cf.append(cf[0])

    
        return np.array(min_edit_cf), np.argmax(t,axis=1)
