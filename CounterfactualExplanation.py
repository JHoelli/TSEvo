import numpy as np 
from Evo import EvolutionaryOptimization
class Explanation():
    def __init__(self, model, data, backend='Torch', **kwargs):
        self.model = model 
        self.backend = backend 
        self.transformer = 'Mean_Stand '
        if type(data) == tuple:
            self.x,self.y =data
            #print('Len Reference Set ', len(self.x.shape))
            if not type(self.y[0])==int and not type(self.y[0])==np.int64:
                print('y was one Hot Encoded')
                self.y=np.argmax(self.y,axis=1)
            if len(self.x.shape)==2:
                print('Reshape Reference Set')
                self.x.reshape(-1,1,self.x.shape[-1])
            #print('Reference Set Constructor',self.x.shape)
        else: 
            self.x,self.y = None, None
            print('Dataset is no Tuple ')
        pass

    def explain_instance(self,original_x,original_y, target_y= None,transformer = 'authentic_opposing_information'):
        """
        Entry Point to explain a instance.

        Args:
            original_x (np.array): The instamce to explain.
            original_y (np.array): Classification Probability of instance.
            target_y (int): Class to be targeted
        Returns:
            [deap.Individual, deap.logbook]: Return the Best Individual and Logbook Info.

        """
        #TODO Prove for dict and size
        
        if len(original_x.shape) <3:
            original_x=np.array([original_x])
        if self.backend == 'tf':
            original_x=original_x.reshape(original_x.shape[0], original_x.shape[2],original_x.shape[1])
        neighborhood =[] 
        #TODO make this bullet proov
        if target_y != None:
            print('Reference Set Target')
            if not type(target_y)==int:
                target_y=np.argmax(original_y,axis=1)[0]
                #print('Target', target_y)
            #print('Target', target_y)
            #print(self.y.shape)
            #print(len(np.where(self.y==target_y)[0]))
            reference_set = self.x[np.where(self.y==target_y)]
            #print('x', self.x.shape)
            #print('Ref',reference_set.shape)
        else: 
            print('Reference Set not Target')
            print('y', self.y.shape)
            print('y', self.y)
            print('original', original_y.shape)
            print(np.argmax(original_y,axis=1)[0])
            reference_set = self.x[np.where(self.y!=np.argmax(original_y,axis=1)[0])]
            print('ReferenceSet class', self.y[self.y!=np.argmax(original_y,axis=1)[0]])
        print('Reference Set from Explanation', reference_set.shape)
        if len(reference_set.shape)==2:
            reference_set= reference_set.reshape(-1,1,reference_set.shape[-1])
            
        window= original_x.shape[-1]
        channels = original_x.shape[-2]
        e=EvolutionaryOptimization(self.model, original_x,original_y,target_y,reference_set, neighborhood, window,channels, self.backend,transformer)
        pop,logbook =e.run()
        return pop,logbook

    def explain(self,original_x,original_y, target_y= None):
        explanation = []
        logbook =[]
        
        for i, item in enumerate(original_x):
            exp, log = self.explain_instance(item, original_y[i], target_y)
            explanation = explanation.append(exp)
            logbook= logbook.append(log)
        
        return explanation, logbook
