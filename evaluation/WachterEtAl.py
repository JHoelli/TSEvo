import datetime
from typing import List, Optional
#TODO https://github.com/e-delaney/Instance-Based_CFE_TSC/blob/main/W-CF/Wachter_Counterfactuals_Multiclass_CBF.ipynb
#TODO is this working correctly
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

DECISION_THRESHOLD = 0.5

def wachter_recourse(  torch_model,   x: np.ndarray,  y_target: List[int]=None,  lr: float = 0.01,  lambda_param: float = 0.01,    n_iter: int = 1000,   t_max_min: float = 1.0,
    norm: int = 1,    clamp: bool = True,    loss_type: str = "MSE",) -> np.ndarray:
    """
    Generates counterfactual example according to Wachter et.al for input instance x
    Parameters
    ----------
    torch_model: black-box-model to discover
    x: factual to explain
    
    lr: learning rate for gradient descent
    lambda_param: weight factor for feature_cost
    y_target: List of one-hot-encoded target class
    n_iter: maximum number of iteration
    t_max_min: maximum time of search
    norm: L-norm to calculate cost
    clamp: If true, feature values will be clamped to (0, 1)
    loss_type: String for loss function (MSE or BCE)
    Returns
    -------
    Counterfactual example as np.ndarray
    """
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    # returns counterfactual instance
    torch.manual_seed(0)
    #print(1)
    #if feature_costs is not None:
    #    feature_costs = torch.from_numpy(feature_costs).float().to(device)

    x = torch.from_numpy(x).float().to(device)
    #TODO what is y_target ?
    
    lamb = torch.tensor(lambda_param).float().to(device)
    #print(2)
    x_new = Variable(x.clone(), requires_grad=True)

    x_new_enc =x_new
    optimizer = optim.Adam([x_new], lr, amsgrad=True)
    softmax = nn.Softmax()
    '''Make Target Class'''
    #print(3)
    org = softmax(torch_model(x)).detach().numpy()#[0][y_target]
    #TODO what os y_target?
    temp=np.zeros_like(org)
    

    
    target_class= np.argsort((org))[0][-2:-1][0]
    temp[0][target_class]=1
    y_target = torch.tensor(temp).float().to(device)
    #print(4)
    #print(target_class)
    #print(x_new)
    f_x_new = softmax(torch_model(x_new))[:, target_class]

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)
    #print('y_target',y_target)
    #print('f_x_new', f_x_new)
    loss_fn = torch.nn.MSELoss()
    #print(5)
    #print(f_x_new)
    #print(DECISION_THRESHOLD)
    while f_x_new <= DECISION_THRESHOLD:
        #print(6)
        it = 0
        while f_x_new<= 0.5 and it < n_iter:
            #print(7)
            #print(it)
            #print(it)
            #print('y_target',target_class)
            optimizer.zero_grad()
            x_new_enc = x_new 
            #print(8)
            f_x_new = softmax(torch_model(x_new_enc))[:, target_class]#[:, 1]
            #f_x_loss = torch.log(f_x_new / (1 - f_x_new))

            cost = (
                torch.dist(x_new_enc, x, norm)
            )
            #print(9)
            f_x_loss = torch_model(x_new_enc).squeeze(axis=0)
            loss =  loss_fn(f_x_loss, y_target) + lamb *cost
            #print(10)
            loss.backward()
            optimizer.step()
            it += 1
        lamb -= 0.05

        if datetime.datetime.now() - t0 > t_max:
            print("Timeout - No Counterfactual Explanation Found")
            return None, None
        elif f_x_new>= 0.5:
            print('f_x_new', f_x_new.detach().numpy()[0])
            print("Counterfactual Explanation Found")
            return x_new_enc.cpu().detach().numpy().squeeze(axis=0), np.argmax(f_x_new.detach().numpy())
    return None, None 