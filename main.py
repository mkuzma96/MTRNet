
"""

This code contains the implementation of our method MTRNet

"""

#%% Packages

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


#%% 1) Method implementation - Adversarial learning approach

############################## Method train ##################################

# Input: train and validation data of shape [Y, T, X, R]
# Output: trained model 

def MTRNet(data_train, y_cont, hyperparams):
    
    n = data_train.shape[0]
    p = data_train.shape[1] - 3
    
    if y_cont:
        y_size = 1 # For continuous Y
    else:
        y_size = len(np.unique(data_train[:,0])) # For categorical Y
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    layer_size_rep = hyperparams['layer_size_rep'][0]
    layer_size_hyp = hyperparams['layer_size_hyp'][0]
    drop = hyperparams['drop'][0]
    lr = hyperparams['lr'][0]
    n_iter = hyperparams['n_iter'][0]
    b_size = hyperparams['b_size'][0]
    alpha = hyperparams['alpha'][0]
    beta = hyperparams['beta'][0]
    lam = hyperparams['lam'][0]
    
    # Add weights for treated-control ratio
    w_trt = np.empty(n) 
    mean_trt = np.mean(data_train[:,1])
    for i in range(n):
        w_trt[i] = data_train[i,1]/(2*mean_trt) + (1-data_train[i,1])/(2*(1-mean_trt))
    d_train = np.concatenate((data_train, w_trt.reshape(n,1)), axis=1) 
            
    # Data pre-processing 
    train = torch.from_numpy(d_train.astype(np.float32))
    train_loader = DataLoader(dataset=train, batch_size=math.ceil(b_size), shuffle=True)
    
    # Model 
    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)
    
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg()
    
    def grad_reverse(x):
        return GradReverse.apply(x)
    
    class Representation(nn.Module):
        def __init__(self, x_size, layer_size_rep, drop):
            super(Representation, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(x_size, layer_size_rep),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_rep, layer_size_rep),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_rep, layer_size_rep),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.BatchNorm1d(layer_size_rep)
                )
        def forward(self, x):
            rep = self.model(x)
            return rep  
        
    class HypothesisT0(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT0, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(layer_size_rep, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_hyp, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_hyp, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),   
                nn.Linear(layer_size_hyp, y_size)
                )            
        def forward(self, rep):
            y0_out = self.model(rep)
            return y0_out  
    
    class HypothesisT1(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT1, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(layer_size_rep, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_hyp, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_hyp, layer_size_hyp),  
                nn.ELU(),
                nn.Dropout(drop),   
                nn.Linear(layer_size_hyp, y_size)
                )            
        def forward(self, rep):
            y1_out = self.model(rep)
            return y1_out 
    
    class T_pred(nn.Module):
        def __init__(self, layer_size_rep, drop):
            super(T_pred, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(layer_size_rep, layer_size_rep),
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_rep, 2)
                )    
        def forward(self, rep2):
            t_out = self.model(rep2)
            return t_out
        
    class R_pred(nn.Module):
        def __init__(self, layer_size_rep, drop):
            super(R_pred, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(layer_size_rep, layer_size_rep),
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(layer_size_rep, 2)
                )    
        def forward(self, rep2):
            r_out = self.model(rep2)
            return r_out    
    
    class MTRNet(nn.Module):
        def __init__(self, x_size, layer_size_rep, layer_size_hyp, y_size, drop):
            super(MTRNet, self).__init__()
            self.representation = Representation(x_size, layer_size_rep, drop)
            self.hypothesisT0 = HypothesisT0(layer_size_rep, layer_size_hyp, y_size, drop)
            self.hypothesisT1 = HypothesisT1(layer_size_rep, layer_size_hyp, y_size, drop)
            self.t_pred = T_pred(layer_size_rep, drop)
            self.r_pred = R_pred(layer_size_rep, drop)
            
        def forward(self, x):
            rep = self.representation(x)
            y0_out = self.hypothesisT0(rep)
            y1_out = self.hypothesisT1(rep)
            rep2 = grad_reverse(rep)
            t_out = self.t_pred(rep2)
            r_out = self.r_pred(rep2)
            return y0_out, y1_out, t_out, r_out
    
    mod_MTRNet = MTRNet(x_size=p, layer_size_rep=layer_size_rep, 
                        layer_size_hyp=layer_size_hyp, y_size=y_size, drop=drop).to(device) 
    optimizer = torch.optim.Adam([{'params': mod_MTRNet.representation.parameters(), 'weight_decay': 0},
                                  {'params': mod_MTRNet.hypothesisT0.parameters()},
                                  {'params': mod_MTRNet.hypothesisT1.parameters()},
                                  {'params': mod_MTRNet.t_pred.parameters(), 'weight_decay': 0},
                                  {'params': mod_MTRNet.r_pred.parameters(), 'weight_decay': 0}],
                                 lr=lr, weight_decay=lam)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97)  
    MSE_Y = nn.MSELoss(reduction='none') # For continuous Y
    Entropy_Y = nn.CrossEntropyLoss(reduction='none') # For categorical Y
    Entropy = nn.CrossEntropyLoss()
    
    # Train the model
    for iteration in range(n_iter):
        for batch in train_loader:
            if y_size == 1:
                y_data = batch[:,0:1]
            else:
                y_data = batch[:,0].long()
            t_data = batch[:,1].long()
            x_data = batch[:,2:(p+2)]
            r_data = batch[:,p+2].long()
            wt_data = batch[:,(p+3):]
            y_data = y_data.to(device)
            t_data = t_data.to(device)
            x_data = x_data.to(device)
            r_data = r_data.to(device)
            wt_data = wt_data.to(device)
            
            # Forward pass
            y0_out, y1_out, t_out, r_out = mod_MTRNet(x_data)
            
            # Losses
            idxR1 = torch.where(r_data == 1)
            idxR1T0 = torch.where((r_data == 1) & (t_data == 0)) 
            idxR1T1 = torch.where((r_data == 1) & (t_data == 1))
            if y_size == 1:
                lossY = torch.mean(wt_data[idxR1T0]*MSE_Y(y0_out[idxR1T0], y_data[idxR1T0])) + \
                    torch.mean(wt_data[idxR1T1]*MSE_Y(y1_out[idxR1T1], y_data[idxR1T1])) # For continuous Y
            else:
                lossY = torch.mean(wt_data[idxR1T0]*Entropy_Y(y0_out[idxR1T0], y_data[idxR1T0])) + \
                    torch.mean(wt_data[idxR1T1]*Entropy_Y(y1_out[idxR1T1], y_data[idxR1T1])) # For categorical Y
            lossT = Entropy(t_out[idxR1], t_data[idxR1])
            lossR = Entropy(r_out, r_data)
            if (iteration+1) % 100 == 0:
                alpha = alpha*1.03
                beta = beta*1.03
            loss = lossY + alpha*lossT + beta*lossR
                    
            # Backward and optimize 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (iteration+1) % 50 == 0:
            print (f'Epoch [{iteration+1}/{n_iter}], Loss: {lossY.item():.4f}')
        if (iteration+1) % 10 == 0:
            alpha = alpha*1.03
            beta = beta*1.03
        scheduler.step()
            
    return mod_MTRNet

############################## Method test  ##################################

# Input: test data of shape [X], trained model
# Output: PO prediction on test data

def MTRNet_pred(data_test, model, y_cont):
    data_test = torch.from_numpy(data_test.astype(np.float32))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_test = data_test.to(device)
    y0_test, y1_test, _ , _ = model(data_test)
    if not y_cont:
        y0_test = nn.Softmax(dim=1)(y0_test)[:,1]
        y1_test = nn.Softmax(dim=1)(y1_test)[:,1]
    ITE = y1_test - y0_test
    return ITE
