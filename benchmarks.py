
"""

This code contains the implementation of benchmarks 

"""

#%% Packages

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.grf import CausalForest

#%% Benchmarks implementation (4 models)

# With respect to data structure, each benchmark has two implementations:
#     1) without weights w_miss (deletion, imputation) and 2) with weights w_miss (reweighting)

###################### Benchmarks train (no reweighting) ######################

# Input: train data (and validation data) of shape [Y, T, X]
# Output: trained model 

# 1) Linear model (Linear)

def Linear(data_train, y_cont):
    
    if y_cont:
        mod_Linear = LinearRegression()
    else:
        mod_Linear = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000)       

    mod_Linear.fit(data_train[:,2:], data_train[:,0], sample_weight=None)
 
    return mod_Linear

# 2) Causal forest (CF)

def CF(data_train, y_cont):

    mod_CF = CausalForest(criterion='mse', n_estimators=100)
    mod_CF.fit(X=data_train[:,2:], T=data_train[:,1], y=data_train[:,0], sample_weight=None)
    
    return mod_CF

# 3) Treatment agnostic representation network (TARNet)

def TARNet(data_train, y_cont, hyperparams):
    
    n = data_train.shape[0]
    p = data_train.shape[1] - 2

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
    class RepresentationTAR(nn.Module):
        def __init__(self, x_size, layer_size_rep, drop):
            super(RepresentationTAR, self).__init__()
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
        
    class HypothesisT0_TAR(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT0_TAR, self).__init__()
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
    
    class HypothesisT1_TAR(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT1_TAR, self).__init__()
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
        
    class TARNet(nn.Module):
        def __init__(self, x_size, layer_size_rep, layer_size_hyp, y_size, drop):
            super(TARNet, self).__init__()
            self.representation = RepresentationTAR(x_size, layer_size_rep, drop)
            self.hypothesisT0 = HypothesisT0_TAR(layer_size_rep, layer_size_hyp, y_size, drop)
            self.hypothesisT1 = HypothesisT1_TAR(layer_size_rep, layer_size_hyp, y_size, drop)
            
        def forward(self, x):
            rep = self.representation(x)
            y0_out = self.hypothesisT0(rep)
            y1_out = self.hypothesisT1(rep)
            return y0_out, y1_out, rep
    
    mod_TARNet = TARNet(x_size=p, layer_size_rep=layer_size_rep, 
                        layer_size_hyp=layer_size_hyp, y_size=y_size, drop=drop).to(device) 
    optimizer = torch.optim.Adam([{'params': mod_TARNet.representation.parameters(), 'weight_decay': 0},
                                  {'params': mod_TARNet.hypothesisT0.parameters()},
                                  {'params': mod_TARNet.hypothesisT1.parameters()}],
                                 lr=lr, weight_decay=lam)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97) 
    MSE_Y = nn.MSELoss(reduction='none') # For continuous Y
    Entropy_Y = nn.CrossEntropyLoss(reduction='none') # For categorical Y
    
    # Train the model
    for iteration in range(n_iter):
        for batch in train_loader:
            if y_size == 1:
                y_data = batch[:,0:1]
            else:
                y_data = batch[:,0].long()
            t_data = batch[:,1]
            x_data = batch[:,2:(p+2)]
            wt_data = batch[:,(p+2):]
            y_data = y_data.to(device)
            t_data = t_data.to(device)
            x_data = x_data.to(device)
            wt_data = wt_data.to(device)
        
            # Forward pass
            y0_out, y1_out, rep = mod_TARNet(x_data)
            
            # Loss 
            idxT0 = torch.where(t_data == 0) 
            idxT1 = torch.where(t_data == 1)
            if y_size == 1:
                loss = torch.mean(wt_data[idxT0]*MSE_Y(y0_out[idxT0], y_data[idxT0])) + \
                    torch.mean(wt_data[idxT1]*MSE_Y(y1_out[idxT1], y_data[idxT1])) # For continuous Y
            else:
                loss = torch.mean(wt_data[idxT0]*Entropy_Y(y0_out[idxT0], y_data[idxT0])) + \
                    torch.mean(wt_data[idxT1]*Entropy_Y(y1_out[idxT1], y_data[idxT1])) # For categorical Y
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (iteration+1) % 50 == 0:
            print (f'Epoch [{iteration+1}/{n_iter}], Loss: {loss.item():.4f}')
        scheduler.step()
        
    return mod_TARNet

# 4) Counterfactual regression Maximum Mean Discrepancy (CFR-MMD)

def CFRMMD(data_train, y_cont, hyperparams):
    
    n = data_train.shape[0]
    p = data_train.shape[1] - 2

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
    class RepresentationMMD(nn.Module):
        def __init__(self, x_size, layer_size_rep, drop):
            super(RepresentationMMD, self).__init__()
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
        
    class HypothesisT0_MMD(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT0_MMD, self).__init__()
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
    
    class HypothesisT1_MMD(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT1_MMD, self).__init__()
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
        
    class CFRMMD(nn.Module):
        def __init__(self, x_size, layer_size_rep, layer_size_hyp, y_size, drop):
            super(CFRMMD, self).__init__()
            self.representation = RepresentationMMD(x_size, layer_size_rep, drop)
            self.hypothesisT0 = HypothesisT0_MMD(layer_size_rep, layer_size_hyp, y_size, drop)
            self.hypothesisT1 = HypothesisT1_MMD(layer_size_rep, layer_size_hyp, y_size, drop)
            
        def forward(self, x):
            rep = self.representation(x)
            y0_out = self.hypothesisT0(rep)
            y1_out = self.hypothesisT1(rep)
            return y0_out, y1_out, rep
    
    mod_CFRMMD = CFRMMD(x_size=p, layer_size_rep=layer_size_rep,
                          layer_size_hyp=layer_size_hyp, y_size=y_size, drop=drop).to(device)           # For continuous y
    optimizer = torch.optim.Adam([{'params': mod_CFRMMD.representation.parameters(), 'weight_decay': 0},
                                  {'params': mod_CFRMMD.hypothesisT0.parameters()},
                                  {'params': mod_CFRMMD.hypothesisT1.parameters()}],
                                 lr=lr, weight_decay=lam)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97)     
    MSE_Y = nn.MSELoss(reduction='none') # For continuous Y
    Entropy_Y = nn.CrossEntropyLoss(reduction='none') # For categorical Y
    
    # Train the model
    for iteration in range(n_iter):
        for batch in train_loader:
            if y_size == 1:
                y_data = batch[:,0:1]
            else:
                y_data = batch[:,0].long()
            t_data = batch[:,1]
            x_data = batch[:,2:(p+2)]
            wt_data = batch[:,(p+2):]
            y_data = y_data.to(device)
            t_data = t_data.to(device)
            x_data = x_data.to(device)
            wt_data = wt_data.to(device)
        
            # Forward pass
            y0_out, y1_out, rep = mod_CFRMMD(x_data)
            
            # Loss 
            idxT0 = torch.where(t_data == 0) 
            idxT1 = torch.where(t_data == 1)
            if y_size == 1:
                lossY = torch.mean(wt_data[idxT0]*MSE_Y(y0_out[idxT0], y_data[idxT0])) + \
                    torch.mean(wt_data[idxT1]*MSE_Y(y1_out[idxT1], y_data[idxT1])) # For continuous Y
            else:
                lossY = torch.mean(wt_data[idxT0]*Entropy_Y(y0_out[idxT0], y_data[idxT0])) + \
                    torch.mean(wt_data[idxT1]*Entropy_Y(y1_out[idxT1], y_data[idxT1])) # For categorical Y
            n_control = idxT0[0].shape[0]
            n_treated = idxT1[0].shape[0]
            if np.min([n_control,n_treated]) > 0:
                rep_T0 = rep[idxT0]
                rep_T1 = rep[idxT1]
                mean_rep_T0 = torch.mean(rep_T0, dim=0)
                mean_rep_T1 = torch.mean(rep_T1, dim=0)
                lossT = 2*torch.sqrt(torch.sum(torch.square(mean_rep_T1 - mean_rep_T0)))
            else:
                lossT = 0
            loss = lossY + alpha*lossT 
        
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()    
        if (iteration+1) % 50 == 0:
            print (f'Epoch [{iteration+1}/{n_iter}], Loss: {lossY.item():.4f}')
        scheduler.step()
        
    return mod_CFRMMD
    
####################### Benchmarks train (reweighting) ########################

# Input: train data (and validation data) of shape [Y, T, X, w_miss] 
# Output: trained model 

# 1) Linear model (Linear)
def Linear_w(data_train, y_cont):
    
    if y_cont:
        mod_Linear = LinearRegression()
    else:
        mod_Linear = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000)       

    mod_Linear.fit(data_train[:,2:-1], data_train[:,0], sample_weight=data_train[:,-1])
 
    return mod_Linear

# 2) Causal forest (CF)

def CF_w(data_train, y_cont):
    
    mod_CF = CausalForest(criterion='mse', n_estimators=100)
    mod_CF.fit(X=data_train[:,2:-1], T=data_train[:,1], y=data_train[:,0], sample_weight=data_train[:,-1])
    
    return mod_CF

# 3) Treatment agnostic representation network (TARNet)

def TARNet_w(data_train, y_cont, hyperparams):
    
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
    class RepresentationTAR(nn.Module):
        def __init__(self, x_size, layer_size_rep, drop):
            super(RepresentationTAR, self).__init__()
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
        
    class HypothesisT0_TAR(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT0_TAR, self).__init__()
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
    
    class HypothesisT1_TAR(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT1_TAR, self).__init__()
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
        
    class TARNet(nn.Module):
        def __init__(self, x_size, layer_size_rep, layer_size_hyp, y_size, drop):
            super(TARNet, self).__init__()
            self.representation = RepresentationTAR(x_size, layer_size_rep, drop)
            self.hypothesisT0 = HypothesisT0_TAR(layer_size_rep, layer_size_hyp, y_size, drop)
            self.hypothesisT1 = HypothesisT1_TAR(layer_size_rep, layer_size_hyp, y_size, drop)
            
        def forward(self, x):
            rep = self.representation(x)
            y0_out = self.hypothesisT0(rep)
            y1_out = self.hypothesisT1(rep)
            return y0_out, y1_out, rep
    
    mod_TARNet = TARNet(x_size=p, layer_size_rep=layer_size_rep, 
                        layer_size_hyp=layer_size_hyp, y_size=y_size, drop=drop).to(device)       # For continuous y
    optimizer = torch.optim.Adam([{'params': mod_TARNet.representation.parameters(), 'weight_decay': 0},
                                  {'params': mod_TARNet.hypothesisT0.parameters()},
                                  {'params': mod_TARNet.hypothesisT1.parameters()}],
                                 lr=lr, weight_decay=lam)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97)  
    MSE_Y = nn.MSELoss(reduction='none') # For continuous Y
    Entropy_Y = nn.CrossEntropyLoss(reduction='none') # For categorical Y
    
    # Train the model
    for iteration in range(n_iter):
        for batch in train_loader:
            if y_size == 1:
                y_data = batch[:,0:1]
            else:
                y_data = batch[:,0].long()
            t_data = batch[:,1]
            x_data = batch[:,2:(p+2)]
            wm_data = batch[:,(p+2):(p+3)]
            wt_data = batch[:,(p+3):]
            y_data = y_data.to(device)
            t_data = t_data.to(device)
            x_data = x_data.to(device)
            wm_data = wm_data.to(device)
            wt_data = wt_data.to(device)
        
            # Forward pass
            y0_out, y1_out, rep = mod_TARNet(x_data)
            
            # Loss 
            idxT0 = torch.where(t_data == 0) 
            idxT1 = torch.where(t_data == 1)
            if y_size == 1:
                loss = torch.mean(wm_data[idxT0]*wt_data[idxT0]*MSE_Y(y0_out[idxT0], y_data[idxT0])) + \
                    torch.mean(wm_data[idxT1]*wt_data[idxT1]*MSE_Y(y1_out[idxT1], y_data[idxT1])) # For continuous Y
            else:
                loss = torch.mean(wm_data[idxT0]*wt_data[idxT0]*Entropy_Y(y0_out[idxT0], y_data[idxT0])) + \
                    torch.mean(wm_data[idxT1]*wt_data[idxT1]*Entropy_Y(y1_out[idxT1], y_data[idxT1])) # For categorical Y
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (iteration+1) % 50 == 0:
            print (f'Epoch [{iteration+1}/{n_iter}], Loss: {loss.item():.4f}')
        scheduler.step()
        
    return mod_TARNet

# 4) Counterfactual regression Maximum Mean Discrepancy (CFR-MMD)

def CFRMMD_w(data_train, y_cont, hyperparams):
    
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
    class RepresentationMMD(nn.Module):
        def __init__(self, x_size, layer_size_rep, drop):
            super(RepresentationMMD, self).__init__()
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
        
    class HypothesisT0_MMD(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT0_MMD, self).__init__()
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
    
    class HypothesisT1_MMD(nn.Module):
        def __init__(self, layer_size_rep, layer_size_hyp, y_size, drop):
            super(HypothesisT1_MMD, self).__init__()
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
        
    class CFRMMD(nn.Module):
        def __init__(self, x_size, layer_size_rep, layer_size_hyp, y_size, drop):
            super(CFRMMD, self).__init__()
            self.representation = RepresentationMMD(x_size, layer_size_rep, drop)
            self.hypothesisT0 = HypothesisT0_MMD(layer_size_rep, layer_size_hyp, y_size, drop)
            self.hypothesisT1 = HypothesisT1_MMD(layer_size_rep, layer_size_hyp, y_size, drop)
            
        def forward(self, x):
            rep = self.representation(x)
            y0_out = self.hypothesisT0(rep)
            y1_out = self.hypothesisT1(rep)
            return y0_out, y1_out, rep
    
    mod_CFRMMD = CFRMMD(x_size=p, layer_size_rep=layer_size_rep,
                          layer_size_hyp=layer_size_hyp, y_size=y_size, drop=drop).to(device)          # For continuous y
    optimizer = torch.optim.Adam([{'params': mod_CFRMMD.representation.parameters(), 'weight_decay': 0},
                                  {'params': mod_CFRMMD.hypothesisT0.parameters()},
                                  {'params': mod_CFRMMD.hypothesisT1.parameters()}],
                                 lr=lr, weight_decay=lam)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97)  
    MSE_Y = nn.MSELoss(reduction='none') # For continuous Y
    Entropy_Y = nn.CrossEntropyLoss(reduction='none') # For categorical Y
    
    # Train the model
    for iteration in range(n_iter):
        for batch in train_loader:
            if y_size == 1:
                y_data = batch[:,0:1]
            else:
                y_data = batch[:,0].long()
            t_data = batch[:,1]
            x_data = batch[:,2:(p+2)]
            wm_data = batch[:,(p+2):(p+3)]
            wt_data = batch[:,(p+3):]
            y_data = y_data.to(device)
            t_data = t_data.to(device)
            x_data = x_data.to(device)
            wm_data = wm_data.to(device)
            wt_data = wt_data.to(device)
        
            # Forward pass
            y0_out, y1_out, rep = mod_CFRMMD(x_data)
            
            # Loss 
            idxT0 = torch.where(t_data == 0) 
            idxT1 = torch.where(t_data == 1)
            if y_size == 1:
                lossY = torch.mean(wm_data[idxT0]*wt_data[idxT0]*MSE_Y(y0_out[idxT0], y_data[idxT0])) + \
                    torch.mean(wm_data[idxT1]*wt_data[idxT1]*MSE_Y(y1_out[idxT1], y_data[idxT1])) # For continuous Y
            else:
                lossY = torch.mean(wm_data[idxT0]*wt_data[idxT0]*Entropy_Y(y0_out[idxT0], y_data[idxT0])) + \
                    torch.mean(wm_data[idxT1]*wt_data[idxT1]*Entropy_Y(y1_out[idxT1], y_data[idxT1])) # For categorical Y
            n_control = idxT0[0].shape[0]
            n_treated = idxT1[0].shape[0]
            if np.min([n_control,n_treated]) > 0:
                rep_T0 = rep[idxT0]
                rep_T1 = rep[idxT1]
                mean_rep_T0 = torch.mean(rep_T0, dim=0)
                mean_rep_T1 = torch.mean(rep_T1, dim=0)
                lossT = 2*torch.sqrt(torch.sum(torch.square(mean_rep_T1 - mean_rep_T0)))
            else:
                lossT = 0
            loss = lossY + alpha*lossT 
        
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (iteration+1) % 50 == 0:
            print (f'Epoch [{iteration+1}/{n_iter}], Loss: {lossY.item():.4f}')
        scheduler.step()
        
    return mod_CFRMMD
    
############################## Benchmarks test ##############################

# Input: test data of shape [X], trained model
# Output: ITE: f(1,x) - f(0,x) prediction on test data

# 1) Linear regression (OLS)

def Linear_pred(data_test, model0, model1, y_cont):
    n = data_test.shape[0]
    if y_cont:
        y0_test = model0.predict(data_test)  
        y1_test = model1.predict(data_test)
        y0_test, y1_test = y0_test.reshape(n,1), y1_test.reshape(n,1)
    else:
        y0_test = model0.predict_proba(data_test)[:,1] 
        y1_test = model1.predict_proba(data_test)[:,1]
    ITE = y1_test - y0_test
    return ITE
        
# 2) Causal forest (CF)

def CF_pred(data_test, model, y_cont):
    n = data_test.shape[0]
    ITE = model.predict(X=data_test)
    if y_cont:
        ITE = ITE.reshape(n,1)
    else:
        ITE = ITE.reshape(-1)
    return ITE
        
# 3) Treatment agnostic representation network (TARNet)

def TARNet_pred(data_test, model, y_cont):
    data_test = torch.from_numpy(data_test.astype(np.float32))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_test = data_test.to(device)
    y0_test, y1_test, _ = model(data_test)
    if not y_cont:
        y0_test = nn.Softmax(dim=1)(y0_test)[:,1]
        y1_test = nn.Softmax(dim=1)(y1_test)[:,1]
    ITE = y1_test - y0_test
    return ITE

# 4) Counterfactual regression Maximum Mean discrepancy (CFR-MMD)

def CFRMMD_pred(data_test, model, y_cont):
    data_test = torch.from_numpy(data_test.astype(np.float32))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_test = data_test.to(device)
    y0_test, y1_test, _ = model(data_test)
    if not y_cont:
        y0_test = nn.Softmax(dim=1)(y0_test)[:,1]
        y1_test = nn.Softmax(dim=1)(y1_test)[:,1]
    ITE = y1_test - y0_test
    return ITE