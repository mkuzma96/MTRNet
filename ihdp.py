
"""

The code contains experiments with IHDP dataset

"""

# Packages

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
from main import *
from benchmarks import *

# Data

# Description:  
#     Outcome: Y - continuous 
#     Treatment: T - binary
#     Observed covariates: X_vec = [x1, ..., xp] - vector of covariates (p=25)

# This semi-sythetic data comprises of 100 simulated datasets with 747 observations in each: 
# 25 covariates, treatment, outcome
# -> We know the true ITE used to simulate the dataset and hence use PEHE as evaluation metric.

# Load data
data = np.load('data/ihdp.npy')
n_sim = data.shape[0]
n = data.shape[1]
p = data.shape[2] - 4

np.random.seed(0)
torch.manual_seed(0)

# Hyperparameters 

# hyperparams_list = {
#     'layer_size_representation': [50, 100, 200],
#     'layer_size_hypothesis': [50, 100, 200],
#     'learn_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001],
#     'dropout_rate':  [0.1, 0.2, 0.3],
#     'num_iterations': [100, 200, 300], 
#     'batch_size': [50, 70, 100],
#     'alphas': [10**(k/2) for k in np.linspace(-4,2,9)],
#     'betas': [10**(k/2) for k in np.linspace(-4,2,9)],
#     'lambdas': [0.0005, 0.0001, 0.00005]
#     }

hyperpar_opt_adv = {
        'layer_size_rep': [200],
        'layer_size_hyp': [100],
        'lr': [0.01],
        'drop': [0.2],
        'n_iter': [300],
        'b_size': [50],
        'alpha': [0.13],
        'beta': [0.06],
        'lam': [0.00005]
        }

hyperpar_opt_tar = {
        'layer_size_rep': [200],
        'layer_size_hyp': [100],
        'lr': [0.005],
        'drop': [0.1],
        'n_iter': [300],
        'b_size': [50],
        'lam': [0.00005]
        }

hyperpar_opt_mmd = {
        'layer_size_rep': [200],
        'layer_size_hyp': [100],
        'lr': [0.0001],
        'drop': [0.1],
        'n_iter': [300],
        'b_size': [50],
        'alpha': [10],
        'lam': [0.00005]
        }

# Experiments

# Save results for n_sim runs
ihdp_results = {
        'MTHD_adv':[[],[],[]],
        'Linear_del':[[],[],[]],
        'Linear_imp':[[],[],[]],
        'Linear_rew':[[],[],[]],
        'CF_del':[[],[],[]],
        'CF_imp':[[],[],[]],
        'CF_rew':[[],[],[]],
        'TARNet_del':[[],[],[]],
        'TARNet_imp':[[],[],[]],
        'TARNet_rew':[[],[],[]],
        'CFRMMD_del':[[],[],[]],
        'CFRMMD_imp':[[],[],[]],
        'CFRMMD_rew':[[],[],[]]
        }

for run in range(n_sim):
    
    print(run)
    
    # Train-Test split -> shape [ITE, Y, T, X, R]
    d_train, d_test = train_test_split(data[run,:,:], test_size=0.1, shuffle=False)

    # Train data -> shape [Y, T, X, R]
    d_train = d_train[:,1:]
    
    # Deletion method data (delete cases with R=0) -> shape [Y, T, X]
    d_train_del = d_train[(d_train[:,-1] == 1),:-1]
    
    # Imputation method data (impute according to p(T|X)) -> shape [Y, T, X]
    modelT = RandomForestClassifier(n_estimators=500)
    modelT.fit(d_train_del[:,2:], d_train_del[:,1])
    T_pred = modelT.predict(d_train[:, 2:-1])
    d_train_imp = np.empty(shape=(d_train.shape[0],p+2))
    d_train_imp[:,:] = d_train[:,:-1]
    for i in range(d_train_imp.shape[0]):
        if d_train[i,-1] == 0:
            d_train_imp[i,1] = T_pred[i]
                
    # Reweighting method data (reweight using p(R|X)) -> shape [Y, T, X, w_miss]
    modelR = RandomForestClassifier(n_estimators=500)
    modelR.fit(d_train[:,2:-1], d_train[:,-1])
    w_miss = np.empty(shape=d_train_del.shape[0])
    w_miss = 1/modelR.predict_proba(d_train_del[:, 2:])[:,1]
    d_train_rew = np.concatenate((d_train_del, w_miss.reshape(d_train_del.shape[0],1)), axis=1)
    
    # Splitting test data into observed T and missing T
    d_test_ot = d_test[(d_test[:,-1] == 1),:]
    d_test_mt = d_test[(d_test[:,-1] == 0),:]
        
    # Performance evaluation metric: PEHE
    def PEHE(ITE_true, ITE_est):
        if type(ITE_est) == torch.Tensor:
            ITE_est = ITE_est.cpu().detach().numpy()
        PEHE = np.square(np.subtract(ITE_true, ITE_est)).mean()
        return np.sqrt(PEHE)
    
    ITE_true = d_test[:,0:1]
    ITE_true_ot = d_test_ot[:,0:1]
    ITE_true_mt = d_test_mt[:,0:1]
    
    # Implementation of method 
    method_adv = MTRNet(data_train=d_train, y_cont=True, hyperparams=hyperpar_opt_adv)
    
    # Performance evaluation for method (d_test_ot, d_test_mt, d_test)
    ITE_est_adv = MTRNet_pred(d_test[:,3:-1], method_adv.eval(), y_cont=True)
    ITE_est_adv_ot = MTRNet_pred(d_test_ot[:,3:-1], method_adv.eval(), y_cont=True)
    ITE_est_adv_mt = MTRNet_pred(d_test_mt[:,3:-1], method_adv.eval(), y_cont=True)
    
    PEHE_hat_adv = PEHE(ITE_true, ITE_est_adv)
    PEHE_hat_adv_ot = PEHE(ITE_true_ot, ITE_est_adv_ot)
    PEHE_hat_adv_mt = PEHE(ITE_true_mt, ITE_est_adv_mt)
    
    ihdp_results['MTHD_adv'][0].append(PEHE_hat_adv)
    ihdp_results['MTHD_adv'][1].append(PEHE_hat_adv_ot)
    ihdp_results['MTHD_adv'][2].append(PEHE_hat_adv_mt)
    
    # Implementation of benchmarks - 3 * 4 models (d_train_del, d_train_imp, d_train_del + w)
    Linear_del0 = Linear(d_train_del[d_train_del[:,1] == 0,:], y_cont=True)
    Linear_imp0 = Linear(d_train_imp[d_train_imp[:,1] == 0,:], y_cont=True)
    Linear_rew0 = Linear_w(d_train_rew[d_train_rew[:,1] == 0,:], y_cont=True)
    
    Linear_del1 = Linear(d_train_del[d_train_del[:,1] == 1,:], y_cont=True)
    Linear_imp1 = Linear(d_train_imp[d_train_imp[:,1] == 1,:], y_cont=True)
    Linear_rew1 = Linear_w(d_train_rew[d_train_rew[:,1] == 1,:], y_cont=True)
    
    CF_del = CF(d_train_del, y_cont=True)
    CF_imp = CF(d_train_imp, y_cont=True)
    CF_rew = CF_w(d_train_rew, y_cont=True)
        
    TARNet_del = TARNet(data_train=d_train_del, y_cont=True, hyperparams=hyperpar_opt_tar)   
    TARNet_imp = TARNet(data_train=d_train_imp, y_cont=True, hyperparams=hyperpar_opt_tar)
    TARNet_rew = TARNet_w(data_train=d_train_rew, y_cont=True, hyperparams=hyperpar_opt_tar)
    CFRMMD_del = CFRMMD(data_train=d_train_del, y_cont=True, hyperparams=hyperpar_opt_mmd)
    CFRMMD_imp = CFRMMD(data_train=d_train_imp, y_cont=True, hyperparams=hyperpar_opt_mmd)
    CFRMMD_rew = CFRMMD_w(data_train=d_train_rew, y_cont=True, hyperparams=hyperpar_opt_mmd)
    
    # Performance evaluation for benchmarks (d_test_ot, d_test_mt, d_test)
    ITE_est_Linear_del = Linear_pred(d_test[:,3:-1], Linear_del0, Linear_del1, y_cont=True)
    ITE_est_Linear_del_ot = Linear_pred(d_test_ot[:,3:-1], Linear_del0, Linear_del1, y_cont=True)
    ITE_est_Linear_del_mt = Linear_pred(d_test_mt[:,3:-1], Linear_del0, Linear_del1, y_cont=True)
    
    PEHE_hat_Linear_del = PEHE(ITE_true, ITE_est_Linear_del)
    PEHE_hat_Linear_del_ot = PEHE(ITE_true_ot, ITE_est_Linear_del_ot)
    PEHE_hat_Linear_del_mt = PEHE(ITE_true_mt, ITE_est_Linear_del_mt)
    
    ihdp_results['Linear_del'][0].append(PEHE_hat_Linear_del)
    ihdp_results['Linear_del'][1].append(PEHE_hat_Linear_del_ot)
    ihdp_results['Linear_del'][2].append(PEHE_hat_Linear_del_mt)
    
    ITE_est_Linear_imp = Linear_pred(d_test[:,3:-1], Linear_imp0, Linear_imp1, y_cont=True)
    ITE_est_Linear_imp_ot = Linear_pred(d_test_ot[:,3:-1], Linear_imp0, Linear_imp1, y_cont=True)
    ITE_est_Linear_imp_mt = Linear_pred(d_test_mt[:,3:-1], Linear_imp0, Linear_imp1, y_cont=True)
    
    PEHE_hat_Linear_imp = PEHE(ITE_true, ITE_est_Linear_imp)
    PEHE_hat_Linear_imp_ot = PEHE(ITE_true_ot, ITE_est_Linear_imp_ot)
    PEHE_hat_Linear_imp_mt = PEHE(ITE_true_mt, ITE_est_Linear_imp_mt)
    
    ihdp_results['Linear_imp'][0].append(PEHE_hat_Linear_imp)
    ihdp_results['Linear_imp'][1].append(PEHE_hat_Linear_imp_ot)
    ihdp_results['Linear_imp'][2].append(PEHE_hat_Linear_imp_mt)
    
    ITE_est_Linear_rew = Linear_pred(d_test[:,3:-1], Linear_rew0, Linear_rew1, y_cont=True)
    ITE_est_Linear_rew_ot = Linear_pred(d_test_ot[:,3:-1], Linear_rew0, Linear_rew1, y_cont=True)
    ITE_est_Linear_rew_mt = Linear_pred(d_test_mt[:,3:-1], Linear_rew0, Linear_rew1, y_cont=True)
    
    PEHE_hat_Linear_rew = PEHE(ITE_true, ITE_est_Linear_rew)
    PEHE_hat_Linear_rew_ot = PEHE(ITE_true_ot, ITE_est_Linear_rew_ot)
    PEHE_hat_Linear_rew_mt = PEHE(ITE_true_mt, ITE_est_Linear_rew_mt)
    
    ihdp_results['Linear_rew'][0].append(PEHE_hat_Linear_rew)
    ihdp_results['Linear_rew'][1].append(PEHE_hat_Linear_rew_ot)
    ihdp_results['Linear_rew'][2].append(PEHE_hat_Linear_rew_mt)
    
    ITE_est_CF_del = CF_pred(d_test[:,3:-1], CF_del, y_cont=True)
    ITE_est_CF_del_ot = CF_pred(d_test_ot[:,3:-1], CF_del, y_cont=True)
    ITE_est_CF_del_mt = CF_pred(d_test_mt[:,3:-1], CF_del, y_cont=True)
    
    PEHE_hat_CF_del = PEHE(ITE_true, ITE_est_CF_del)
    PEHE_hat_CF_del_ot = PEHE(ITE_true_ot, ITE_est_CF_del_ot)
    PEHE_hat_CF_del_mt = PEHE(ITE_true_mt, ITE_est_CF_del_mt)
    
    ihdp_results['CF_del'][0].append(PEHE_hat_CF_del)
    ihdp_results['CF_del'][1].append(PEHE_hat_CF_del_ot)
    ihdp_results['CF_del'][2].append(PEHE_hat_CF_del_mt)
    
    ITE_est_CF_imp = CF_pred(d_test[:,3:-1], CF_imp, y_cont=True)
    ITE_est_CF_imp_ot = CF_pred(d_test_ot[:,3:-1], CF_imp, y_cont=True)
    ITE_est_CF_imp_mt = CF_pred(d_test_mt[:,3:-1], CF_imp, y_cont=True)
    
    PEHE_hat_CF_imp = PEHE(ITE_true, ITE_est_CF_imp)
    PEHE_hat_CF_imp_ot = PEHE(ITE_true_ot, ITE_est_CF_imp_ot)
    PEHE_hat_CF_imp_mt = PEHE(ITE_true_mt, ITE_est_CF_imp_mt)
    
    ihdp_results['CF_imp'][0].append(PEHE_hat_CF_imp)
    ihdp_results['CF_imp'][1].append(PEHE_hat_CF_imp_ot)
    ihdp_results['CF_imp'][2].append(PEHE_hat_CF_imp_mt)
    
    ITE_est_CF_rew = CF_pred(d_test[:,3:-1], CF_rew, y_cont=True)
    ITE_est_CF_rew_ot = CF_pred(d_test_ot[:,3:-1], CF_rew, y_cont=True)
    ITE_est_CF_rew_mt = CF_pred(d_test_mt[:,3:-1], CF_rew, y_cont=True)
    
    PEHE_hat_CF_rew = PEHE(ITE_true, ITE_est_CF_rew)
    PEHE_hat_CF_rew_ot = PEHE(ITE_true_ot, ITE_est_CF_rew_ot)
    PEHE_hat_CF_rew_mt = PEHE(ITE_true_mt, ITE_est_CF_rew_mt)
    
    ihdp_results['CF_rew'][0].append(PEHE_hat_CF_rew)
    ihdp_results['CF_rew'][1].append(PEHE_hat_CF_rew_ot)
    ihdp_results['CF_rew'][2].append(PEHE_hat_CF_rew_mt)
    
    ITE_est_TARNet_del = TARNet_pred(d_test[:,3:-1], TARNet_del.eval(), y_cont=True)
    ITE_est_TARNet_del_ot = TARNet_pred(d_test_ot[:,3:-1], TARNet_del.eval(), y_cont=True)
    ITE_est_TARNet_del_mt = TARNet_pred(d_test_mt[:,3:-1], TARNet_del.eval(), y_cont=True)
    
    PEHE_hat_TARNet_del = PEHE(ITE_true, ITE_est_TARNet_del)
    PEHE_hat_TARNet_del_ot = PEHE(ITE_true_ot, ITE_est_TARNet_del_ot)
    PEHE_hat_TARNet_del_mt = PEHE(ITE_true_mt, ITE_est_TARNet_del_mt)
    
    ihdp_results['TARNet_del'][0].append(PEHE_hat_TARNet_del)
    ihdp_results['TARNet_del'][1].append(PEHE_hat_TARNet_del_ot)
    ihdp_results['TARNet_del'][2].append(PEHE_hat_TARNet_del_mt)
    
    ITE_est_TARNet_imp = TARNet_pred(d_test[:,3:-1], TARNet_imp.eval(), y_cont=True)
    ITE_est_TARNet_imp_ot = TARNet_pred(d_test_ot[:,3:-1], TARNet_imp.eval(), y_cont=True)
    ITE_est_TARNet_imp_mt = TARNet_pred(d_test_mt[:,3:-1], TARNet_imp.eval(), y_cont=True)
    
    PEHE_hat_TARNet_imp = PEHE(ITE_true, ITE_est_TARNet_imp)
    PEHE_hat_TARNet_imp_ot = PEHE(ITE_true_ot, ITE_est_TARNet_imp_ot)
    PEHE_hat_TARNet_imp_mt = PEHE(ITE_true_mt, ITE_est_TARNet_imp_mt)
    
    ihdp_results['TARNet_imp'][0].append(PEHE_hat_TARNet_imp)
    ihdp_results['TARNet_imp'][1].append(PEHE_hat_TARNet_imp_ot)
    ihdp_results['TARNet_imp'][2].append(PEHE_hat_TARNet_imp_mt)
    
    ITE_est_TARNet_rew = TARNet_pred(d_test[:,3:-1], TARNet_rew.eval(), y_cont=True)
    ITE_est_TARNet_rew_ot = TARNet_pred(d_test_ot[:,3:-1], TARNet_rew.eval(), y_cont=True)
    ITE_est_TARNet_rew_mt = TARNet_pred(d_test_mt[:,3:-1], TARNet_rew.eval(), y_cont=True)
    
    PEHE_hat_TARNet_rew = PEHE(ITE_true, ITE_est_TARNet_rew)
    PEHE_hat_TARNet_rew_ot = PEHE(ITE_true_ot, ITE_est_TARNet_rew_ot)
    PEHE_hat_TARNet_rew_mt = PEHE(ITE_true_mt, ITE_est_TARNet_rew_mt)
    
    ihdp_results['TARNet_rew'][0].append(PEHE_hat_TARNet_rew)
    ihdp_results['TARNet_rew'][1].append(PEHE_hat_TARNet_rew_ot)
    ihdp_results['TARNet_rew'][2].append(PEHE_hat_TARNet_rew_mt)
    
    ITE_est_CFRMMD_del = CFRMMD_pred(d_test[:,3:-1], CFRMMD_del.eval(), y_cont=True)
    ITE_est_CFRMMD_del_ot = CFRMMD_pred(d_test_ot[:,3:-1], CFRMMD_del.eval(), y_cont=True)
    ITE_est_CFRMMD_del_mt = CFRMMD_pred(d_test_mt[:,3:-1], CFRMMD_del.eval(), y_cont=True)
    
    PEHE_hat_CFRMMD_del = PEHE(ITE_true, ITE_est_CFRMMD_del)
    PEHE_hat_CFRMMD_del_ot = PEHE(ITE_true_ot, ITE_est_CFRMMD_del_ot)
    PEHE_hat_CFRMMD_del_mt = PEHE(ITE_true_mt, ITE_est_CFRMMD_del_mt)
    
    ihdp_results['CFRMMD_del'][0].append(PEHE_hat_CFRMMD_del)
    ihdp_results['CFRMMD_del'][1].append(PEHE_hat_CFRMMD_del_ot)
    ihdp_results['CFRMMD_del'][2].append(PEHE_hat_CFRMMD_del_mt)
    
    ITE_est_CFRMMD_imp = CFRMMD_pred(d_test[:,3:-1], CFRMMD_imp.eval(), y_cont=True)
    ITE_est_CFRMMD_imp_ot = CFRMMD_pred(d_test_ot[:,3:-1], CFRMMD_imp.eval(), y_cont=True)
    ITE_est_CFRMMD_imp_mt = CFRMMD_pred(d_test_mt[:,3:-1], CFRMMD_imp.eval(), y_cont=True)
    
    PEHE_hat_CFRMMD_imp = PEHE(ITE_true, ITE_est_CFRMMD_imp)
    PEHE_hat_CFRMMD_imp_ot = PEHE(ITE_true_ot, ITE_est_CFRMMD_imp_ot)
    PEHE_hat_CFRMMD_imp_mt = PEHE(ITE_true_mt, ITE_est_CFRMMD_imp_mt)
    
    ihdp_results['CFRMMD_imp'][0].append(PEHE_hat_CFRMMD_imp)
    ihdp_results['CFRMMD_imp'][1].append(PEHE_hat_CFRMMD_imp_ot)
    ihdp_results['CFRMMD_imp'][2].append(PEHE_hat_CFRMMD_imp_mt)
    
    ITE_est_CFRMMD_rew = CFRMMD_pred(d_test[:,3:-1], CFRMMD_rew.eval(), y_cont=True)
    ITE_est_CFRMMD_rew_ot = CFRMMD_pred(d_test_ot[:,3:-1], CFRMMD_rew.eval(), y_cont=True)
    ITE_est_CFRMMD_rew_mt = CFRMMD_pred(d_test_mt[:,3:-1], CFRMMD_rew.eval(), y_cont=True)
    
    PEHE_hat_CFRMMD_rew = PEHE(ITE_true, ITE_est_CFRMMD_rew)
    PEHE_hat_CFRMMD_rew_ot = PEHE(ITE_true_ot, ITE_est_CFRMMD_rew_ot)
    PEHE_hat_CFRMMD_rew_mt = PEHE(ITE_true_mt, ITE_est_CFRMMD_rew_mt)
    
    ihdp_results['CFRMMD_rew'][0].append(PEHE_hat_CFRMMD_rew)
    ihdp_results['CFRMMD_rew'][1].append(PEHE_hat_CFRMMD_rew_ot)
    ihdp_results['CFRMMD_rew'][2].append(PEHE_hat_CFRMMD_rew_mt)
