
"""

The code contains experiments with Jobs dataset

"""

# Packages

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from main import *
from benchmarks import *

# Data

# Description:  
#     Outcome: Y - binary {0,1} representing employment status (Y=1 employed, Y=0 unemployed)
#     Treatment: T - participation in government training program (T=1 participated, T=0 didn't participate)
#     Observed covariates: X_vec = [x1, ..., xp] - vector of covariates (p=7)

# The data comprises of 3212 observations combining 722 observations from randomized trial (RCT) 
# and 2490 observations from observational study: 7 covariates, treatment, outcome.
# -> True ITE is not accessible in this dataset, so as evaluation metric we use policy risk
#    evaluated on the randomized portion of the data.

# Load data
data = np.load('data/jobs.npy')
n = data.shape[0]
p = data.shape[1] - 3

np.random.seed(0)
torch.manual_seed(0)

# RCT portion
RCT_No = 722
d_rand = data[:RCT_No,:]
np.random.shuffle(d_rand)
d_nonrand = data[RCT_No:,:]

# Hyperparameters 

# hyperparams_list = {
#     'layer_size_representation': [50, 100, 200],
#     'layer_size_hypothesis': [50, 100, 200],
#     'learn_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001],
#     'dropout_rate':  [0.1, 0.2, 0.3],
#     'num_iterations': [100, 200, 300], 
#     'batch_size': [200, 300, 500],
#     'alphas': [10**(k/2) for k in np.linspace(-4,2,9)],
#     'betas': [10**(k/2) for k in np.linspace(-4,2,9)],
#     'lambdas': [0.0005, 0.0001, 0.00005]
#     }

hyperpar_opt_adv = {
        'layer_size_rep': [100],
        'layer_size_hyp': [100],
        'lr': [0.005],
        'drop': [0.3],
        'n_iter': [300],
        'b_size': [300],
        'alpha': [0.75],
        'beta': [0.75],
        'lam': [0.0005]
        }

hyperpar_opt_tar = {
        'layer_size_rep': [200],
        'layer_size_hyp': [200],
        'lr': [0.0005],
        'drop': [0.1],
        'n_iter': [300],
        'b_size': [200],
        'lam': [0.00005]
        }

hyperpar_opt_mmd = {
        'layer_size_rep': [200],
        'layer_size_hyp': [200],
        'lr': [0.0005],
        'drop': [0.1],
        'n_iter': [300],
        'b_size': [300],
        'alpha': [0.32],
        'lam': [0.00005]
        }

# Experiments
    
# Save results for n_sim runs
jobs_results = {
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

n_sims = 10
for run in range(n_sims):

    print(run)

    # Train-Test split -> shape [Y, T, X, R]
    d_train_val, d_test = train_test_split(d_rand, test_size=0.1, shuffle=False)
    d_train = np.concatenate((d_nonrand, d_train_val), axis=0)
    
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
    
    # Performance evaluation metric: policy risk
    def Rpol(data_test, ITE_est):
        if type(ITE_est) == torch.Tensor:
            ITE_est = ITE_est.cpu().detach().numpy()
        p_f1 = ITE_est > 0
        p_f0 = ITE_est <= 0
        prob_pf1 = np.mean(p_f1) 
        if np.sum((p_f1) & (data_test[:,1]==1)) != 0 and np.sum((p_f0) & (data_test[:,1]==0)) != 0:
            Rpol = 1 - (np.mean(data_test[(p_f1) & (data_test[:,1]==1), 0])*prob_pf1 +\
                        np.mean(data_test[(p_f0) & (data_test[:,1]==0), 0])*(1-prob_pf1))
        else:
            Rpol = np.nan 
        return Rpol
            
    # Implementation of method
    method_adv = MTRNet(data_train=d_train, y_cont=False, hyperparams=hyperpar_opt_adv)
        
    # Performance evaluation for method (d_test_ot, d_test_mt, d_test)
    ITE_est_adv = MTRNet_pred(d_test[:,2:-1], method_adv.eval(), y_cont=False)
    ITE_est_adv_ot = MTRNet_pred(d_test_ot[:,2:-1], method_adv.eval(), y_cont=False)
    ITE_est_adv_mt = MTRNet_pred(d_test_mt[:,2:-1], method_adv.eval(), y_cont=False)
    
    Rpol_hat_adv = Rpol(d_test, ITE_est_adv)
    Rpol_hat_adv_ot = Rpol(d_test_ot, ITE_est_adv_ot)
    Rpol_hat_adv_mt = Rpol(d_test_mt, ITE_est_adv_mt)
    
    jobs_results['MTHD_adv'][0].append(Rpol_hat_adv)
    jobs_results['MTHD_adv'][1].append(Rpol_hat_adv_ot)
    jobs_results['MTHD_adv'][2].append(Rpol_hat_adv_mt)

    # Implementation of benchmarks - 3 * 4 models (d_train_del, d_train_imp, d_train_del + w)
    Linear_del0 = Linear(d_train_del[d_train_del[:,1] == 0,:], y_cont=False)
    Linear_imp0 = Linear(d_train_imp[d_train_imp[:,1] == 0,:], y_cont=False)
    Linear_rew0 = Linear_w(d_train_rew[d_train_rew[:,1] == 0,:], y_cont=False)
    
    Linear_del1 = Linear(d_train_del[d_train_del[:,1] == 1,:], y_cont=False)
    Linear_imp1 = Linear(d_train_imp[d_train_imp[:,1] == 1,:], y_cont=False)
    Linear_rew1 = Linear_w(d_train_rew[d_train_rew[:,1] == 1,:], y_cont=False)
    
    CF_del = CF(d_train_del, y_cont=False)
    CF_imp = CF(d_train_imp, y_cont=False)
    CF_rew = CF_w(d_train_rew, y_cont=False)
       
    TARNet_del = TARNet(data_train=d_train_del, y_cont=False, hyperparams=hyperpar_opt_tar)
    TARNet_imp = TARNet(data_train=d_train_imp, y_cont=False, hyperparams=hyperpar_opt_tar)
    TARNet_rew = TARNet_w(data_train=d_train_rew, y_cont=False, hyperparams=hyperpar_opt_tar)
    CFRMMD_del = CFRMMD(data_train=d_train_del, y_cont=False, hyperparams=hyperpar_opt_mmd)
    CFRMMD_imp = CFRMMD(data_train=d_train_imp, y_cont=False, hyperparams=hyperpar_opt_mmd)
    CFRMMD_rew = CFRMMD_w(data_train=d_train_rew, y_cont=False, hyperparams=hyperpar_opt_mmd)
    
    # Performance evaluation for benchmarks (d_test_ot, d_test_mt, d_test)
    ITE_est_Linear_del = Linear_pred(d_test[:,2:-1], Linear_del0, Linear_del1, y_cont=False)
    ITE_est_Linear_del_ot = Linear_pred(d_test_ot[:,2:-1], Linear_del0, Linear_del1, y_cont=False)
    ITE_est_Linear_del_mt = Linear_pred(d_test_mt[:,2:-1], Linear_del0, Linear_del1, y_cont=False)
    
    Rpol_hat_Linear_del = Rpol(d_test, ITE_est_Linear_del)
    Rpol_hat_Linear_del_ot = Rpol(d_test_ot, ITE_est_Linear_del_ot)
    Rpol_hat_Linear_del_mt = Rpol(d_test_mt, ITE_est_Linear_del_mt)
    
    jobs_results['Linear_del'][0].append(Rpol_hat_Linear_del)
    jobs_results['Linear_del'][1].append(Rpol_hat_Linear_del_ot)
    jobs_results['Linear_del'][2].append(Rpol_hat_Linear_del_mt)
    
    ITE_est_Linear_imp = Linear_pred(d_test[:,2:-1], Linear_imp0, Linear_imp1, y_cont=False)
    ITE_est_Linear_imp_ot = Linear_pred(d_test_ot[:,2:-1], Linear_imp0, Linear_imp1, y_cont=False)
    ITE_est_Linear_imp_mt = Linear_pred(d_test_mt[:,2:-1], Linear_imp0, Linear_imp1, y_cont=False)
    
    Rpol_hat_Linear_imp = Rpol(d_test, ITE_est_Linear_imp)
    Rpol_hat_Linear_imp_ot = Rpol(d_test_ot, ITE_est_Linear_imp_ot)
    Rpol_hat_Linear_imp_mt = Rpol(d_test_mt, ITE_est_Linear_imp_mt)
    
    jobs_results['Linear_imp'][0].append(Rpol_hat_Linear_imp)
    jobs_results['Linear_imp'][1].append(Rpol_hat_Linear_imp_ot)
    jobs_results['Linear_imp'][2].append(Rpol_hat_Linear_imp_mt)
    
    ITE_est_Linear_rew = Linear_pred(d_test[:,2:-1], Linear_rew0, Linear_rew1, y_cont=False)
    ITE_est_Linear_rew_ot = Linear_pred(d_test_ot[:,2:-1], Linear_rew0, Linear_rew1, y_cont=False)
    ITE_est_Linear_rew_mt = Linear_pred(d_test_mt[:,2:-1], Linear_rew0, Linear_rew1, y_cont=False)
    
    Rpol_hat_Linear_rew = Rpol(d_test, ITE_est_Linear_rew)
    Rpol_hat_Linear_rew_ot = Rpol(d_test_ot, ITE_est_Linear_rew_ot)
    Rpol_hat_Linear_rew_mt = Rpol(d_test_mt, ITE_est_Linear_rew_mt)
    
    jobs_results['Linear_rew'][0].append(Rpol_hat_Linear_rew)
    jobs_results['Linear_rew'][1].append(Rpol_hat_Linear_rew_ot)
    jobs_results['Linear_rew'][2].append(Rpol_hat_Linear_rew_mt)
    
    ITE_est_CF_del = CF_pred(d_test[:,2:-1], CF_del, y_cont=False)
    ITE_est_CF_del_ot = CF_pred(d_test_ot[:,2:-1], CF_del, y_cont=False)
    ITE_est_CF_del_mt = CF_pred(d_test_mt[:,2:-1], CF_del, y_cont=False)
    
    Rpol_hat_CF_del = Rpol(d_test, ITE_est_CF_del)
    Rpol_hat_CF_del_ot = Rpol(d_test_ot, ITE_est_CF_del_ot)
    Rpol_hat_CF_del_mt = Rpol(d_test_mt, ITE_est_CF_del_mt)
    
    jobs_results['CF_del'][0].append(Rpol_hat_CF_del)
    jobs_results['CF_del'][1].append(Rpol_hat_CF_del_ot)
    jobs_results['CF_del'][2].append(Rpol_hat_CF_del_mt)
    
    ITE_est_CF_imp = CF_pred(d_test[:,2:-1], CF_imp, y_cont=False)
    ITE_est_CF_imp_ot = CF_pred(d_test_ot[:,2:-1], CF_imp, y_cont=False)
    ITE_est_CF_imp_mt = CF_pred(d_test_mt[:,2:-1], CF_imp, y_cont=False)
    
    Rpol_hat_CF_imp = Rpol(d_test, ITE_est_CF_imp)
    Rpol_hat_CF_imp_ot = Rpol(d_test_ot, ITE_est_CF_imp_ot)
    Rpol_hat_CF_imp_mt = Rpol(d_test_mt, ITE_est_CF_imp_mt)
    
    jobs_results['CF_imp'][0].append(Rpol_hat_CF_imp)
    jobs_results['CF_imp'][1].append(Rpol_hat_CF_imp_ot)
    jobs_results['CF_imp'][2].append(Rpol_hat_CF_imp_mt)
    
    ITE_est_CF_rew = CF_pred(d_test[:,2:-1], CF_rew, y_cont=False)
    ITE_est_CF_rew_ot = CF_pred(d_test_ot[:,2:-1], CF_rew, y_cont=False)
    ITE_est_CF_rew_mt = CF_pred(d_test_mt[:,2:-1], CF_rew, y_cont=False)
    
    Rpol_hat_CF_rew = Rpol(d_test, ITE_est_CF_rew)
    Rpol_hat_CF_rew_ot = Rpol(d_test_ot, ITE_est_CF_rew_ot)
    Rpol_hat_CF_rew_mt = Rpol(d_test_mt, ITE_est_CF_rew_mt)
    
    jobs_results['CF_rew'][0].append(Rpol_hat_CF_rew)
    jobs_results['CF_rew'][1].append(Rpol_hat_CF_rew_ot)
    jobs_results['CF_rew'][2].append(Rpol_hat_CF_rew_mt)
    
    ITE_est_TARNet_del = TARNet_pred(d_test[:,2:-1], TARNet_del.eval(), y_cont=False)
    ITE_est_TARNet_del_ot = TARNet_pred(d_test_ot[:,2:-1], TARNet_del.eval(), y_cont=False)
    ITE_est_TARNet_del_mt = TARNet_pred(d_test_mt[:,2:-1], TARNet_del.eval(), y_cont=False)
    
    Rpol_hat_TARNet_del = Rpol(d_test, ITE_est_TARNet_del)
    Rpol_hat_TARNet_del_ot = Rpol(d_test_ot, ITE_est_TARNet_del_ot)
    Rpol_hat_TARNet_del_mt = Rpol(d_test_mt, ITE_est_TARNet_del_mt)
    
    jobs_results['TARNet_del'][0].append(Rpol_hat_TARNet_del)
    jobs_results['TARNet_del'][1].append(Rpol_hat_TARNet_del_ot)
    jobs_results['TARNet_del'][2].append(Rpol_hat_TARNet_del_mt)
    
    ITE_est_TARNet_imp = TARNet_pred(d_test[:,2:-1], TARNet_imp.eval(), y_cont=False)
    ITE_est_TARNet_imp_ot = TARNet_pred(d_test_ot[:,2:-1], TARNet_imp.eval(), y_cont=False)
    ITE_est_TARNet_imp_mt = TARNet_pred(d_test_mt[:,2:-1], TARNet_imp.eval(), y_cont=False)
    
    Rpol_hat_TARNet_imp = Rpol(d_test, ITE_est_TARNet_imp)
    Rpol_hat_TARNet_imp_ot = Rpol(d_test_ot, ITE_est_TARNet_imp_ot)
    Rpol_hat_TARNet_imp_mt = Rpol(d_test_mt, ITE_est_TARNet_imp_mt)
    
    jobs_results['TARNet_imp'][0].append(Rpol_hat_TARNet_imp)
    jobs_results['TARNet_imp'][1].append(Rpol_hat_TARNet_imp_ot)
    jobs_results['TARNet_imp'][2].append(Rpol_hat_TARNet_imp_mt)
    
    ITE_est_TARNet_rew = TARNet_pred(d_test[:,2:-1], TARNet_rew.eval(), y_cont=False)
    ITE_est_TARNet_rew_ot = TARNet_pred(d_test_ot[:,2:-1], TARNet_rew.eval(), y_cont=False)
    ITE_est_TARNet_rew_mt = TARNet_pred(d_test_mt[:,2:-1], TARNet_rew.eval(), y_cont=False)
    
    Rpol_hat_TARNet_rew = Rpol(d_test, ITE_est_TARNet_rew)
    Rpol_hat_TARNet_rew_ot = Rpol(d_test_ot, ITE_est_TARNet_rew_ot)
    Rpol_hat_TARNet_rew_mt = Rpol(d_test_mt, ITE_est_TARNet_rew_mt)
    
    jobs_results['TARNet_rew'][0].append(Rpol_hat_TARNet_rew)
    jobs_results['TARNet_rew'][1].append(Rpol_hat_TARNet_rew_ot)
    jobs_results['TARNet_rew'][2].append(Rpol_hat_TARNet_rew_mt)
    
    ITE_est_CFRMMD_del = CFRMMD_pred(d_test[:,2:-1], CFRMMD_del.eval(), y_cont=False)
    ITE_est_CFRMMD_del_ot = CFRMMD_pred(d_test_ot[:,2:-1], CFRMMD_del.eval(), y_cont=False)
    ITE_est_CFRMMD_del_mt = CFRMMD_pred(d_test_mt[:,2:-1], CFRMMD_del.eval(), y_cont=False)
    
    Rpol_hat_CFRMMD_del = Rpol(d_test, ITE_est_CFRMMD_del)
    Rpol_hat_CFRMMD_del_ot = Rpol(d_test_ot, ITE_est_CFRMMD_del_ot)
    Rpol_hat_CFRMMD_del_mt = Rpol(d_test_mt, ITE_est_CFRMMD_del_mt)
    
    jobs_results['CFRMMD_del'][0].append(Rpol_hat_CFRMMD_del)
    jobs_results['CFRMMD_del'][1].append(Rpol_hat_CFRMMD_del_ot)
    jobs_results['CFRMMD_del'][2].append(Rpol_hat_CFRMMD_del_mt)
    
    ITE_est_CFRMMD_imp = CFRMMD_pred(d_test[:,2:-1], CFRMMD_imp.eval(), y_cont=False)
    ITE_est_CFRMMD_imp_ot = CFRMMD_pred(d_test_ot[:,2:-1], CFRMMD_imp.eval(), y_cont=False)
    ITE_est_CFRMMD_imp_mt = CFRMMD_pred(d_test_mt[:,2:-1], CFRMMD_imp.eval(), y_cont=False)
    
    Rpol_hat_CFRMMD_imp = Rpol(d_test, ITE_est_CFRMMD_imp)
    Rpol_hat_CFRMMD_imp_ot = Rpol(d_test_ot, ITE_est_CFRMMD_imp_ot)
    Rpol_hat_CFRMMD_imp_mt = Rpol(d_test_mt, ITE_est_CFRMMD_imp_mt)
    
    jobs_results['CFRMMD_imp'][0].append(Rpol_hat_CFRMMD_imp)
    jobs_results['CFRMMD_imp'][1].append(Rpol_hat_CFRMMD_imp_ot)
    jobs_results['CFRMMD_imp'][2].append(Rpol_hat_CFRMMD_imp_mt)
    
    ITE_est_CFRMMD_rew = CFRMMD_pred(d_test[:,2:-1], CFRMMD_rew.eval(), y_cont=False)
    ITE_est_CFRMMD_rew_ot = CFRMMD_pred(d_test_ot[:,2:-1], CFRMMD_rew.eval(), y_cont=False)
    ITE_est_CFRMMD_rew_mt = CFRMMD_pred(d_test_mt[:,2:-1], CFRMMD_rew.eval(), y_cont=False)
    
    Rpol_hat_CFRMMD_rew = Rpol(d_test, ITE_est_CFRMMD_rew)
    Rpol_hat_CFRMMD_rew_ot = Rpol(d_test_ot, ITE_est_CFRMMD_rew_ot)
    Rpol_hat_CFRMMD_rew_mt = Rpol(d_test_mt, ITE_est_CFRMMD_rew_mt)
    
    jobs_results['CFRMMD_rew'][0].append(Rpol_hat_CFRMMD_rew)
    jobs_results['CFRMMD_rew'][1].append(Rpol_hat_CFRMMD_rew_ot)
    jobs_results['CFRMMD_rew'][2].append(Rpol_hat_CFRMMD_rew_mt)
