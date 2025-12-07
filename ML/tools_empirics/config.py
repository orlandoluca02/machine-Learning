# ======================================
# PARAMETER CONFIGURATION
# Author:      Tommaso Zipoli
# Last Mod:    08/11/2025
# ======================================

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from model import SinActivation, SoftSine

class Params:
    def __init__(self):
        # ==========================
        # Data features
        # ==========================
        self.filename = "processed2.xlsx"
        self.scaler = None  # {None, MinMaxScaler(), StandardScaler()}
        self.col_to_drop = [
            'ACOGNO',
            'ANDENOx',
            'TWEXAFEGSMTHx',
            'UMCSENTx',
            'CP3Mx',
            'COMPAPFFx',
            'VIXCLSx'
        ]   

        # ==========================
        # Autoencoder architecture
        # ==========================
        self.hidden_dim_1 = 20
        self.hidden_dim_2 = 20
        self.num_factors = 4

        # ==========================
        # Regularization settings
        # ========================== 
        self.lambda_lasso = 30
        self.lambda_orth = 0e-0
        self.use_lasso_reg = True
        self.use_orth_reg = True
        
        # ==========================
        # Training settings
        # ==========================
        self.device = torch.device("cpu")
        self.batch_size = 32
        self.num_epochs = 1000
        self.patience = 20
        self.verbose = True
        self.print_every = 100
        self.lr = 1e-3
        self.seed = 42

        # ==========================
        # Cross-validation
        # ==========================
        self.search_fn = 'grid'       # options: 'grid', 'random', 'bayesian'
        self.split_fn = 'forward'     # options: 'forward', 'backward'
        self.n_iter = 20
        self.len_valset = 28
        self.num_folds = 5

        # ==========================
        # Tuning values
        # ==========================
        self.activation_grid = {
            'linear': nn.Identity(),
            #'tanh': nn.Tanh(),
            #'sin' : SinActivation(),
            'SiLU' : nn.SiLU()
        }
        #fine
        self.grid = {
            'num_factors' : [4, 6],
            'hidden_dim_1': [10, 20],
            'lambda_lasso': [2e-2, 5e-2],
            'lambda_orth' : [1e-3, 1e-2]
        }
