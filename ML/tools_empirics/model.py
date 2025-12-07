# ======================================
# AUTOENCODER
# Author:      Tommaso Zipoli
# Last Mod:    08/11/2025
# ======================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import regularization


# ====================
#   AUTOENCODER
# ====================

class Autoencoder(nn.Module):
    def __init__(self, data_dim, hidden_dim_1, hidden_dim_2, num_factors, activation):
        super().__init__()
        
        self.encoder = Encoder(data_dim, hidden_dim_1, hidden_dim_2, activation, num_factors)
        self.decoder = Decoder(data_dim, hidden_dim_1, activation, num_factors)
        self.mse = nn.MSELoss(reduction='sum')
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if layer is self:
                continue
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        factors = self.encoder(x)
        output = self.decoder(factors)
        return output, factors

    def loss(self, x, x_hat, factors, lambda_lasso, lambda_orth,
             use_lasso_reg=False, use_orth_reg=False):
        """
        Total loss = reconstruction
                    + (optional) L1 regularization
                    + (optional) factor orthogonality loss
                    + (optional) anchor correlation loss.
        """
        total_loss = self.mse(x, x_hat)

        if use_lasso_reg:
            reg_loss = lambda_lasso * regularization.lasso(self.decoder)
            total_loss += reg_loss

        if use_orth_reg:
            orth_loss = lambda_orth * regularization.orth_loss(factors.detach())
            total_loss += orth_loss


        return total_loss


# ====================
#      ENCODER
# ====================

class Encoder(nn.Module):
    def __init__(self, data_dim, hidden_dim_1, hidden_dim_2, activation, num_factors):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(data_dim, hidden_dim_1),
            activation,
            nn.Linear(hidden_dim_1, num_factors),
            nn.BatchNorm1d(num_factors, affine=False)
        )

    def forward(self, x):
        return self.out(x)


# ====================
#      DECODER
# ====================

class Decoder(nn.Module):
    def __init__(self, data_dim, hidden_dim_1, activation, num_factors):
        super().__init__()
        self.out = nn.Linear(num_factors, data_dim)
        
    def forward(self, x):
        return self.out(x)


# ====================
#      ACTIVATION FUNCTIONS
# ====================

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SoftSine(nn.Module):
    def forward(self, x, lam=0.3):
        """
        lam: coefficiente di mescolanza tra SiLU e sin(x)
             0 -> SiLU puro
             1 -> sin(x) puro
        """
        return (1 - lam) * F.silu(x) + lam * torch.sin(x)