# ======================================
# PROJECT README
# Author: Tommaso Zipoli
# Last Modified: 21/10/2025
# ======================================

This document describes the role of each Python file in the project.

----------------------------------------
📁 empirics/
----------------------------------------

main_nonlinear.py
    - Entry point for training and evaluating the non-linear Autoencoder model.
    - Loads data, runs two-step cross-validation, and return oos performance metrics.

----------------------------------------
📁 tools_empirics/
----------------------------------------

config.py
    - Defines the Params class, which stores all configuration values.
    - Includes model architecture, training settings, cross-validation settings, and tuning grids.

cross_validation.py
    - Contains logic for running two-step cross-validation.
    - Step 1: tuning affine and non-affine model separately.
    - Step 2: tuning ensemble weight.

data.py
    - Handles data loading, scaling, export, and conversion to PyTorch tensors.
    - Wraps input data in a standardized structure.

model.py
    - Defines the Autoencoder model architecture, including the forward pass.  
    - The custom loss function computes reconstruction loss and integrates optional regularization terms, which are controlled via configuration flags.

plotting.py
    - Utility functions for visualizing results.

regularization.py
    - Implements different regularization strategies:
    - lasso: compute L1 norm of decoder weights to encourage sparsity.
    - orth_loss: structural penalty, computes correlation between latent factors.


oos_nonlinear.py
    - Contains a function for model-based forecasting and R^2 computing

search.py
    - Implements different search strategies:
        - grid_search_cross_validate
        - random_search_cross_validate
        - bayesian_search_cross_validate
    - These are plugged into crossval.py via Params.search_fn.

splits.py
    - Provides data splitting functions (e.g., forward_split, backward_split).
    - Used for cross-validation over time-series or ordered data.

utils.py
    - Miscellaneous helpers.