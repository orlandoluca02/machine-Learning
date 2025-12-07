# ======================================
# AUTOENCODER FACTOR MODEL - MAIN FILE
# Author:      Tommaso Zipoli
# Last Mod:    11/11/2025
# ======================================

# ======================================
# SECTION: Imports
# ======================================
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# ======================================
# SECTION: Setup Paths & Imports from Tools
# ======================================
empirics_path = Path(__file__).parent.resolve()
root_path = empirics_path.parent
data_path = root_path / "data"
tools_path = root_path / "tools_empirics"
fig_path = root_path / "figures"
fig_path.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(tools_path))

# Internal modules
from config import Params
from data import Data
from model import Autoencoder
from cross_validation import run_two_step_cv
from splits import forward_split, backward_split
from search import grid_search_cross_validate, random_search_cross_validate, bayesian_search_cross_validate
from plotting import (
    plot_decoder_weights,
    plot_inter_model_correlation_matrix,
    plot_featurewise_r2_comparison
)
from utils import set_seed
from oos_nonlinear import get_oos_predictions

# ======================================
# SECTION: Maps
# ======================================
SEARCH_FN_MAP = {
    'grid': grid_search_cross_validate,
    'random': random_search_cross_validate,
    'bayesian': bayesian_search_cross_validate
}

SPLIT_FN_MAP = {
    'forward': forward_split,
    'backward': backward_split
}

# ======================================
# SECTION: Main Function
# ======================================
def main(params):
    # --- Load and preprocess data
    data = Data(str(data_path / params.filename))
    data.process_EA(params.scaler)
    data.to_tensor()
    print(f"Data matrix is of shape {data.shape}")

    # --- Prepare containers
    output_dict, factors_dict = {}, {}
    best_params_dict = {}    
    results_dict = {}

    # --- Set seed
    set_seed(params.seed)
    
    # --- Loop over activation functions
    for key, activation_fn in params.activation_grid.items():
        if key == "linear":
            continue
    
        print("\n" + "=" * 40)
        print(f"Activation: {key}")
        print("=" * 40)
    
        set_seed(params.seed)
    
        # Define nonlinear model
        model_second = Autoencoder(
            data_dim=data.N,
            hidden_dim_1=params.hidden_dim_1,
            hidden_dim_2=params.hidden_dim_2,
            num_factors=params.num_factors,
            activation=activation_fn
        )
    
        # Run two-step CV, reuse linear best params
        results = run_two_step_cv(
            model_linear=Autoencoder(
                data_dim=data.N,
                hidden_dim_1=params.hidden_dim_1,
                hidden_dim_2=params.hidden_dim_2,
                num_factors=params.num_factors,
                activation=params.activation_grid["linear"]
            ),
            model_second=model_second,
            tensor_data=data.tensor,
            base_params=params,
            grid=params.grid,
            search_fn=SEARCH_FN_MAP[params.search_fn],
            split_fn=SPLIT_FN_MAP[params.split_fn],
            n_iter=params.n_iter,
            plot_scatter=False,
            fig_path=fig_path,
            reuse_linear_params=best_params_dict.get("linear")
        )
    
        results_dict[key] = results


        print(f"\n>>> Ensemble Results ({key}) <<<")
        print(f"  Linear OOS R²   = {results['r2_linear']:.4f}")
        print(f"  Second OOS R²   = {results['r2_second']:.4f}")
        print(f"  Ensemble OOS R² = {results['r2_ensemble']:.4f}")
        print(f"  Optimal α       = {results['best_alpha']:.2f}\n")

        # Store output for ensemble-level diagnostics later
        output_dict[key] = results

        # --- Plot feature-wise R² comparison
        plot_featurewise_r2_comparison(
            results["r2_feat_lin"],
            results["r2_feat_second"],
            results["delta_feat"],
            second_model_label=key,
            fig_path = fig_path,
            filename=f"featurewise_r2_linear_vs_{key}.png"
        )

    print("\n=== All models evaluated successfully ===")



# ======================================
# SECTION: Entry Point
# ======================================
if __name__ == "__main__":
    params = Params()
    main(params)
