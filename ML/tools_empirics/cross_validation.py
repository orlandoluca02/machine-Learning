# ======================================
# TWO-STEP CROSS-VALIDATION
# Author:      Tommaso Zipoli
# Last Mod:    11/11/2025
# ======================================

from copy import deepcopy
import numpy as np
import torch
from itertools import product
from splits import forward_split, backward_split
from search import grid_search_cross_validate, random_search_cross_validate, bayesian_search_cross_validate
from oos_nonlinear import get_oos_predictions
from plotting import (
    plot_decoder_weights,
    plot_inter_model_correlation_matrix,
    plot_featurewise_r2_comparison
)


# ======================================
# SECTION: Main Cross-Validation Entry
# ======================================
def run_two_step_cv(model_linear,
                    model_second,
                    tensor_data,
                    base_params,
                    grid,
                    search_fn,
                    split_fn=forward_split,
                    n_iter=30,
                    plot_scatter=True,
                    fig_path=None,
                    reuse_linear_params=None):
    """
    Run two-step CV:
      1) Tune base hyperparameters for each model (optionally reuse linear model params).
      2) Compute OOS predictions.
      3) Compute residual correlations.
      4) Tune ensemble α (fold-averaged R²).
    
    Parameters
    ----------
    reuse_linear_params : dict or None
        If provided, use these as the linear model best params instead of running CV.
    """

    # --- Step 1: Tune linear model (or reuse)
    if reuse_linear_params is not None:
        print("\nReusing precomputed linear model parameters")
        best_lin_params = reuse_linear_params
        # Optionally, you can compute OOS R² for logging
        _, _, _, _, r2_lin = get_oos_predictions(model_linear, tensor_data, best_lin_params, split_fn)
    else:
        print("\n" + "="*40)
        print("Tuning 'linear' model")
        print("="*40)
        best_lin_params, mse_lin = search_fn(
            model=model_linear,
            X=tensor_data,
            base_params=deepcopy(base_params),
            param_grid=grid,
            split_fn=split_fn,
            n_iter=n_iter,
            plot_scatter=plot_scatter,
            fig_path=fig_path
        )
        # Compute OOS R²
        _, _, _, _, r2_lin = get_oos_predictions(model_linear, tensor_data, best_lin_params, split_fn)

    # --- Step 1b: Tune second (nonlinear) model
    print("\n" + "="*40)
    print("Tuning 'second' model")
    print("="*40)
    best_second_params, mse_second = search_fn(
        model=model_second,
        X=tensor_data,
        base_params=deepcopy(base_params),
        param_grid=grid,
        split_fn=split_fn,
        n_iter=n_iter,
        plot_scatter=plot_scatter,
        fig_path=fig_path
    )

    # Plot loading heatmaps
    print("\nPlotting decoder weights...")

    plot_decoder_weights(
        model_linear.decoder.out.weight.detach().cpu().numpy(),
        fig_path,
        "decoder_weights_linear"
    )
    
    plot_decoder_weights(
        model_second.decoder.out.weight.detach().cpu().numpy(),
        fig_path,
        "decoder_weights_second"
    )


    # --- Step 2: Obtain OOS predictions
    print("\n" + "="*40)
    print("Generating out-of-sample predictions...")
    print("="*40)
    X_true_oos_lin, out_lin_oos, mask_lin, folds_lin, r2_lin = get_oos_predictions(
        model_linear, tensor_data, best_lin_params, split_fn
    )
    X_true_oos_sec, out_sec_oos, mask_sec, folds_sec, r2_second = get_oos_predictions(
        model_second, tensor_data, best_second_params, split_fn
    )
    print(f"    Linear model: R² = {r2_lin:.4f}")
    print(f"    Second model: R² = {r2_second:.4f}")

    # Check masks alignment
    assert np.all(mask_lin == mask_sec), "Validation masks differ between models."
    assert np.all(folds_lin == folds_sec), "Fold assignments differ between models."

    X_true = X_true_oos_lin
    out_lin = out_lin_oos
    out_second = out_sec_oos
    fold_ids = folds_lin

    # --- Step 2b: Residual correlations (OOS only)
    val_mask = mask_lin
    res_lin = (X_true - out_lin)[val_mask]
    res_second = (X_true - out_second)[val_mask]

    res_corr_feat = [np.corrcoef(res_lin[:, i], res_second[:, i])[0, 1]
                     for i in range(res_lin.shape[1])]
    res_corr_total = np.corrcoef(res_lin.flatten(), res_second.flatten())[0, 1]
    print(f"    Overall residual correlation: {res_corr_total:.4f}")

    print("\nPlotting cross-model latent similarity...")

    with torch.no_grad():
        F1 = model_linear.encoder(tensor_data).cpu().numpy()
        F2 = model_second.encoder(tensor_data).cpu().numpy()
    
    plot_inter_model_correlation_matrix(
        F1,
        F2,
        model_names=["Linear model", "Nonlinear model"],
        fig_path=fig_path
    )

    # --- Step 3: Tune ensemble α
    print("\n" + "="*40)
    print("Tuning nonlinear model weight (α)")
    print("="*40)
    alphas = np.linspace(0, 1, 11)
    r2_global_list, r2_feat_list = [], []
    unique_folds = np.unique(fold_ids[fold_ids >= 0])

    for alpha in alphas:
        out_ens = (1 - alpha) * out_lin + alpha * out_second
        fold_r2s, fold_r2feats = [], []

        for fold in unique_folds:
            fold_mask = fold_ids == fold
            y_true_fold = X_true[fold_mask]
            y_pred_fold = out_ens[fold_mask]

            # Global R²
            rss = np.sum((y_true_fold - y_pred_fold)**2)
            tss = np.sum((y_true_fold - np.mean(y_true_fold))**2)
            fold_r2s.append(1 - rss / tss)

            # Feature-wise R²
            fold_r2_feat = 1 - np.sum((y_true_fold - y_pred_fold)**2, axis=0) / (
                np.sum((y_true_fold - np.mean(y_true_fold, axis=0, keepdims=True))**2, axis=0) + 1e-12
            )
            fold_r2feats.append(fold_r2_feat)

        r2_global_list.append(np.mean(fold_r2s))
        r2_feat_list.append(np.mean(fold_r2feats, axis=0))
        print(f"    α = {alpha:.1f}: mean R² = {r2_global_list[-1]:.4f}")

    best_idx = np.argmax(r2_global_list)
    alpha_best = alphas[best_idx]
    r2_ens_best = r2_global_list[best_idx]
    r2_feat_ens = r2_feat_list[best_idx]

    print(f"\nBest α = {alpha_best:.2f} | Ensemble mean R² = {r2_ens_best:.4f}")

    # --- Feature-wise R² (single models)
    eps = 1e-6  # prevent division by tiny variance
    X_centered = X_true - X_true.mean(axis=0, keepdims=True)
    den = np.maximum((X_centered**2).sum(axis=0), eps)
    
    r2_feat_lin = 1 - ((X_true - out_lin)**2).sum(axis=0) / den
    r2_feat_second = 1 - ((X_true - out_second)**2).sum(axis=0) / den
    r2_feat_ens = 1 - ((X_true - out_ens)**2).sum(axis=0) / den
    
    # --- Ensemble gain per feature
    delta_feat = r2_feat_ens - r2_feat_lin
    delta_feat = np.clip(delta_feat, -1, 1)

    return {
        "best_alpha": alpha_best,
        "best_lin_params": best_lin_params,
        "best_second_params": best_second_params,
        "r2_linear": r2_lin,
        "r2_second": r2_second,
        "r2_ensemble": r2_ens_best,
        "r2_feat_lin": r2_feat_lin,
        "r2_feat_second": r2_feat_second,
        "r2_feat_ens": r2_feat_ens,
        "delta_feat": delta_feat,
        "res_corr_feat": res_corr_feat,
        "res_corr_total": res_corr_total,
        "mask": mask_lin,
        "fold_ids": fold_ids
    }
