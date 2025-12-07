# ======================================
# OOS PREDICTIONS FOR AUTOENCODER MODELS
# Author:      Tommaso Zipoli
# Last Mod:    11/11/2025
# ======================================

from copy import deepcopy
import torch
import numpy as np
from splits import forward_split, backward_split


def get_oos_predictions(model, X, params, split_fn=forward_split):
    """
    Train model with K-fold CV and return *out-of-sample* predictions.

    Returns:
        y_true_oos: np.ndarray       (only validation samples)
        y_pred_oos: np.ndarray       (OOS predictions, same shape)
        val_mask: np.ndarray[bool]   (True for validation samples in X)
        fold_ids: np.ndarray[int]    (Fold index per sample, -1 for train)
        r2_avg: float                (mean fold-wise R²)
    """
    X_np = X.cpu().numpy()
    N = len(X_np)
    val_mask = np.zeros(N, dtype=bool)
    fold_ids = np.full(N, -1, dtype=int)

    y_pred_all = np.zeros_like(X_np)
    r2_list = []

    for fold, (train_idx, val_idx) in enumerate(split_fn(X, params.num_folds, params.len_valset)):
        X_train = X[train_idx].to(params.device)
        X_val = X[val_idx].to(params.device)

        model_fold = deepcopy(model).to(params.device)
        model_fold.reset_parameters()
        optimizer = torch.optim.Adam(model_fold.parameters(), lr=params.lr)

        # Training loop
        for epoch in range(params.num_epochs):
            model_fold.train()
            optimizer.zero_grad()
            output, factors = model_fold(X_train)
            loss = model_fold.loss(
                X_train, output, factors,
                params.lambda_lasso,
                params.lambda_orth,
                params.use_lasso_reg,
                params.use_orth_reg,
            )
            loss.backward()
            optimizer.step()

        # Validation predictions
        model_fold.eval()
        with torch.no_grad():
            output, _ = model_fold(X_val)
            output_np = output.cpu().numpy()
            X_val_np = X_val.cpu().numpy()

            y_pred_all[val_idx] = output_np
            val_mask[val_idx] = True
            fold_ids[val_idx] = fold

            # Fold R²
            rss = np.sum((X_val_np - output_np) ** 2)
            tss = np.sum((X_val_np - np.mean(X_val_np)) ** 2)
            r2_list.append(1 - rss / tss)

    r2_avg = np.mean(r2_list)
    return X_np, y_pred_all, val_mask, fold_ids, r2_avg

