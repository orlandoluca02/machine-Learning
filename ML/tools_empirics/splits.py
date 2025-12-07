# ======================================
# SPLIT
# Author:      Tommaso Zipoli
# Last Mod:    06/11/2025
# ======================================

from sklearn.model_selection import TimeSeriesSplit

def backward_split(X, num_folds, len_valset):
    total_len = len(X)
    for k in range(num_folds):
        start_val = total_len - (k + 1) * len_valset
        end_val = total_len - k * len_valset
        val_idx = list(range(start_val, end_val))
        train_idx = list(range(0, start_val))
        yield train_idx, val_idx

def forward_split(X, num_folds, len_valset):
    tscv = TimeSeriesSplit(n_splits=num_folds, test_size=len_valset)
    for train_idx, val_idx in tscv.split(X):
        yield train_idx, val_idx
