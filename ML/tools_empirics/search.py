# ======================================
# SEARCH
# Author:      Tommaso Zipoli
# Last Mod:    11/11/2025
# ======================================

# ======================================
# SECTION: Imports
# ======================================
import math
import random
import optuna
import warnings
from copy import deepcopy
from itertools import product
from splits import forward_split, backward_split
from oos_nonlinear import get_oos_predictions
from plotting import plot_mse_scatter


# ======================================
# SECTION: Grid Search
# ======================================
def grid_search_cross_validate(model, X, base_params, param_grid,
                               split_fn=forward_split, n_iter=30,
                               plot_scatter=False, fig_path=None):
    """
    Perform exhaustive grid search cross-validation using the updated
    get_oos_predictions() interface.
    """
    keys, values = zip(*param_grid.items())
    combinations = list(product(*values))

    best_r2 = float('-inf')
    best_params_dict = {}
    results = []

    for combo in combinations:
        combo_params = dict(zip(keys, combo))
        # print(f"\n=== Testing combination: {combo_params} ===")

        # --- Clone and update base parameters ---
        current_params = deepcopy(base_params)
        for key, val in combo_params.items():
            setattr(current_params, key, val)

        model_instance = deepcopy(model)

        # --- Run OOS predictions and collect results ---
        _, _, _, _, r2 = get_oos_predictions(model_instance, X, current_params, split_fn)

        results.append((*combo, r2))
        print(f"\nTesting combination: {combo_params},  OOS R² = {r2:.4f}")

        if math.isnan(r2):
            print("Warning: OOS R² is NaN! Skipping update.")
            continue

        if r2 > best_r2:
            best_r2 = r2
            best_params_dict = combo_params

    print(f"\n>> Best Params: {best_params_dict}")
    print(f">> Best OOS R²: {best_r2:.4f}")

    # --- Store best parameters back into a params object ---
    best_params_obj = deepcopy(base_params)
    for key, val in best_params_dict.items():
        setattr(best_params_obj, key, val)

    # --- Optional: Plot grid results ---
    if plot_scatter:
        plot_mse_scatter(results=results, keys=keys, param_grid=param_grid, fig_path=fig_path)

    return best_params_obj, best_r2


# ======================================
# SECTION: Random Search
# ======================================
def random_search_cross_validate(model, X, base_params, param_grid,
                                 split_fn=forward_split, n_iter=30,
                                 plot_scatter=False, fig_path=None):
    """
    Perform randomized search cross-validation using the updated
    get_oos_predictions() interface.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_r2 = float('-inf')
    best_params_dict = {}
    results = []

    for i in range(n_iter):
        random_combo = [random.choice(v) for v in values]
        combo_params = dict(zip(keys, random_combo))
        print(f"\n=== Random Trial {i+1}/{n_iter}: {combo_params} ===")

        # --- Clone and update base parameters ---
        current_params = deepcopy(base_params)
        for key, val in combo_params.items():
            setattr(current_params, key, val)

        model_instance = deepcopy(model)

        # --- Run OOS predictions and collect results ---
        _, _, _, _, r2 = get_oos_predictions(model_instance, X, current_params, split_fn)

        results.append((*random_combo, r2))

        if math.isnan(r2):
            print("Warning: OOS R² is NaN! Skipping update.")
            continue

        if r2 > best_r2:
            best_r2 = r2
            best_params_dict = combo_params

    print(f"\n>> Best Random Search Params: {best_params_dict}")
    print(f">> Best Avg OOS R²: {best_r2:.4f}")

    # --- Store best parameters back into a params object ---
    best_params_obj = deepcopy(base_params)
    for key, val in best_params_dict.items():
        setattr(best_params_obj, key, val)

    # --- Optional: Plot random search results ---
    if plot_scatter:
        plot_mse_scatter(results=results, keys=keys, param_grid=param_grid, fig_path=fig_path)

    return best_params_obj, best_r2


# ======================================
# SECTION: Bayesian Search (Optuna)
# ======================================
def bayesian_search_cross_validate(model, X, base_params, param_grid,
                                   split_fn=forward_split, n_iter=30,
                                   plot_scatter=False, fig_path=None):
    """
    Perform Bayesian optimization-based cross-validation (Optuna)
    using the updated get_oos_predictions() interface.
    """
    if plot_scatter:
        warnings.warn("Plotting is not supported for Bayesian search and will be skipped.")

    def objective(trial):
        # --- Clone and sample parameters from search space ---
        current_params = deepcopy(base_params)
        for key, values in param_grid.items():
            if isinstance(values[0], int):
                val = trial.suggest_int(key, min(values), max(values))
            elif isinstance(values[0], float):
                val = trial.suggest_float(key, min(values), max(values))
            elif isinstance(values[0], str):
                val = trial.suggest_categorical(key, values)
            else:
                raise ValueError(f"Unsupported type for parameter {key}")
            setattr(current_params, key, val)

        model_instance = deepcopy(model)

        # --- Run OOS predictions and collect results ---
        _, _, _, _, r2 = get_oos_predictions(model_instance, X, current_params, split_fn)
        return r2

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_iter)

    print(f"\n>> Best Bayesian Params: {study.best_params}")
    print(f">> Best Avg OOS R²: {study.best_value:.4f}")

    # --- Store best parameters back into a params object ---
    best_params_obj = deepcopy(base_params)
    for key, val in study.best_params.items():
        setattr(best_params_obj, key, val)

    return best_params_obj, study.best_value
