import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ====== utility locale ======
def rhat_by_strategy(tau, strategy, results_ABC, results_IC, R_fixed, nearest_key):
    if strategy == "ABC":
        return int(results_ABC[nearest_key(results_ABC, tau)]["rhat1"])
    elif strategy == "IC":
        return int(results_IC[nearest_key(results_IC, tau)]["rhat"])
    else:
        return int(R_fixed)

# ============================================================
# OOS one-sided
# ============================================================

def make_one_sided_train_and_forward_test(X, u, tau, h, h_test, kernel_weights):
    w_full = kernel_weights(u, tau, h)

    mask_train = (u >= max(0, tau - h)) & (u <= tau)
    idx_train = np.where(mask_train)[0]
    if len(idx_train) == 0:
        return None

    u_max = min(tau + h_test, 1.0 - 1e-12)
    mask_test = (u > tau) & (u <= u_max)
    idx_test = np.where(mask_test)[0]
    if len(idx_test) == 0:
        return None

    w_train = w_full[idx_train]
    w_train = np.maximum(w_train, 0)
    s = w_train.sum()
    if s <= 0:
        return None
    w_train /= s

    mu_train = (w_train[:, None] * X[idx_train, :]).sum(axis=0, keepdims=True)
    Xc_train = X[idx_train, :] - mu_train
    X_train_w = (np.sqrt(w_train)[:, None]) * Xc_train

    X_test_c = X[idx_test, :] - mu_train

    return X_train_w, Xc_train, w_train, mu_train, idx_train, X_test_c, idx_test


def run_oos_onesided(X, u, tau_grid, h, h_test, strategy,
                     results_ABC=None, results_IC=None, R_fixed=None,
                     min_test_obs=4,
                     kernel_weights=None,
                     nearest_key=None):

    R2_oos, MSE_oos = {}, {}

    for tau in sorted(t for t in tau_grid if (t >= h) and (t <= 1 - 1.5*h)):
        rhat = rhat_by_strategy(tau, strategy, results_ABC, results_IC, R_fixed, nearest_key)
        if rhat <= 0:
            continue

        pack = make_one_sided_train_and_forward_test(X, u, float(tau), h, h_test, kernel_weights)
        if pack is None:
            continue
        X_train_w, Xc_train, w_train, mu_train, idx_train, X_test_c, idx_test = pack
        T_tr, N = X_train_w.shape
        T_te = X_test_c.shape[0]
        if T_tr <= rhat or T_te < max(min_test_obs, rhat + 1):
            continue

        # PCA weighted on the train
        S_T = X_train_w @ X_train_w.T
        eigvals, eigvecs = np.linalg.eigh(S_T)
        idx = np.argsort(eigvals)[::-1][:rhat]
        V = eigvecs[:, idx]
        F_hat = np.sqrt(T_tr) * V

        W_half = np.sqrt(w_train)[:, None]
        Lambda_hat = (Xc_train.T @ (W_half * F_hat)) / w_train.sum()

        # projection OLS on the test
        G = np.linalg.pinv(Lambda_hat.T @ Lambda_hat)
        F_test = X_test_c @ Lambda_hat @ G
        X_pred = F_test @ Lambda_hat.T

        errors = X_test_c - X_pred
        SSE = np.linalg.norm(errors)**2
        SST = np.linalg.norm(X_test_c)**2
        MSE = SSE / (T_te * N)
        if SST > 0:
            R2_oos[float(f"{tau:.6f}")] = 1.0 - SSE / SST
            MSE_oos[float(f"{tau:.6f}")] = MSE

    return R2_oos, MSE_oos


def compare_oos_strategies(X, u, tau_grid, h, h_test,
                           contender, baseline, R_fixed,
                           results_ABC, results_IC,
                           kernel_weights, nearest_key):
    R2_cont, MSE_cont = run_oos_onesided(
        X, u, tau_grid, h, h_test, contender,
        results_ABC=results_ABC, results_IC=results_IC, R_fixed=R_fixed,
        kernel_weights=kernel_weights, nearest_key=nearest_key
    )
    R2_base, MSE_base = run_oos_onesided(
        X, u, tau_grid, h, h_test, baseline,
        results_ABC=results_ABC, results_IC=results_IC, R_fixed=R_fixed,
        kernel_weights=kernel_weights, nearest_key=nearest_key
    )

    taus_common = sorted(set(R2_cont.keys()).intersection(R2_base.keys()))
    if len(taus_common) == 0:
        print("Nessuna finestra OOS in comune tra le due strategie.")
        return None

    r2_c = np.array([R2_cont[t] for t in taus_common])
    r2_b = np.array([R2_base[t] for t in taus_common])
    mse_c = np.array([MSE_cont[t] for t in taus_common])
    mse_b = np.array([MSE_base[t] for t in taus_common])

    dR2  = r2_c - r2_b
    dMSE = mse_b - mse_c

    print(f"[{contender} vs {baseline}] su {len(taus_common)} τ in comune")
    print(f"R²:  meanΔ={np.nanmean(dR2):.3f}, medianΔ={np.nanmedian(dR2):.3f}, share(Δ>0)={np.mean(dR2>0):.2%}")
    print(f"MSE: meanΔ={np.nanmean(dMSE):.6f}, medianΔ={np.nanmedian(dMSE):.6f}, share(Δ>0)={np.mean(dMSE>0):.2%}")

    t_r2, p_r2 = stats.ttest_rel(r2_c, r2_b, nan_policy='omit')
    t_mse, p_mse = stats.ttest_rel(mse_c, mse_b, nan_policy='omit')
    print(f"Paired t-test ΔR²:  t={t_r2:.2f}, p={p_r2:.4f}")
    print(f"Paired t-test ΔMSE: t={t_mse:.2f}, p={p_mse:.4f}")

    return taus_common, dR2, dMSE, R2_cont, R2_base, MSE_cont, MSE_base
