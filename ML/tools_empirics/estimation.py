import numpy as np

# ============================================================
# ESTIMATION – principal block
# ============================================================

def estimate_local_dfm(X, u, tau_grid, h, results_IC, results_ABC,
                       select_method="ABC", R_fixed=None,
                       use_R_fixed_for_Lambda=True):

    results = {}
    T = X.shape[0]

    def nearest_key(d, key):
        keys = np.array(list(d.keys()))
        return keys[np.argmin(np.abs(keys - float(key)))]

    for tau in tau_grid:
        tau_key = float(np.round(tau, 6))
        from tools_empirics.kernels import make_X_r
        X_r, w = make_X_r(X, u, tau, h)

        if select_method == "ABC":
            tau_match = nearest_key(results_ABC, tau_key)
            R_tau_IC = int(results_ABC[tau_match]["rhat1"])
            IC_vals, V_vals = None, None
        elif select_method == "IC":
            tau_match = nearest_key(results_IC, tau_key)
            R_tau_IC = int(results_IC[tau_match]["rhat"])
            IC_vals, V_vals = None, None
        else:
            R_tau_IC, IC_vals, V_vals = R_fixed, None, None

        R_tau_est = R_fixed if use_R_fixed_for_Lambda else R_tau_IC

        # PCA on X_r
        S_T = X_r @ X_r.T
        eigvals, eigvecs = np.linalg.eigh(S_T)
        idx = np.argsort(eigvals)[::-1][:R_tau_est]
        V = eigvecs[:, idx]
        F_hat = np.sqrt(T) * V
        Lambda_hat = ((F_hat.T @ X_r) / T).T

        # Factors OLS on the central point
        t_center = int(round(tau * T)) - 1
        Lambda_tau = Lambda_hat
        G = Lambda_tau.T @ Lambda_tau
        F_tau = np.linalg.solve(G, Lambda_tau.T @ X[t_center, :])

        # Diagnostic
        recon_pca = np.linalg.norm(X_r - F_hat @ Lambda_hat.T)
        x_center = X[t_center, :]
        x_hat_center = F_tau @ Lambda_tau.T
        recon_ols = np.linalg.norm(x_center - x_hat_center)
        n_eff = (w.sum() ** 2) / (w @ w)
        SSE_local = np.linalg.norm(x_center - x_hat_center) ** 2
        SST_local = np.linalg.norm(x_center) ** 2
        R2_local = 1.0 - SSE_local / SST_local

        results[tau_key] = {
            "w": w,
            "n_eff": n_eff,
            "R_tau_IC": R_tau_IC,
            "R_tau_est": R_tau_est,
            "F_hat": F_hat,
            "Lambda_hat": Lambda_hat,
            "IC_vals": IC_vals if select_method == "IC" else None,
            "V_vals": V_vals if select_method == "IC" else None,
            "F_tau": F_tau,
            "recon_pca": recon_pca,
            "recon_ols": recon_ols,
            "eigvals_topR": eigvals[idx],
            "R2_local": R2_local,
            "eigvals_sum": eigvals.sum(),
        }

    # Quick diagnostic printout (prime 5 τ)
    for tau_key in list(results.keys())[:5]:
        r = results[tau_key]
        print(
            f"τ={tau_key:.3f}  "
            f"n_eff={r['n_eff']:.1f}  "
            f"||Xr - FΛ'||={r['recon_pca']:.3f}  "
            f"||x_t - FΛ'||={r['recon_ols']:.3f}  "
            f"R²_local={r['R2_local']:.3f}"
        )

    return results
