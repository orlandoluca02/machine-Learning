import numpy as np
import matplotlib.pyplot as plt
from .kernels import make_X_r
from config import FIG_DIR
from .plotting2 import save_and_close

# ============================================================
# Information criterion to select the number of factors (Bai–Ng)
# ============================================================

def select_num_factors_local(X_r, h, kmax=10, penalty_type=1):
    T, N = X_r.shape
    U, s, Vt = np.linalg.svd(X_r, full_matrices=False)
    eigvals = (s**2) / T
    V = np.array([eigvals[k:].sum() / N for k in range(kmax + 1)])
    T_eff = int(T * h)
    # Compute the penalty term depending on the type
    if penalty_type == 1:
        # First penalty option: Logarithm of sum of eigenvalues and dynamic penalty
        pen = ((N + T_eff) / (N * T_eff)) * np.log((N*T_eff)/(N+T_eff))
    elif penalty_type == 2:
        # Second penalty option: Adjusted by sqrt(min(T_eff, N))
        pen = ((N + T_eff) / (N * T_eff)) * 2 * np.log(np.min([np.sqrt(T_eff), np.sqrt(N)]))
    else:
        raise ValueError("Invalid penalty type. Use 1 or 2.")


    IC = np.log(V) + np.arange(kmax + 1) * pen
    k_hat = int(np.argmin(IC))
    return k_hat, IC, V


# ============================================================
# Wrapper – perform select_num_factors_local on every τ
# ============================================================

def IC_over_tau(X, u, tau_grid, h, kmax=10, plot=True):
    results_IC = {}
    for tau in tau_grid:
        tau_key = float(np.round(tau, 6))
        X_r, w = make_X_r(X, u, tau, h)
        R_tau_IC, IC_vals, V_vals = select_num_factors_local(X_r, h, kmax)
        results_IC[tau_key] = {"rhat": R_tau_IC, "IC_vals": IC_vals, "V_vals": V_vals}
        print(f"τ={tau_key:.3f} → R̂_IC={R_tau_IC}")

    if plot:
        taus = np.array(sorted(results_IC.keys()))
        R_vec = np.array([results_IC[t]["rhat"] for t in taus])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.step(taus, R_vec, where="mid", lw=1.5, color="black")
        ax.set_xlabel("τ")
        ax.set_ylabel(r"$\hat R_{IC}(\tau)$")
        ax.set_title("Number of factors estimated locally – IC Bai–Ng")
        ax.grid(True, alpha=0.3)

        save_and_close(fig, "IC_over_tau", FIG_DIR)

    return results_IC
