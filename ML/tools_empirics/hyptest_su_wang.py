import numpy as np
from numpy.linalg import svd, eigh, norm
from .kernels import make_X_r, epanechnikov, h_rule_of_thumb
from .plotting2 import save_and_close
from config import FIG_DIR
import matplotlib.pyplot as plt

# =======================
# Utility
# =======================
def sqrt_matrix(Sigma):
    eigvals, eigvecs = eigh(Sigma)          # Sigma simmetrica
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

def compute_sigma_0(res, iT):
    # fedele a: crossprod(res)/iT con res = scale(X)
    return (res.T @ res) / iT

def compute_M_hat(F_loc, F_global, L_loc, B_global, iT, ip, m):
    num = norm((F_loc.T @ F_global) / iT, "fro")**2
    return num / (m * ip)

def compute_B_pT(F_loc, F_global, res, h, iT, ip):
    return 0.0

def compute_V_pT(F_loc, res, h, iT, ip):
    return np.var(res) + 1e-12

def residuals_local_weighted(X, F_loc, L_loc, u, tau, h, kernel_func=epanechnikov):
    w = (1 / h) * kernel_func((u - tau) / h)
    w_sum = np.sum(w)
    mu_tau = np.sum(w[:, None] * X, axis=0) / w_sum
    X_hat_r = F_loc @ L_loc.T
    X_hat = X_hat_r + mu_tau
    return X - X_hat

# =======================
# principle test (replicates the R package TVMVP)
# =======================
def su_wang_hyptest(X, m, h=None, kernel_func=epanechnikov, iB=500, seed=123):
    rng = np.random.default_rng(seed)
    iT, ip = X.shape
    u = np.linspace(1, iT, iT) / iT
    if h is None:
        h = h_rule_of_thumb(iT, ip)

    # ---- Step 1: Global PCA su scale(X)
    X_std = (X - X.mean(0)) / X.std(0, ddof=1)
    U, s, Vt = svd(X_std, full_matrices=False)
    F_global = np.sqrt(iT) * U[:, :m]
    # B_global <- t((1/iT) t(F_global) %*% returns)
    B_global = (X_std.T @ F_global) / iT

    # ---- Step 2: τ-grid e J_tau
    tau_grid = np.arange(h, 1 - h + 1e-12, h / 16)
    J_tau = np.empty(len(tau_grid))
    for j, tau in enumerate(tau_grid):
        X_r, w = make_X_r(X_std, u, tau, h)
        U_r, s_r, Vt_r = svd(X_r, full_matrices=False)
        L_loc = Vt_r.T[:, :m]
        F_loc = X_r @ L_loc
        res = residuals_local_weighted(X_std, F_loc, L_loc, u, tau, h, kernel_func)
        M = compute_M_hat(F_loc, F_global, L_loc, B_global, iT, ip, m)
        Bcorr = compute_B_pT(F_loc, F_global, res, h, iT, ip)
        V = compute_V_pT(F_loc, res, h, iT, ip)
        J_tau[j] = (iT * np.sqrt(ip) * np.sqrt(h) * M - Bcorr) / np.sqrt(V)

    J_sup = np.max(J_tau)
    print(f"sup_tau J_pT = {J_sup:.6f}")

    # ---- Step 3: Bootstrap (fedele a R)
    # Sigma_0 <- compute_sigma_0(scale(X), iT, ip)
    Sigma_0 = compute_sigma_0(X_std, iT)
    sqrt_S0 = sqrt_matrix(Sigma_0)
    J_boot = np.empty(iB)

    for b in range(iB):
        zeta_star = rng.standard_normal((iT, ip))
        # e_star <- t( sqrt_S0 %*% t(zeta_star) )
        e_star = (sqrt_S0 @ zeta_star.T).T
        X_star = F_global @ B_global.T + e_star
        U_s, s_s, Vt_s = svd(X_star, full_matrices=False)
        F_global_s = np.sqrt(iT) * U_s[:, :m]
        # B_global_star <- t((1/iT) t(F_global_star) %*% X_star)
        B_global_s = (X_star.T @ F_global_s) / iT

        J_star_tau = np.empty(len(tau_grid))
        for j, tau in enumerate(tau_grid):
            X_r_star, w = make_X_r(X_star, u, tau, h)
            U_rs, s_rs, Vt_rs = svd(X_r_star, full_matrices=False)
            L_s = Vt_rs.T[:, :m]
            F_s = X_r_star @ L_s
            res_star = residuals_local_weighted(X_star, F_s, L_s, u, tau, h, kernel_func)
            M = compute_M_hat(F_s, F_global_s, L_s, B_global_s, iT, ip, m)
            Bcorr = compute_B_pT(F_s, F_global_s, res_star, h, iT, ip)
            V = compute_V_pT(F_s, res_star, h, iT, ip)
            J_star_tau[j] = (iT * np.sqrt(ip) * np.sqrt(h) * M - Bcorr) / np.sqrt(V)

        J_boot[b] = np.max(J_star_tau)
        if (b + 1) % max(1, iB // 10) == 0:
            print(f"Bootstrap {b+1}/{iB}...")

    # ---- Step 4: p-value
    p_value = np.mean(J_boot >= J_sup)
    print(f"sup_tau J_pT = {J_sup:.4f} | p-value = {p_value:.4f}")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(tau_grid, J_tau, lw=1.5, color="black")
    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel(r"$J(\tau)$")
    ax1.set_title("Local statistic J(τ)")
    ax1.grid(True, alpha=0.3)
    save_and_close(fig1, "hyptest_Jtau", FIG_DIR)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(J_boot, bins=30, color="lightgray", edgecolor="black")
    ax2.axvline(J_sup, color="red", lw=1.5, label="Observed J_sup")
    ax2.legend(frameon=False)
    ax2.set_xlabel("J*")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Bootstrap distribution of J_sup under H₀")
    save_and_close(fig2, "hyptest_bootstrap_Jsup", FIG_DIR)

    return {
        "J_sup": J_sup,
        "p_value": p_value,
        "J_tau": J_tau,
        "bootstrap": J_boot,
        "tau_grid": tau_grid
    }
