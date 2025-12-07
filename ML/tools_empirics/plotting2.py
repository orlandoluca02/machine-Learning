import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from config import FIG_DIR
from .utils2 import local_confidence_band, nearest_key
from pathlib import Path

def save_and_close(fig, name: str, folder: Path, show=True):
    folder.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    if show:
        fig.show()
    plt.close(fig)

# ============================================================
# Initial plots
# ============================================================

def plot_standardized_series(df, X):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, X)
    ax.set_title("Standardized series (global z-scores)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Standardized value")
    save_and_close(fig, "standardized_series", FIG_DIR)


def plot_global_correlation(df, X):
    corr_X = np.corrcoef(X, rowvar=False)
    N = corr_X.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_X, vmin=-1, vmax=1, cmap='coolwarm', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', rotation=90)

    max_ticks = 20
    labels = list(map(str, df.columns))
    if N <= max_ticks:
        ax.set_xticks(np.arange(N)); ax.set_yticks(np.arange(N))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        idx = np.linspace(0, N - 1, num=max_ticks, dtype=int)
        ax.set_xticks(idx); ax.set_yticks(idx)
        ax.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=7)
        ax.set_yticklabels([labels[i] for i in idx], fontsize=7)

    ax.set_title('Global Correlation Matrix – standardized variables')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Variables')
    save_and_close(fig, "corr_matrix", FIG_DIR)


# ============================================================
# Plot on R(τ)
# ============================================================

def plot_num_factors_over_time(select_method, results_IC, results_ABC, R_fixed, df, X, tau_grid):
    taus = np.array(sorted(tau_grid))
    T = X.shape[0]
    t_idx = np.clip((taus * T).round().astype(int) - 1, 0, T - 1)
    try:
        tau_dates = df.index[t_idx]
    except Exception:
        tau_dates = taus

    if select_method == "IC":
        R_vec = np.array([int(results_IC[nearest_key(results_IC, t)]["rhat"]) for t in taus])
        title = "Number of factors estimated locally – Bai–Ng IC"
        ylabel = r"$\hat R_{IC}(\tau)$"; color = "black"
    elif select_method == "ABC":
        R_vec = np.array([int(results_ABC[nearest_key(results_ABC, t)]["rhat1"]) for t in taus])
        title = "Number of factors estimated locally – ABC"
        ylabel = r"$\hat R_{ABC}(\tau)$"; color = "darkred"
    else:
        R_vec = np.full(len(taus), int(R_fixed))
        title = "Number of factors fixed – R_fixed"
        ylabel = "R_fixed"; color = "gray"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(tau_dates, R_vec, where='mid', lw=1.6, color=color)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.datetime64):
        import pandas as pd
        events = {"Lehman 2008": "2008-09-15", "COVID-19": "2020-03-01", "Invasione Ucraina": "2022-02-24"}
        for name, date_str in events.items():
            date = pd.Timestamp(date_str)
            ax.axvline(date, color='gray', ls='--', lw=0.8)
            ax.text(date, ax.get_ylim()[1]*0.95, name, rotation=90, va='top', ha='center', fontsize=8, color='gray')

    save_and_close(fig, "R_of_tau", FIG_DIR)
    return taus, R_vec


def plot_abc_at_tau(results_ABC, tau_target, title_prefix="ABC criterion"):
    if not results_ABC:
        raise ValueError("results_ABC is empty: perform before ABC_over_tau.")

    tau_used = nearest_key(results_ABC, tau_target)
    res = results_ABC[tau_used]
    abc_mat = res["abc"]
    c_grid = res["c_grid"]
    sabc = abc_mat.std(axis=0)
    r_last = abc_mat[-1, :]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(c_grid, r_last, where="mid", color="black", lw=1.5, label=r"$\hat r_{T,c,N}$")
    ax.plot(c_grid, 5 * sabc, "r--", lw=1.2, label=r"$5S_c$")
    ax.set_xlabel("Penalty parameter c")
    ax.set_ylabel(r"$\hat r_{T,c,N}$")
    ax.set_title(f"{title_prefix} – τ={tau_used:.3f}")
    ax.grid(True, alpha=0.4)
    ax.legend(frameon=False)
    save_and_close(fig, f"ABC_curve_tau_{tau_used:.3f}", FIG_DIR)


def plot_R2_over_tau(results, window=11, passes=2, filename="R2_local_over_tau_smooth"):
    if results is None or len(results) == 0:
        print("[WARN] plot_R2_over_tau: 'results' is empty.")
        return

    taus = np.array(sorted(results.keys()))
    R2_vals = np.array([results[t]["R2_local"] for t in taus])

    R2_sm = R2_vals.copy()
    for _ in range(passes):
        R2_sm = pd.Series(R2_sm).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(taus, R2_vals, color="#c7c7c7", lw=1.1, marker="o", ms=3, label="R² local – raw")
    ax.plot(taus, R2_sm,   color="#b2182b", lw=2.2, label=f"R² local – MA x{passes} (k={window})")

    ax.set_title("Local $R^2$ – strong moving average")
    ax.set_xlabel("τ (normalized time)")
    ax.set_ylabel(r"$R^2_{local}$ (in-sample)")
    ax.grid(True, alpha=0.3)

    save_and_close(fig, filename, FIG_DIR)


# ============================================================
# Heatmap IC / ABC
# ============================================================

def plot_heatmap_ic(results, kmax, taus):
    IC_matrix = np.full((len(taus), kmax + 1), np.nan)
    for j, t in enumerate(taus):
        IC_vals = results[t].get("IC_vals", None)
        if IC_vals is not None:
            IC_matrix[j, :len(IC_vals)] = IC_vals

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(IC_matrix.T, aspect='auto', origin='lower',
                   extent=[taus.min(), taus.max(), 0, kmax], cmap='viridis')
    fig.colorbar(im, ax=ax, label='IC(k, τ)')
    ax.set_title("Heatmap – local information criterion IC(k, τ)")
    ax.set_xlabel("τ")
    ax.set_ylabel("Number of factors k")
    save_and_close(fig, "IC_heatmap", FIG_DIR)


def plot_heatmap_abc(results_ABC):
    taus = np.array(sorted(results_ABC.keys()))
    c_grid = results_ABC[taus[0]]["c_grid"]
    cmax = c_grid.max()
    ABC_mat = np.full((len(taus), len(c_grid)), np.nan)
    for j, t in enumerate(taus):
        ABC_vals = results_ABC[t]["abc"]
        if ABC_vals is not None and ABC_vals.size > 0:
            ABC_mat[j, :] = ABC_vals[-1, :]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(ABC_mat.T, aspect='auto', origin='lower',
                   extent=[taus.min(), taus.max(), c_grid.min(), cmax], cmap='magma')
    fig.colorbar(im, ax=ax, label='r*(c)')
    ax.set_title("Heatmap – criterio ABC r*(c, τ)")
    ax.set_xlabel("τ")
    ax.set_ylabel("Parametro di penalità c")
    save_and_close(fig, "ABC_heatmap", FIG_DIR)


# ============================================================
# Loadings TV vs global
# ============================================================

def plot_loadings_tv_vs_global(df, X, results, factor_idx, vars_interest_idx, years, names):
    taus = np.array(sorted(results.keys()))
    Lambda_tau_all = np.stack([results[t]['Lambda_hat'] for t in taus], axis=2)
    R = Lambda_tau_all.shape[1]

    pca_global = PCA(n_components=R)
    F_global = pca_global.fit_transform(X)
    Lambda_global = pca_global.components_.T

    for i in vars_interest_idx:
        lam_tv = Lambda_tau_all[i, factor_idx, :]
        lam_const = Lambda_global[i, factor_idx]
        lower, upper = local_confidence_band(lam_tv, level=0.90)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(years, lam_tv, color='black', lw=1.8, label='λ̂_i(τ)')
        ax.plot(years, lower, color='black', ls=':', lw=1.0, label='Lower bound (90% conf.)')
        ax.plot(years, upper, color='black', ls=':', lw=1.0, label='Upper bound (90% conf.)')
        ax.axhline(lam_const, color='black', ls='--', lw=2.0, label='Global λ̂_i')
        name = names[vars_interest_idx.index(i)]
        ax.set_title(f"Variable {name} – loading on factor {factor_idx + 1}")
        ax.set_xlabel("Year")
        ax.set_ylabel(r"$\hat{\lambda}_i(\tau)$")
        ax.grid(True, alpha=0.3)
        save_and_close(fig, f"loading_tv_vs_global_var{i}_F{factor_idx+1}", FIG_DIR)

# ============================================================
# OOS Performance and Comparison
# ============================================================

def plot_oos_comparison_summary(select_method,
                                R2_oos_onesided, MSE_oos_onesided,
                                results_ABC, results_ABC_smooth, results_IC,
                                X, u, tau_grid, h, h_test, R_fixed,
                                kernel_weights, nearest_key):

    from .oos import compare_oos_strategies

    contender = select_method
    baseline = "fixed"

    # ====== 1) Comparison between contender vs fixed ======
    out = compare_oos_strategies(
        X, u, tau_grid, h, h_test,
        contender, baseline, R_fixed,
        results_ABC, results_IC,
        kernel_weights, nearest_key,
    )

    if out is not None:
        taus_common, dR2, dMSE, R2_cont, R2_base, MSE_cont, MSE_base = out

        # ---- ΔR² plot ----
        fig, ax = plt.subplots(figsize=(8.5, 4))
        ax.plot(taus_common, dR2, color="#1b63a7", lw=2, marker='o', ms=4, mec='white', mew=0.6)
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("τ (normalized time)", fontsize=11)
        ax.set_ylabel("ΔR² (contender – fixed)", fontsize=11)
        ax.set_title(f"OOS R² Improvement – {contender} vs fixed", fontsize=12, pad=8)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        save_and_close(fig, f"OOS_R2_diff_{contender}_vs_fixed", FIG_DIR)

        # ---- ΔMSE plot ----
        fig, ax = plt.subplots(figsize=(8.5, 4))
        ax.plot(taus_common, dMSE, color="#b2182b", lw=2, marker='s', ms=4, mec='white', mew=0.6)
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("τ (normalized time)", fontsize=11)
        ax.set_ylabel("ΔMSE (fixed – contender)", fontsize=11)
        ax.set_title(f"OOS MSE Improvement – {contender} vs fixed", fontsize=12, pad=8)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        save_and_close(fig, f"OOS_MSE_diff_{contender}_vs_fixed", FIG_DIR)

    # ====== 2) Plot R² e MSE OOS ======
    if len(R2_oos_onesided) > 0:
        taus_os = np.array(sorted(R2_oos_onesided.keys()))
        R2_os_vals = np.array([R2_oos_onesided[t] for t in taus_os])
        MSE_os_vals = np.array([MSE_oos_onesided[t] for t in taus_os])

        fig, ax1 = plt.subplots(figsize=(9.5, 4.5))
        color_r2, color_mse = "#1b63a7", "#b2182b"

        ax1.plot(taus_os, R2_os_vals, color=color_r2, lw=2, marker='o', ms=4,
                 mec='white', mew=0.6, label=r"$R^2_{OOS}$")
        ax1.set_xlabel("τ (normalized time)", fontsize=11)
        ax1.set_ylabel(r"$R^2_{OOS}$", color=color_r2, fontsize=11)
        ax1.tick_params(axis='y', labelcolor=color_r2)
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(taus_os, MSE_os_vals, color=color_mse, lw=2, marker='s', ms=4,
                 mec='white', mew=0.6, label="MSE OOS")
        ax2.set_ylabel("MSE OOS", color=color_mse, fontsize=11)
        ax2.tick_params(axis='y', labelcolor=color_mse)

        ax1.set_title(f"OOS Performance (one-sided window) – {select_method}", fontsize=12, pad=8)
        fig.tight_layout()
        save_and_close(fig, f"OOS_performance_{select_method}", FIG_DIR)

        # ====== 3) print results ======
        print("=" * 70)
        print(f"OOS Performance Summary – {select_method}")
        print("-" * 70)
        print(f"Mean R² OOS  = {np.nanmean(R2_os_vals):.3f} | "
              f"min={np.nanmin(R2_os_vals):.3f} | max={np.nanmax(R2_os_vals):.3f}")
        print(f"Mean MSE OOS = {np.nanmean(MSE_os_vals):.6f} | "
              f"min={np.nanmin(MSE_os_vals):.6f} | max={np.nanmax(MSE_os_vals):.6f}")
        print("=" * 70)
    else:
        print("[WARN] No OOS results available for the selected method.")

# ============================================================
# Plot R² su τ – OUT-OF-SAMPLE (strong smoothed)
# ============================================================

def plot_R2_oos_over_tau(R2_oos_onesided, window=11, passes=2, filename="R2_oos_over_tau_smooth"):

    if not R2_oos_onesided or len(R2_oos_onesided) == 0:
        print("[WARN] plot_R2_oos_over_tau: nessun R² OOS disponibile.")
        return

    taus = np.array(sorted(R2_oos_onesided.keys()))
    R2_vals = np.array([R2_oos_onesided[t] for t in taus])

    # strong smoothing
    R2_sm = R2_vals.copy()
    for _ in range(passes):
        R2_sm = pd.Series(R2_sm).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(taus, R2_vals, color="#c7c7c7", lw=1.1, marker="o", ms=3)
    ax.plot(taus, R2_sm,   color="#b2182b", lw=2.2)

    ax.set_title("Local $R^2$ OOS – strong moving average")
    ax.set_xlabel("τ (normalized time)")
    ax.set_ylabel(r"$R^2_{OOS}$ (out-of-sample)")
    ax.grid(True, alpha=0.3)

    save_and_close(fig, filename, FIG_DIR)
