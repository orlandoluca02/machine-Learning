import numpy as np
import pandas as pd

from config import INPUT_FILE
from tools_empirics.plotting2 import (
    plot_standardized_series, plot_global_correlation,
    plot_num_factors_over_time, plot_abc_at_tau,
    plot_R2_over_tau, plot_heatmap_ic, plot_heatmap_abc,
    plot_loadings_tv_vs_global, plot_oos_comparison_summary,
    plot_R2_oos_over_tau
)
from tools_empirics.kernels import h_rule_of_thumb, make_X_r, kernel_weights
from tools_empirics.factors_ic import IC_over_tau
from tools_empirics.factors_abc import ABC_over_tau
from tools_empirics.estimation import estimate_local_dfm
from tools_empirics.hyptest_su_wang import su_wang_hyptest
from tools_empirics.alignment import procrustes_align_results
from tools_empirics.utils2 import nearest_key, names_to_idx

def main():
    # 0) Load
    df = pd.read_excel(INPUT_FILE, index_col=0, parse_dates=True)
    X = df.values.astype(float)
    T, N = X.shape

    # Plots iniziali
    plot_standardized_series(df, X)
    plot_global_correlation(df, X)

    # 1) Bandwidth e τ-grid
    h = h_rule_of_thumb(T, N)
    tau_grid = np.arange(h, 1 - h + 1e-12, h/16)
    print(f"h = {h:.4f}  |  numero di τ: {len(tau_grid)}")

    # 2) State index
    u = np.arange(1, T+1) / T

    # 3) IC locale
    results_IC = IC_over_tau(X, u, tau_grid, h, kmax=10, plot=True)

    # 4) ABC locale
    results_ABC = ABC_over_tau(
        X, u, tau_grid, h,
        kmax=10, nbck=None, cmax=3.0,
        seed=12345, plot_each_tau=False, plot_summary=True
    )

    # examples of ABC curves for specific taus
    for t in [0.25, 0.75, 0.15, 0.35, 0.50]:
        plot_abc_at_tau(results_ABC, tau_target=t)

    # mean r̂
    mean_rhat_IC = np.mean([v["rhat"] for v in results_IC.values()])
    mean_rhat_ABC = np.mean([v["rhat1"] for v in results_ABC.values()])
    print(f"Media R̂_IC = {mean_rhat_IC:.2f}, Media R̂_ABC = {mean_rhat_ABC:.2f}")

    mean_rhat_IC_int = int(np.round(mean_rhat_IC))
    mean_rhat_ABC_int = int(np.round(mean_rhat_ABC))

    # ESTIMATION SETTINGS
    R_fixed = mean_rhat_ABC_int
    select_method = "ABC"          # "IC", "ABC", "fixed"
    use_R_fixed_for_Lambda = True

    # 5) Complete local estimation
    results = estimate_local_dfm(
        X, u, tau_grid, h, results_IC, results_ABC,
        select_method=select_method, R_fixed=R_fixed,
        use_R_fixed_for_Lambda=use_R_fixed_for_Lambda
    )

    # 6) Graph r(tau) over time
    taus, R_vec = plot_num_factors_over_time(
        select_method=select_method,
        results_IC=results_IC,
        results_ABC=results_ABC,
        R_fixed=R_fixed,
        df=df,  # X_df originale
        X=X,
        tau_grid=tau_grid
    )

    # 7) Procrustes alignment
    taus_sorted = np.array(sorted(results.keys()))
    results = procrustes_align_results(results, taus_sorted)

    # 8) Interpretability check
    names = np.array(df.columns)
    Lambda_stack = np.stack([results[t]["Lambda_hat"] for t in taus_sorted], axis=2)
    Lambda_mean = Lambda_stack.mean(axis=2)
    Rnum = Lambda_mean.shape[1]
    for r in range(Rnum):
        load_r = Lambda_mean[:, r]
        top = np.argsort(-np.abs(load_r))[:10]
        print(f"\nFACTOR {r+1}: principali variabili (per |loading| medio)")
        for i in top:
            print(f"{names[i]:25s}  λ̄={load_r[i]:.3f}")

    # 9) Plot R² local over τ
    plot_R2_over_tau(results, window=11, passes=2)

    # 10) Loadings TV vs fixed
    cols = list(df.columns)
    F1_names = ["IPNRG_EA","SHIX_EA","REER42_EA","TRNNRG_EA","IPDCOG_EA",
                "M2_EACC","TRNCAG_EA","TRNMN_EA","IPCAG_EA","UNEU25_EA"]
    F2_names = ["ICONFIX_EA","ESENTIX_EA", "BCI_EA", "SCONFIX_EA", "IRT3M_EACC",
                "IRT6M_EACC", "UNETOT_EA", "UNEO25_EA", "PPIING_EA", "UNEU25_EA"]
    F1_idx = names_to_idx(F1_names, cols)
    F2_idx = names_to_idx(F2_names, cols)
    print("F1 idx:", F1_idx);
    print("F2 idx:", F2_idx)

    Ttot = X.shape[0]
    t_idx = np.ceil(taus * Ttot).astype(int) - 1
    t_idx = np.clip(t_idx, 0, Ttot - 1)
    years = df.index[t_idx]
    years_label = [str(y.year) for y in years]
    # plot for F1 (factor_idx=0)
    use_F1 = False
    if use_F1:
        vars_interest_idx = F1_idx
        names = F1_names
        factor_idx = 0  # for example, factor 1
    else:
        vars_interest_idx = F2_idx
        names = F2_names
        factor_idx = 1  # for example, factor 2
    plot_loadings_tv_vs_global(df, X, results, factor_idx=factor_idx, vars_interest_idx=vars_interest_idx, years=years,
                               names=names)
    # 11) Heatmap IC/ABC
    if select_method == "IC":
        plot_heatmap_ic(results, kmax=10, taus=taus_sorted)
    elif select_method == "ABC":
        plot_heatmap_abc(results_ABC)

    # 12) Stability test Su & Wang
    m = mean_rhat_ABC_int
    B_boot_hyptest = 300
    res_hyptest = su_wang_hyptest(
        X,
        m=m,
        h=h,
        iB=B_boot_hyptest
    )
    pval = res_hyptest["p_value"]
    Jsup = res_hyptest["J_sup"]
    print(f"J_sup = {Jsup:.3f} | p-value = {pval:.3f}")
    qs = np.percentile(res_hyptest["bootstrap"], [5, 50, 95])
    print(
        f"J_sup={res_hyptest['J_sup']:.3f} | p={res_hyptest['p_value']:.3f} | J* q5={qs[0]:.2f}, q50={qs[1]:.2f}, q95={qs[2]:.2f}")

    # 13) OOS one-sided
    from tools_empirics.oos import run_oos_onesided, compare_oos_strategies
    h_test = 0.3 * h
    R2_oos_onesided, MSE_oos_onesided = run_oos_onesided(
        X, u, tau_grid, h, h_test, select_method,
        results_ABC=results_ABC, results_IC=results_IC, R_fixed=R_fixed,
        kernel_weights=kernel_weights, nearest_key=nearest_key
    )

    if len(R2_oos_onesided) > 0:
        taus_os = np.array(sorted(R2_oos_onesided.keys()))
        R2_os_vals = np.array([R2_oos_onesided[t] for t in taus_os])
        MSE_os_vals = np.array([MSE_oos_onesided[t] for t in taus_os])

        fig, ax1 = plt.subplots(figsize=(9,4))
        ax1.plot(taus_os, R2_os_vals, marker='o', label='R² OOS')
        ax1.set_xlabel("τ (normalized time)")
        ax1.set_ylabel("R² OOS")

        ax2 = ax1.twinx()
        ax2.plot(taus_os, MSE_os_vals, marker='s', label='MSE OOS')
        ax2.set_ylabel("MSE OOS")

        plt.title(f"OOS performance (one-sided window) – {select_method}")
        plt.grid(True)
        plt.tight_layout()

        print(f"Mean R² OOS = {np.nanmean(R2_os_vals):.3f} | min={np.nanmin(R2_os_vals):.3f} | max={np.nanmax(R2_os_vals):.3f}")
        print(f"Median MSE OOS = {np.nanmean(MSE_os_vals):.6f} | min={np.nanmin(MSE_os_vals):.6f} | max={np.nanmax(MSE_os_vals):.6f}")
    else:
        print("No R²/MSE OOS obtained: test too small or r̂ too big with respect to the local train.")

    # 14) comparison contender vs fixed
    plot_oos_comparison_summary(
        select_method,
        R2_oos_onesided, MSE_oos_onesided,
        results_ABC, results_ABC, results_IC,
        X, u, tau_grid, h, h_test, R_fixed,
        kernel_weights, nearest_key
    )

    plot_R2_oos_over_tau(R2_oos_onesided, window=11, passes=2)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
