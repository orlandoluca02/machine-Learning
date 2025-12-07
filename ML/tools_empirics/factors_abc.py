import numpy as np
import matplotlib.pyplot as plt
from .kernels import make_X_r
from config import FIG_DIR
from .plotting2 import save_and_close


# ============================================================
# ABC Criterion (Alessi–Barigozzi–Capasso, 2010)
# ============================================================

def ABC_crit(X, kmax=10, nbck=None, cmax=3.0, graph=True, step=500, T_eff=None, rng=None):
    X = np.asarray(X, dtype=float)
    T, n = X.shape
    if rng is None:
        rng = np.random.default_rng(12345)
    if nbck is None:
        nbck = int(np.floor(n / 10)) if n >= 10 else max(1, n - 1)
    if T_eff is None:
        T_eff = T

    c_count = int(np.floor(cmax * step))
    c_grid = np.arange(1, c_count + 1) / float(step)
    N_values = list(range(n - nbck, n + 1))
    abc = np.zeros((len(N_values), c_count), dtype=int)

    for s, N_sub in enumerate(N_values):
        idx = rng.permutation(n)[:N_sub]
        xs = X[:, idx]
        covm = np.cov(xs, rowvar=False, ddof=1)
        eigv = np.linalg.eigvalsh(covm)[::-1]

        IC1 = np.zeros(kmax + 1)
        for k in range(0, kmax + 1):
            IC1[k] = eigv[k:].sum() if k < N_sub else 0.0

        p = ((N_sub + T_eff) / (N_sub * T_eff)) * np.log((N_sub * T_eff) / (N_sub + T_eff))
        k_vec = np.arange(0, kmax + 1, dtype=float)

        for j, cc in enumerate(c_grid):
            IC = (IC1 / N_sub) + k_vec * p * cc
            abc[s, j] = int(np.argmin(IC))

    sabc = abc.std(axis=0)
    r_last = abc[-1, :]

    segments = []
    r_prev = r_last[0]
    c_start = c_grid[0]
    for cj, rj in zip(c_grid[1:], r_last[1:]):
        if rj != r_prev:
            c_end = cj
            segments.append([int(r_prev), c_start, c_end, c_end - c_start])
            r_prev, c_start = rj, cj
    segments.append([int(r_prev), c_start, c_grid[-1], c_grid[-1] - c_start])
    ABC_mat = np.array(segments) if len(segments) else np.empty((0, 4))

    def pick_r(width_thresh):
        if ABC_mat.shape[0] == 0:
            return 0
        mask = ABC_mat[:, 3] > width_thresh
        idx = np.where(mask)[0]
        if idx.size >= 2:
            return int(ABC_mat[idx[1], 0])
        elif idx.size == 1:
            return int(ABC_mat[idx[0], 0])
        return int(ABC_mat[np.argmax(ABC_mat[:, 3]), 0])

    rhat1 = pick_r(0.05)
    rhat2 = pick_r(0.01)

    if graph:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(c_grid, r_last, 'k-', lw=1.4, label=r"$\hat r_{T,c;N}$")
        ax.plot(c_grid, 5 * sabc, 'k--', lw=1.0, label=r"$5\,S_c$")
        ax.set_xlim(0, cmax)
        ax.set_xlabel("c")
        ax.set_title("ABC – r*(c) e stabilità")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        save_and_close(fig, "ABC_single_tau", FIG_DIR)


    return rhat1, rhat2, abc, c_grid


# ============================================================
# Wrapper – perform ABC_crit locally on every τ
# ============================================================

def ABC_over_tau(X, u, tau_grid, h, kmax=10, nbck=None, cmax=3.0,
                 seed=12345, plot_each_tau=True, plot_summary=True):
    rng = np.random.default_rng(seed)
    results_ABC = {}

    for tau in tau_grid:
        tau_key = float(np.round(tau, 6))
        X_r, w = make_X_r(X, u, tau, h)
        T_eff = (w.sum()**2) / np.sum(w**2)
        r1, r2, abc_mat, c_grid = ABC_crit(
            X_r, kmax=kmax, nbck=nbck, cmax=cmax,
            graph=bool(plot_each_tau), T_eff=T_eff, rng=rng
        )
        print(f"τ = {tau_key:.3f} -> rhat1 = {r1}, rhat2 = {r2}")
        results_ABC[tau_key] = {"rhat1": r1, "rhat2": r2, "T_eff": T_eff, "abc": abc_mat, "c_grid": c_grid}

    if plot_summary:
        taus = np.array(sorted(results_ABC.keys()))
        R_vec = np.array([results_ABC[t]["rhat1"] for t in taus])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.step(taus, R_vec, where="mid", color="black", lw=1.4)
        ax.set_xlabel("τ")
        ax.set_ylabel(r"$\hat R_{ABC}(\tau)$")
        ax.set_title("Number of factors estimated locally – ABC")
        ax.grid(True, alpha=0.3)
        save_and_close(fig, "ABC_over_tau", FIG_DIR)

    return results_ABC
