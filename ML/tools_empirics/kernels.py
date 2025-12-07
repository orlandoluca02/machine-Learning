import numpy as np

# ============================================================
# Bandwidth e τ-grid
# ============================================================

def h_rule_of_thumb(T, N):
    c = 2.35 / (12**0.5)
    h = c * (T ** (-1/5)) * (N ** (-1/10))
    h = max(2.0/T, min(h, 0.25))
    return h


# ============================================================
# Kernel and weight
# ============================================================

def epanechnikov(z: np.ndarray) -> np.ndarray:
    m = (np.abs(z) < 1).astype(float)
    return 0.75 * (1.0 - z**2) * m


def kernel_weights(u, tau, h, K=epanechnikov):
    z = (u - tau) / h
    return (1/h) * K(z)


# ============================================================
# Local window
# ============================================================

def make_X_r(X: np.ndarray, u: np.ndarray, tau: float, h: float):
    w = kernel_weights(u, tau, h)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("Pesi nulli: controlla h/tau.")
    mu_tau = (w[:, None] * X).sum(axis=0, keepdims=True) / w_sum
    X_loc = X - mu_tau
    W_half = np.sqrt(w)[:, None]
    X_r = W_half * X_loc
    return X_r, w
