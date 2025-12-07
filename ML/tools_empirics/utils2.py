import numpy as np

# ============================================================
# Miscellanea of utilities
# ============================================================

def nearest_key(d, key):
    """Find nearest key τ nearer in a dictionary of results."""
    keys = np.array(list(d.keys()))
    return keys[np.argmin(np.abs(keys - float(key)))]


def names_to_idx(names, cols):
    not_found = [n for n in names if n not in cols]
    if not_found:
        print("Warning: not found:", not_found)
    return [cols.index(n) for n in names if n in cols]


# ============================================================
# Confidence band
# ============================================================

def local_confidence_band(lam_tv, level=0.90):
    sigma = np.std(lam_tv) * 0.3
    z = 1.65 if level == 0.90 else 1.96
    lower = lam_tv - z * sigma
    upper = lam_tv + z * sigma
    return lower, upper


# ============================================================
# FDR – Benjamini–Hochberg
# ============================================================

def fdr_bh(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n+1) / n)
    ok = np.where(ranked <= thresh)[0]
    k = ok.max()+1 if ok.size else 0
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(n-1, -1, -1):
        q[i] = min(prev, ranked[i] * n / (i+1))
        prev = q[i]
    q_full = np.empty_like(q)
    q_full[order] = q
    return k, (order[:k] if k>0 else np.array([],dtype=int)), q_full
