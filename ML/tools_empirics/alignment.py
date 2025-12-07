import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

def procrustes_align_results(results, taus_sorted):
    R = results[taus_sorted[0]]["F_hat"].shape[1]
    F_ref = results[taus_sorted[0]]["F_hat"]
    for t in taus_sorted[1:]:
        F_curr = results[t]["F_hat"]
        Lambda_curr = results[t]["Lambda_hat"]
        R_opt, _ = orthogonal_procrustes(F_curr, F_ref)
        results[t]["F_hat"] = F_curr @ R_opt
        results[t]["Lambda_hat"] = Lambda_curr @ R_opt
        F_ref = results[t]["F_hat"]
    return results
