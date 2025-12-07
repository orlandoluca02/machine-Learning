# ======================================
# UTILITY FUNCTIONS
# Author:      Tommaso Zipoli
# Last Mod:    15/10/2025
# ======================================


import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _log_anchor_indices(params):
    print("\nAnchor Indices:")
    for i in range(1, 5):
        index = getattr(params, f'a{i}_index', None)
        print(f"  Factor {i}: {index}")