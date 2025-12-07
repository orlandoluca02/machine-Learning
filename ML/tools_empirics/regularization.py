# ======================================
# REGULARIZATION
# Author:      Tommaso Zipoli
# Last Mod:    21/10/2025
# ======================================

import torch

# ====================
#   LASSO
# ====================

def lasso(module):
    """
    Compute L1 norm of all parameters in the module.

    Args:
        module (nn.Module): Module whose parameters to regularize.

    Returns:
        Scalar L1 norm loss.
    """
    return sum(torch.sum(torch.abs(p)) for p in module.parameters())

# ====================
#   FACTOR ORTHOGONALITY
# ====================

def orth_loss(factors):
    """
    Computes a penalty for correlation between columns (factors) in the input matrix.
    
    Args:
        factors (torch.tensor) : T x r matrix of factors

    Returns:
        Scalar L1 norm loss.
    """
    X = (factors - factors.mean(dim=0)) / (factors.std(dim=0) + 1e-8)
    corr_matrix = torch.matmul(X.T, X) / (X.shape[0] - 1)
    off_diag = corr_matrix - torch.eye(corr_matrix.size(0), device=corr_matrix.device)
    penalty = (off_diag ** 2).sum()
    return penalty


# ====================
#   ANCHORING
# ====================

def anchor_loss(self, x, a1_index, a2_index, a3_index, a4_index, method):
    """
    Computes the anchor loss by correlating the first 4 latent factors
    with the respective anchor feature subsets.

    Args:
        x:          Input tensor of shape (T, D)
        a1_index:   List of feature indices for anchor 1
        a2_index:   List of feature indices for anchor 2
        a3_index:   List of feature indices for anchor 3
        a4_index:   List of feature indices for anchor 4
        method:     String, either "Pearson" or "cosine"

    Returns:
        Scalar loss (lower is better).
    """
    z = self.encoder(x)  # shape: (T, num_factors)

    def get_anchors(index):
        if index is None or len(index) == 0:
            return torch.zeros(x.shape[0], 0, device=x.device)
        return x[:, index]

    a1 = get_anchors(a1_index)
    a2 = get_anchors(a2_index)
    a3 = get_anchors(a3_index)
    a4 = get_anchors(a4_index)

    def safe_corr(factor, anchors, method):
        if anchors.shape[1] == 0:
            return 0.0
        if method == "Pearson":
            return correlation_loss(factor, anchors)
        if method == "cosine":
            return cosine_similarity_loss(factor, anchors)
        raise ValueError(f"Unknown correlation method: {method}")

    loss = (
        safe_corr(z[:, 0], a1, method) +
        safe_corr(z[:, 1], a2, method) +
        safe_corr(z[:, 2], a3, method) +
        safe_corr(z[:, 3], a4, method)
    )
    return loss


def correlation_loss(factor, anchors):
    """
    Compute average Pearson correlation loss between a latent factor
    and each of the anchor variables.

    Args:
        factor:  Tensor of shape (T,)
        anchors: Tensor of shape (T, N_vars)

    Returns:
        Scalar correlation loss.
    """
    z_std = (factor - factor.mean()) / (factor.std() + 1e-8)
    a_mean = anchors - anchors.mean(dim=0)
    cov = (z_std.unsqueeze(1) * a_mean).mean(dim=0)
    a_std = anchors.std(dim=0) + 1e-8
    corr = cov / a_std
    return 1 - corr.abs().mean()


def cosine_similarity_loss(factor, anchors):
    """
    Compute average cosine similarity loss between a latent factor
    and each anchor variable.

    Args:
        factor:  Tensor of shape (T,)
        anchors: Tensor of shape (T, N_vars)

    Returns:
        Scalar cosine loss (1 - mean cosine similarity).
    """
    factor_norm = factor / (factor.norm(p=2) + 1e-8)
    anchor_norms = anchors / (anchors.norm(p=2, dim=0, keepdim=True) + 1e-8)
    cos_sim = torch.matmul(factor_norm, anchor_norms)  # (N_vars,)
    return 1 - cos_sim.abs().mean()
