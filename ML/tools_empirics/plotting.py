# ======================================
# PLOTTING UTILITIES
# Author:      Tommaso Zipoli
# Last Mod:    11/11/2025
# ======================================


# ======================================
# SECTION: Imports
# ======================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


# === Global font settings ===
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# === Reusable custom grayscale → blue colormap ===
colors = [
    (0.3, 0.3, 0.3),  # dark gray
    (0.7, 0.7, 0.7),  # light gray (negative)
    (1.0, 1.0, 1.0),  # white at zero
    (0.6, 0.8, 1.0),  # light blue
    (0.0, 0.2, 0.8),  # dark blue
]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("gray_white_blue", colors, N=256)


# ======================================
# SECTION: Plot Loading Heatmap
# ======================================

def plot_decoder_weights(weights, fig_path, filename):
    """
    Plot decoder weights as a heatmap using the shared grayscale-to-blue colormap.

    Args:
        weights: numpy array of shape (features, latent_factors)
        fig_path: Path object for saving the plot
        filename: string, name for the exported file
    """
    sns.heatmap(
        weights,
        cmap=CUSTOM_CMAP,
        center=0,
        cbar_kws={'label': 'Weight Value'}
    )
    plt.xlabel("Latent Factors", fontname='Times New Roman')
    plt.ylabel("Original Features", fontname='Times New Roman')
    plt.title("Decoder Weights", fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig(fig_path / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()


# ======================================
# SECTION: Plot Inter-Model Correlation
# ======================================

def plot_inter_model_correlation_matrix(F1, F2, model_names, fig_path):
    """
    Plot the F1'F2 cross-correlation (dot product) matrix between two models' latent spaces.
    """
    # Compute F'F (normalized correlation-like matrix)
    # Normalize columns so that correlations are scale-independent
    F1_norm = (F1 - F1.mean(0)) / F1.std(0)
    F2_norm = (F2 - F2.mean(0)) / F2.std(0)
    corr_matrix = np.dot(F1_norm.T, F2_norm) / F1_norm.shape[0]

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=CUSTOM_CMAP,
        center=0,
        xticklabels=[f"{i}" for i in range(F2.shape[1])],
        yticklabels=[f"{i}" for i in range(F1.shape[1])],
        cbar_kws={'label': "F'F (normalized)"},
    )
    plt.title(f"Cross-Model Latent Similarity: {model_names[0]} vs {model_names[1]}", fontname='Times New Roman')
    plt.xlabel(model_names[1], fontname='Times New Roman')
    plt.ylabel(model_names[0], fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig(fig_path / "F1F2_crosscorr.png", dpi=300, bbox_inches='tight')
    plt.show()


# ======================================
# SECTION: Plot Feature-Wise R² Comparison
# ======================================

def plot_featurewise_r2_comparison(
    r2_feat_lin: np.ndarray,
    r2_feat_second: np.ndarray,
    delta_feat: np.ndarray,
    second_model_label: str,
    fig_path: Path = None,
    filename: str = None,
    show: bool = True
):
    """
    Plot feature-wise R² comparison between linear and second model,
    and the ensemble ΔR² gain below.

    Args:
        r2_feat_lin (np.ndarray): Feature-wise R² for the linear model.
        r2_feat_second (np.ndarray): Feature-wise R² for the second model.
        delta_feat (np.ndarray): Feature-wise ensemble gain ΔR².
        second_model_label (str): Label of the second model (e.g. "nonlinear").
        fig_path (Path, optional): Directory to save the plot.
        filename (str, optional): Custom filename (defaults auto-generated).
        show (bool): Whether to display the plot interactively.
    """

    # --- Prepare feature indices
    features = np.arange(len(delta_feat))

    # --- Create figure with two vertically stacked barplots
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # ======= Upper panel: R² comparison =======
    bar_width = 0.4
    axes[0].bar(features - bar_width / 2, r2_feat_lin,
                width=bar_width, color='0.3', alpha=0.9, label='Linear')
    axes[0].bar(features + bar_width / 2, r2_feat_second,
                width=bar_width, color='0.7', alpha=0.9, label=second_model_label)

    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_title(f"Feature-wise $R^2$ comparison (Linear vs {second_model_label})")
    axes[0].axhline(0, color='0.2', lw=1)
    axes[0].legend(frameon=False)
    axes[0].grid(True, linestyle=':', color='0.85')

    # ======= Lower panel: ΔR² (ensemble gain) =======
    axes[1].bar(features, delta_feat, color='0.5', width=0.6)
    axes[1].axhline(0, color='0.2', lw=1)
    axes[1].set_xlabel("Feature index")
    axes[1].set_ylabel(r"$\Delta R^2$")
    axes[1].set_title("Feature-wise Ensemble Gain")
    axes[1].grid(True, linestyle=':', color='0.85')

    # --- Layout and optional save
    plt.tight_layout()
    plt.savefig(fig_path / filename, dpi=300, bbox_inches='tight')

    if filename is None:
        filename = f"r2_comparison_and_delta_linear_{second_model_label}.png"
        plt.savefig(fig_path / filename, dpi=300, bbox_inches="tight")
    else:
        plt.savefig(fig_path / filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[Plot saved] {fig_path / filename}" if fig_path else "[Plot displayed only]")


# ======================================
# SECTION: Plot MSE Scatter
# ======================================

def plot_mse_scatter(
    results,
    keys,
    param_grid,
    fig_path
):
    if len(keys) == 2:
        plot_mse_scatter_2d(
            results=results,
            keys=keys,
            param_grid=param_grid,
            fig_path=fig_path
        )
    elif len(keys) == 3:
        plot_mse_scatter_3d(
            results=results,
            keys=keys,
            param_grid=param_grid,
            fig_path=fig_path
        )
    else:
        warnings.warn(
            f"[SKIPPED] Plotting skipped: only supports 2 or 3 parameters, got {len(keys)}.",
            category=UserWarning
        )
        return


def plot_mse_scatter_2d(
    results,
    keys,
    param_grid,
    fig_path,
    title=None
):
    param1, param2 = keys
    values1 = param_grid[param1]
    values2 = param_grid[param2]

    x_vals, y_vals, mse_vals = [], [], []

    # Collect valid result points
    for res in results:
        try:
            val1, val2, mse = res[0], res[1], res[2]
            if val1 in values1 and val2 in values2 and not np.isnan(mse):
                x_vals.append(val1)
                y_vals.append(val2)
                mse_vals.append(mse)
        except (IndexError, ValueError):
            continue

    if not mse_vals:
        print(f"[INFO] Skipping {title or f'{param1} vs {param2}'} — no valid data.")
        return

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        x_vals,
        y_vals,
        c=mse_vals,
        cmap='Blues_r',
        edgecolor='black',
        s=100,
        alpha=0.8
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Validation MSE", rotation=270, labelpad=15)

    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(title or f"Validation MSE Scatter Plot ({param1} vs {param2})")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_path / f"val_mse_scatter_{param1}_vs_{param2}.png")
    plt.show()


def plot_mse_scatter_3d(
    results,
    keys,
    param_grid,
    fig_path,
    title=None
):
    if len(keys) != 3:
        raise ValueError("This function requires exactly 3 parameters in 'keys'.")

    param1, param2, param3 = keys
    values1 = param_grid[param1]
    values2 = param_grid[param2]
    values3 = param_grid[param3]

    x_vals, y_vals, z_vals, mse_vals = [], [], [], []

    # Collect valid results
    for res in results:
        try:
            val1, val2, val3, mse = res[0], res[1], res[2], res[3]
            if (
                val1 in values1 and
                val2 in values2 and
                val3 in values3 and
                not np.isnan(mse)
            ):
                x_vals.append(val1)
                y_vals.append(val2)
                z_vals.append(val3)
                mse_vals.append(mse)
        except (IndexError, ValueError):
            continue

    if not mse_vals:
        print(f"[INFO] Skipping {title or f'{param1} vs {param2} vs {param3}'} — no valid data.")
        return

    # Normalize MSE values for color mapping
    mse_vals = np.array(mse_vals)
    norm = plt.Normalize(vmin=np.min(mse_vals), vmax=np.max(mse_vals))
    colors = plt.cm.Blues_r(norm(mse_vals))

    # Plot 3D scatter
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        c=colors,
        s=80,
        edgecolor='black',
        alpha=0.9
    )

    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel(param3)
    ax.set_title(title or f"3D Validation MSE Scatter Plot\n({param1} vs {param2} vs {param3})")

    plt.tight_layout()
    plt.savefig(fig_path / f"val_mse_scatter_{param1}_vs_{param2}_vs_{param3}.png")
    plt.show()
