"""
Plotting functions for estiMINT package.

Equivalent to: plotting.R
"""

from typing import Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


def plot_obs_pred(
    obs: ArrayLike,
    pred: ArrayLike,
    title: str,
    path_png: str,
    xlab: str = "Observed",
    ylab: str = "Predicted"
) -> None:
    """
    Create observed vs predicted scatter plot.
    
    Equivalent to R's plot_obs_pred() function.
    
    Parameters
    ----------
    obs : array-like
        Observed values
    pred : array-like
        Predicted values
    title : str
        Plot title
    path_png : str
        Path to save PNG file
    xlab : str, optional
        X-axis label (default: "Observed")
    ylab : str, optional
        Y-axis label (default: "Predicted")
        
    Returns
    -------
    None
        Saves plot to file
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    
    # Plot diagonal reference line
    min_val = min(obs.min(), pred.min())
    max_val = max(obs.max(), pred.max())
    ax.plot(
        [min_val, max_val], 
        [min_val, max_val], 
        linestyle="--", 
        color="gray", 
        linewidth=1,
        zorder=1
    )
    
    # Scatter plot with jitter and alpha
    # Add small jitter to y-values (matching R's position_jitter)
    jitter = np.random.uniform(-0.0001, 0.0001, size=len(pred))
    ax.scatter(
        obs, 
        pred + jitter, 
        alpha=0.35, 
        s=1.1 ** 2 * 20,  # Convert point size
        edgecolors="none",
        zorder=2
    )
    
    # Set equal aspect ratio
    ax.set_aspect("equal", adjustable="box")
    
    # Labels and title
    ax.set_xlabel(xlab, fontsize=13)
    ax.set_ylabel(ylab, fontsize=13)
    ax.set_title(title, fontsize=13)
    
    # Style (matching theme_bw)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    
    # Set axis limits to be equal
    ax.set_xlim(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val))
    ax.set_ylim(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val))
    
    # Ensure output directory exists
    Path(path_png).parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(path_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
