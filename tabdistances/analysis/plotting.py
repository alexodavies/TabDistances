"""Shared matplotlib configuration and reusable plot helpers.

``PAPER_RC`` is the single canonical definition of the rcParams dict used
across all analysis and grid scripts.  Import it once rather than defining it
in every file.

Grid-search plot helpers (``fig_auc_histogram``, ``fig_cat_param_means``,
``fig_float_param_scatter``, ``fig_parallel_coords``) are also centralised
here so ``grid/basic-grid-search.py`` and ``grid/grid-isambard.py`` share
a single implementation.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paper-style rcParams — font.size=8, 300 dpi
# ---------------------------------------------------------------------------

#: rcParams dict for publication-quality figures.
#: Apply with ``with mpl.rc_context(PAPER_RC): ...`` or call
#: :func:`apply_paper_style` to set globally.
PAPER_RC: dict = {
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.5,
    "figure.dpi":      300,
}


def apply_paper_style() -> None:
    """Apply :data:`PAPER_RC` globally (modifies ``matplotlib.rcParams``)."""
    mpl.rcParams.update(PAPER_RC)


# ---------------------------------------------------------------------------
# Shared axis helpers
# ---------------------------------------------------------------------------

def stairs_filled(
    ax: plt.Axes,
    density: np.ndarray,
    bins: np.ndarray,
    color: str,
    label: str,
    alpha: float = 0.15,
    lw: float = 1.5,
) -> None:
    """Draw a filled staircase histogram on *ax*.

    Parameters
    ----------
    ax : plt.Axes
    density : np.ndarray
        Bin heights (length = len(bins) - 1).
    bins : np.ndarray
        Bin edges (length = len(density) + 1).
    color : str
        Edge and fill colour.
    label : str
        Legend label.
    alpha : float
        Fill opacity.
    lw : float
        Edge line width.
    """
    patch = ax.stairs(density, bins, fill=True, label=label)
    patch.set_facecolor((*mpl.colors.to_rgb(color), alpha))
    patch.set_edgecolor(color)
    patch.set_linewidth(lw)


# ---------------------------------------------------------------------------
# Grid-search result plots
# (shared between grid/basic-grid-search.py and grid/grid-isambard.py)
# ---------------------------------------------------------------------------

def fig_auc_histogram(df: pd.DataFrame, metric: str = "auc") -> plt.Figure:
    """Histogram of discriminator AUC (or other scalar metric) across all trials.

    Parameters
    ----------
    df : pd.DataFrame
        Grid results with a column named *metric*.
    metric : str
        Column name to plot (default ``"auc"``).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[metric], bins=30, edgecolor="white", color="#4C72B0")
    ax.axvline(df[metric].min(),  color="red",    linestyle="--",
               label=f"best={df[metric].min():.4f}")
    ax.axvline(df[metric].mean(), color="orange", linestyle="--",
               label=f"mean={df[metric].mean():.4f}")
    ax.set_xlabel(f"{metric}  (lower = more realistic)" if "auc" in metric else metric)
    ax.set_ylabel("Count")
    ax.set_title(f"{metric} distribution across grid")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_cat_param_means(
    df: pd.DataFrame,
    cat_params: Sequence[str],
    metric: str = "auc",
) -> plt.Figure:
    """Bar chart of mean *metric* for each level of each categorical parameter.

    Parameters
    ----------
    df : pd.DataFrame
        Grid results.
    cat_params : sequence of str
        Names of categorical / boolean parameter columns.
    metric : str
        Column to average (default ``"auc"``).
    """
    n = len(cat_params)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, param in zip(axes, cat_params):
        means = df.groupby(param)[metric].mean().sort_values()
        bars  = ax.bar([str(v) for v in means.index], means.values, color="#DD8452")
        ax.set_title(param)
        ax.set_xlabel("value")
        if ax is axes[0]:
            ax.set_ylabel(f"mean {metric}")
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Mean {metric} by categorical parameter value", y=1.02)
    fig.tight_layout()
    return fig


def fig_float_param_scatter(
    df: pd.DataFrame,
    float_params: Sequence[str],
    metric: str = "auc",
) -> plt.Figure:
    """Scatter plot of *metric* vs each continuous / integer parameter.

    Parameters
    ----------
    df : pd.DataFrame
        Grid results.
    float_params : sequence of str
        Names of numeric parameter columns.
    metric : str
        Column to plot on the y-axis (default ``"auc"``).
    """
    n = len(float_params)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, param in zip(axes, float_params):
        ax.scatter(df[param], df[metric], alpha=0.3, s=10, color="#55A868")
        z  = np.polyfit(df[param].astype(float), df[metric], 1)
        xs = np.linspace(df[param].min(), df[param].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color="red", linewidth=1.5)
        ax.set_xlabel(param)
        if ax is axes[0]:
            ax.set_ylabel(metric)
        ax.set_title(param)
    fig.suptitle(f"{metric} vs continuous/integer parameters", y=1.02)
    fig.tight_layout()
    return fig


def fig_parallel_coords(
    df: pd.DataFrame,
    param_names: Sequence[str],
    metric: str = "auc",
    top_n: int = 100,
) -> plt.Figure:
    """Parallel coordinates plot for the *top_n* trials (lowest *metric*).

    Parameters
    ----------
    df : pd.DataFrame
        Grid results containing all *param_names* columns and *metric*.
    param_names : sequence of str
        Parameter columns to include as axes.
    metric : str
        Column used to rank and colour trials (default ``"auc"``).
    top_n : int
        Number of best trials to display.
    """
    top = df.nsmallest(top_n, metric).copy()

    plot_df = pd.DataFrame()
    for col in list(param_names) + [metric]:
        if col not in top.columns:
            continue
        vals = top[col]
        if vals.dtype == object or vals.dtype == bool:
            uniq = sorted(vals.unique(), key=str)
            plot_df[col] = vals.map({v: i for i, v in enumerate(uniq)})
        else:
            plot_df[col] = vals.astype(float)

    normed = plot_df.copy()
    for col in normed.columns:
        rng = normed[col].max() - normed[col].min()
        if rng > 0:
            normed[col] = (normed[col] - normed[col].min()) / rng

    cmap   = plt.cm.viridis_r
    colors = [cmap(v) for v in normed[metric]]

    fig, ax = plt.subplots(figsize=(max(12, len(normed.columns) * 1.4), 5))
    cols = normed.columns.tolist()
    for row, color in zip(normed.itertuples(index=False), colors):
        ax.plot(range(len(cols)), list(row), color=color, alpha=0.4, linewidth=0.8)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Normalised value")
    ax.set_title(
        f"Parallel coordinates – top {top_n} trials (colour = {metric}, darker = better)"
    )
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(top[metric].min(), top[metric].max()),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=metric, pad=0.01)
    fig.tight_layout()
    return fig
