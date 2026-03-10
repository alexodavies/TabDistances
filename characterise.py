#!/usr/bin/env python3
"""
characterise.py — Interpretable characterisation metrics across three table sources.

Computes structural and statistical metrics for CleanTabLib, TabICL, and TabFM tables,
then compares distributions across sources using summary statistics and Wasserstein-1 distances.

Structural
  - Row / column count distributions
  - Missingness rate (fraction of missing cells)
  - Missingness structure (mean pairwise correlation of per-column missingness indicators;
    high value → structured MNAR/MAR-like missingness, low → MCAR)
  - Numeric-column fraction
  - Categorical-column cardinality distribution

Statistical
  - Marginal distribution shape: mean absolute skewness and excess kurtosis per table
    (column-wise, numeric features only)
  - Mean absolute pairwise Pearson correlation across feature columns
    (measures inter-feature dependence strength)
  - Class balance: normalised entropy of target column (1 = uniform, 0 = single class)
  - Target leakage: max |Pearson r| between any numeric feature and target column
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.patches import Ellipse
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.decomposition import PCA
from tqdm import tqdm
from umap import UMAP

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

SOURCES = {
    # "CleanTabLib": Path("tables/parquet"),
    "TabICL":      Path("tables-tabicl/parquet"),
    "TabFM":       Path("tables-tabfm/parquet"),
    "T4":          Path("tables-t4/parquet"),
}
COLORS = {
    "CleanTabLib": "#4C72B0",
    "TabICL":      "#DD8452",
    "TabFM":       "#55A868",
    "T4":          "#C44E52",
}

MAX_TABLES = 100_000   # cap per source for speed; set to None to use all
MAX_ROWS   = 100000000  # max rows sampled per table

# ── Column-shape histogram config ──────────────────────────────────────────────

SHAPE_N_BINS    = 50
SHAPE_BIN_EDGES = np.linspace(-3.0, 3.0, SHAPE_N_BINS + 1)
SHAPE_BIN_CTR   = 0.5 * (SHAPE_BIN_EDGES[:-1] + SHAPE_BIN_EDGES[1:])

# Common target-column names (case-sensitive first match)
TARGET_NAMES = {"target", "label", "class", "y", "Target", "Label", "Class"}

# ── Per-table metrics ──────────────────────────────────────────────────────────

def _find_target(df: pd.DataFrame) -> str | None:
    for name in TARGET_NAMES:
        if name in df.columns:
            return name
    return None



def _normalised_entropy(series: pd.Series) -> float:
    """Entropy of value-count distribution normalised to [0, 1]."""
    counts = series.dropna().value_counts(normalize=True)
    if len(counts) < 2:
        return 0.0
    entropy    = float(-np.sum(counts * np.log(counts + 1e-12)))
    max_entropy = np.log(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_table_metrics(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    target_col     = _find_target(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [
        c for c in df.columns
        if (df[c].nunique() / max(len(df), 1)) < 0.2
    ]
    feat_cols    = [c for c in numeric_cols if c != target_col and c not in cat_cols]

    # ── Structural ──────────────────────────────────────────────────────────
    cat_fraction = len(cat_cols) / n_cols if n_cols else 0.0

    if cat_cols:
        cardinalities        = [df[c].nunique() for c in cat_cols]
        mean_cat_cardinality = float(np.mean(cardinalities))
        max_cat_cardinality  = float(np.max(cardinalities))
    else:
        mean_cat_cardinality = 0.0
        max_cat_cardinality  = 0.0

    # ── Statistical ─────────────────────────────────────────────────────────
    # Marginal distribution shape (numeric features, excluding target)
    if feat_cols:
        feat_data = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        skewnesses = feat_data.skew(skipna=True).abs()
        kurtoses   = feat_data.kurt(skipna=True).abs()
        mean_skewness = float(skewnesses.mean(skipna=True))
        mean_kurtosis = float(kurtoses.mean(skipna=True))
    else:
        mean_skewness = float("nan")
        mean_kurtosis = float("nan")

    # Mean absolute pairwise Pearson correlation (inter-feature dependence)
    if len(feat_cols) >= 2:
        corr_matrix = df[feat_cols].corr(method="pearson").abs().to_numpy().copy()
        n = corr_matrix.shape[0]
        corr_matrix[range(n), range(n)] = 1
        # np.fill_diagonal(corr_matrix, 0)
        mean_pairwise_corr = float(np.nanmean(corr_matrix))
    else:
        mean_pairwise_corr = float("nan")

    # Class balance: normalised entropy over target class distribution (1 = uniform, 0 = single class)
    class_balance = _normalised_entropy(df[target_col]) if target_col else float("nan")

    return {
        "n_rows":               n_rows,
        "n_cols":               n_cols,
        "cat_fraction":         cat_fraction,
        "mean_cat_cardinality": mean_cat_cardinality,
        "max_cat_cardinality":  max_cat_cardinality,
        "mean_skewness":        mean_skewness,
        "mean_kurtosis":        mean_kurtosis,
        "mean_pairwise_corr":   mean_pairwise_corr,
        "class_balance":        class_balance,
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def load_source_metrics(
    parquet_dir: Path,
    source_name: str,
    max_tables: int | None,
    max_rows: int,
) -> pd.DataFrame:
    files = sorted(parquet_dir.glob("*.parquet"))
    if max_tables and len(files) > max_tables:
        rng   = np.random.default_rng(42)
        files = rng.choice(files, size=max_tables, replace=False).tolist()

    records = []
    for f in tqdm(files, desc=f"  {source_name}", unit="tbl"):
        # try:
        df = pd.read_parquet(f)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
        m = compute_table_metrics(df)
        m["source"] = source_name
        records.append(m)
        # except Exception as e:
            # print(e)
            # pass

    return pd.DataFrame(records)


# ── Column-shape histograms ────────────────────────────────────────────────────

def column_shape_pmf(series: pd.Series) -> np.ndarray | None:
    """
    Z-score a numeric column, clip to [-3, 3], bin into SHAPE_N_BINS bins.
    Returns a (SHAPE_N_BINS,) float32 PMF, or None if the column is invalid.
    """
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) < 10:
        return None
    std = vals.std()
    if std < 1e-8:
        return None
    z    = (vals - vals.mean()) / std
    hist = np.histogram(z, bins=SHAPE_BIN_EDGES)[0].astype(np.float32)
    s    = hist.sum()
    if s > 0:
        hist /= s
    return hist


SHAPE_SUBSAMPLE_THRESHOLD = 5_000   # subsample 1-in-5 only above this many tables
SHAPE_SUBSAMPLE_RATE      = 5       # keep 1 in every N tables

def load_source_shape_histograms(
    parquet_dir: Path,
    source_name: str,
    max_tables: int | None,
    max_rows: int,
) -> np.ndarray:
    """
    For each table in parquet_dir, compute a column-shape PMF for every numeric
    column.  Returns a (N_total_columns, SHAPE_N_BINS) float32 array — one row
    per numeric column across all tables.

    When the source has more than SHAPE_SUBSAMPLE_THRESHOLD tables, a random
    1-in-SHAPE_SUBSAMPLE_RATE subset is drawn before processing.
    """
    files = sorted(parquet_dir.glob("*.parquet"))
    if max_tables and len(files) > max_tables:
        rng   = np.random.default_rng(42)
        files = rng.choice(files, size=max_tables, replace=False).tolist()

    if len(files) > SHAPE_SUBSAMPLE_THRESHOLD:
        n_total = len(files)
        n_keep  = max(1, n_total // SHAPE_SUBSAMPLE_RATE)
        rng     = np.random.default_rng(42)
        files   = rng.choice(files, size=n_keep, replace=False).tolist()
        print(f"  [{source_name}] large source ({n_total} tables) "
              f"— using {n_keep} randomly sampled tables for shape histograms")

    pmfs = []
    for f in tqdm(files, desc=f"  {source_name} (shapes)", unit="tbl"):
        df = pd.read_parquet(f)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
        for col in df.select_dtypes(include="number").columns:
            pmf = column_shape_pmf(df[col])
            if pmf is not None:
                pmfs.append(pmf)

    return np.stack(pmfs) if pmfs else np.zeros((0, SHAPE_N_BINS), dtype=np.float32)


# ── Metric metadata (display label, axis scale) ────────────────────────────────

METRIC_META: dict[str, tuple[str, str]] = {
    "n_rows":               ("Row count",                      "log"),
    "n_cols":               ("Column count",                   "linear"),
    "cat_fraction":         ("Categorical column fraction",    "linear"),
    "mean_cat_cardinality": ("Mean categorical cardinality",   "log"),
    "max_cat_cardinality":  ("Max categorical cardinality",    "log"),
    "mean_skewness":        ("Mean |skewness|",                "log"),
    "mean_kurtosis":        ("Mean |excess kurtosis|",         "log"),
    "mean_pairwise_corr":   ("Mean |pairwise Pearson corr|",  "linear"),
    "class_balance":        ("Class balance (norm. entropy)",  "linear"),
}


# ── Reporting ──────────────────────────────────────────────────────────────────

def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return wasserstein_distance(a, b)


def print_summary(all_metrics: pd.DataFrame) -> None:
    sources      = list(SOURCES.keys())
    metric_cols  = [m for m in METRIC_META if m in all_metrics.columns]
    pairs        = [(sources[i], sources[j])
                    for i in range(len(sources))
                    for j in range(i + 1, len(sources))]

    # ── Mean ± std ──────────────────────────────────────────────────────────
    print("\n" + "═" * 100)
    print("  MEAN ± STD PER SOURCE")
    print("═" * 100)
    rows = []
    for metric in metric_cols:
        label, _ = METRIC_META[metric]
        row = {"Metric": label}
        for src in sources:
            vals = all_metrics.loc[all_metrics["source"] == src, metric].dropna()
            row[src] = f"{vals.mean():.3f} ± {vals.std():.3f}" if len(vals) else "—"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("Metric").to_string())

    # ── Wasserstein distances ────────────────────────────────────────────────
    print("\n" + "═" * 100)
    print("  WASSERSTEIN-1 DISTANCES BETWEEN SOURCE DISTRIBUTIONS")
    print("═" * 100)
    rows = []
    for metric in metric_cols:
        label, scale = METRIC_META[metric]
        row = {"Metric": label}
        for a, b in pairs:
            va = all_metrics.loc[all_metrics["source"] == a, metric].values
            vb = all_metrics.loc[all_metrics["source"] == b, metric].values
            if scale == "log":
                va, vb = np.log10(va + 1), np.log10(vb + 1)
            row[f"{a} vs {b}"] = f"{_wasserstein(va, vb):.4f}"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("Metric").to_string())
    print()


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_distributions(
    all_metrics: pd.DataFrame,
    output_path: str = "characterisation.png",
) -> None:
    metric_cols = [m for m in METRIC_META if m in all_metrics.columns]
    n          = len(metric_cols)
    ncols      = 3
    nrows      = (n + ncols - 1) // ncols
    sources    = list(SOURCES.keys())

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    fig.suptitle(
        "Table Characterisation: Distribution by Source",
        fontsize=14, fontweight="bold", y=1.01,
    )
    axes = axes.flatten()

    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        label, scale = METRIC_META[metric]

        for src in sources:
            vals = all_metrics.loc[all_metrics["source"] == src, metric].dropna().values
            if len(vals) == 0:
                continue
            if scale == "log":
                vals = vals[vals > 0]
                if len(vals) == 0:
                    continue
                bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 41)
            else:
                bins = 40
            ax.hist(vals, bins=bins, alpha=0.5, label=src, color=COLORS[src], density=True)

        if scale == "log":
            ax.set_xscale("log")
        ax.set_title(label, fontsize=8.5)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot      → {output_path}")


# ── Pairwise scatter / density plots ───────────────────────────────────────────

SCATTER_PAIRS = [
    ("n_rows",        "n_cols",                "log",    "log"),
    ("mean_skewness", "mean_kurtosis",         "log",    "log"),
    ("cat_fraction",  "mean_cat_cardinality", "linear", "log"),
]


def plot_pairwise(
    all_metrics: pd.DataFrame,
    output_path: str = "characterisation_pairwise.png",
) -> None:
    sources = list(SOURCES.keys())
    ncols   = len(SCATTER_PAIRS)

    # Outer grid: row 0 = scatter+marginals, row 1 = KDE
    fig = plt.figure(figsize=(5 * ncols, 11))
    fig.suptitle(
        "Table Characterisation: Pairwise Scatter & KDE",
        fontsize=14, fontweight="bold", y=1.01,
    )
    outer_gs = gridspec.GridSpec(2, ncols, figure=fig, hspace=0.45, wspace=0.35)

    # Pre-compute raw data for each pair × source (no log transform — axes handle scale)
    data = {}  # (col, src) -> (xv, yv)
    for col, (xk, yk, xscale, yscale) in enumerate(SCATTER_PAIRS):
        for src in sources:
            sub = all_metrics[all_metrics["source"] == src]
            xv  = sub[xk].dropna().values
            yv  = sub[yk].reindex(sub[xk].dropna().index).values
            mask = ~np.isnan(xv) & ~np.isnan(yv)
            xv, yv = xv[mask], yv[mask]
            if xscale == "log":
                keep = xv > 0; xv, yv = xv[keep], yv[keep]
            if yscale == "log":
                keep = yv > 0; xv, yv = xv[keep], yv[keep]
            data[(col, src)] = (xv, yv)

    for col, (xk, yk, xscale, yscale) in enumerate(SCATTER_PAIRS):
        xlabel, _ = METRIC_META[xk]
        ylabel, _ = METRIC_META[yk]
        xlab = xlabel
        ylab = ylabel

        all_x = np.concatenate([data[(col, s)][0] for s in sources if len(data[(col, s)][0])])
        all_y = np.concatenate([data[(col, s)][1] for s in sources if len(data[(col, s)][1])])
        xlim  = (all_x.min(), all_x.max()) if len(all_x) else (0, 1)
        ylim  = (all_y.min(), all_y.max()) if len(all_y) else (0, 1)

        # ── Row 0: scatter with marginal histograms ──────────────────────────
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer_gs[0, col],
            width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_xh = fig.add_subplot(inner[0, 0])          # top x-marginal
        ax_sc = fig.add_subplot(inner[1, 0])           # scatter
        ax_yh = fig.add_subplot(inner[1, 1])           # right y-marginal
        fig.add_subplot(inner[0, 1]).set_visible(False) # corner (blank)

        xbins = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 31) if xscale == "log" else 30
        ybins = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), 31) if yscale == "log" else 30
        for src in sources:
            xv, yv = data[(col, src)]
            if len(xv) == 0:
                continue
            ax_sc.scatter(xv, yv, s=6, alpha=0.25, color=COLORS[src], label=src, linewidths=0)
            ax_xh.hist(xv, bins=xbins, alpha=0.5, color=COLORS[src], density=True)
            ax_yh.hist(yv, bins=ybins, alpha=0.5, color=COLORS[src], density=True,
                       orientation="horizontal")

        if xscale == "log":
            ax_sc.set_xscale("log"); ax_xh.set_xscale("log")
        if yscale == "log":
            ax_sc.set_yscale("log"); ax_yh.set_yscale("log")
        ax_sc.set_xlim(xlim); ax_sc.set_ylim(ylim)
        ax_xh.set_xlim(xlim); ax_yh.set_ylim(ylim)
        ax_xh.set_xticklabels([]); ax_xh.set_yticklabels([])
        ax_yh.set_yticklabels([]); ax_yh.set_xticklabels([])
        ax_sc.set_xlabel(xlab, fontsize=8.5)
        ax_sc.set_ylabel(ylab, fontsize=8.5)
        ax_xh.set_title(f"{xlabel} vs {ylabel}", fontsize=9)
        ax_sc.legend(fontsize=7, markerscale=2)

        # ── Row 1: KDE contours ──────────────────────────────────────────────
        ax_kde = fig.add_subplot(outer_gs[1, col])
        # KDE is computed in log space for numerical quality; grid is projected
        # back to original values so the log-scale axes display correctly.
        xg_log = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), 80) if xscale == "log" else np.linspace(xlim[0], xlim[1], 80)
        yg_log = np.linspace(np.log10(ylim[0]), np.log10(ylim[1]), 80) if yscale == "log" else np.linspace(ylim[0], ylim[1], 80)
        Xg_log, Yg_log = np.meshgrid(xg_log, yg_log)
        Xg_plot = 10 ** Xg_log if xscale == "log" else Xg_log
        Yg_plot = 10 ** Yg_log if yscale == "log" else Yg_log
        for src in sources:
            xv, yv = data[(col, src)]
            if len(xv) > 10:
                try:
                    xv_k = np.log10(xv) if xscale == "log" else xv
                    yv_k = np.log10(yv) if yscale == "log" else yv
                    kde  = gaussian_kde(np.vstack([xv_k, yv_k]))
                    Zg   = kde(np.vstack([Xg_log.ravel(), Yg_log.ravel()])).reshape(Xg_log.shape)
                    ax_kde.contour(Xg_plot, Yg_plot, Zg, levels=5, colors=[COLORS[src]],
                                   linewidths=1.0, alpha=0.85)
                except Exception:
                    pass

        if xscale == "log": ax_kde.set_xscale("log")
        if yscale == "log": ax_kde.set_yscale("log")
        ax_kde.set_xlim(xlim); ax_kde.set_ylim(ylim)
        ax_kde.set_xlabel(xlab, fontsize=8.5)
        ax_kde.set_ylabel(ylab, fontsize=8.5)
        ax_kde.set_title(f"{xlabel} vs {ylabel} (KDE)", fontsize=9)
        ax_kde.legend(
            handles=[plt.Line2D([0], [0], color=COLORS[s], lw=1.5, label=s) for s in sources],
            fontsize=7,
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved pairwise  → {output_path}")


# ── Column-shape embedding helpers ────────────────────────────────────────────

def _mean_js_divergence(hists_a: np.ndarray, hists_b: np.ndarray, n_sample: int = 300) -> float:
    if len(hists_a) == 0 or len(hists_b) == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    ia  = rng.choice(len(hists_a), min(n_sample, len(hists_a)), replace=False)
    ib  = rng.choice(len(hists_b), min(n_sample, len(hists_b)), replace=False)
    dists = [jensenshannon(hists_a[i], hists_b[j]) ** 2 for i in ia for j in ib]
    return float(np.nanmean(dists))


def _render_shape_embedding(
    proj: np.ndarray,
    labels_arr: np.ndarray,
    sources: list[str],
    ax_sc: plt.Axes,
    ax_el: plt.Axes,
    xlabel: str,
    ylabel: str,
) -> None:
    for src in sources:
        mask = labels_arr == src
        ax_sc.scatter(proj[mask, 0], proj[mask, 1],
                      s=4, alpha=0.20, color=COLORS[src], label=src, linewidths=0)
    ax_sc.set_xlabel(xlabel); ax_sc.set_ylabel(ylabel)
    ax_sc.set_title("Scatter")
    ax_sc.legend(fontsize=8, markerscale=3)

    for src in sources:
        mask = labels_arr == src
        pts  = proj[mask]
        if len(pts) < 3:
            continue
        mu   = pts.mean(0)
        cov  = np.cov(pts.T)
        evals, evecs = np.linalg.eigh(cov)
        order        = evals.argsort()[::-1]
        evals, evecs = evals[order], evecs[:, order]
        angle        = np.degrees(np.arctan2(*evecs[:, 0][::-1]))
        for n_std, lw, alpha in [(1, 1.8, 0.85), (2, 0.8, 0.40)]:
            ell = Ellipse(
                xy=mu,
                width=2 * n_std * np.sqrt(max(evals[0], 0)),
                height=2 * n_std * np.sqrt(max(evals[1], 0)),
                angle=angle,
                edgecolor=COLORS[src], facecolor="none",
                linewidth=lw, alpha=alpha,
                label=src if n_std == 1 else None,
            )
            ax_el.add_patch(ell)
        ax_el.scatter(*mu, s=40, color=COLORS[src], zorder=5)

    pad = max(abs(proj).max() * 0.15, 0.05)
    ax_el.set_xlim(proj[:, 0].min() - pad, proj[:, 0].max() + pad)
    ax_el.set_ylim(proj[:, 1].min() - pad, proj[:, 1].max() + pad)
    ax_el.set_xlabel(xlabel); ax_el.set_ylabel(ylabel)
    ax_el.set_title("Covariance Ellipses (1σ, 2σ)")
    ax_el.legend(fontsize=8)


def plot_shape_pca(
    all_hists: np.ndarray,
    all_labels: list[str],
    output_path: str = "shape_pca.png",
) -> None:
    sources    = list(dict.fromkeys(all_labels))
    pca        = PCA(n_components=2, random_state=42)
    proj       = pca.fit_transform(all_hists)
    labels_arr = np.array(all_labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Column-Shape Histogram PCA — Per-column Projection",
                 fontsize=13, fontweight="bold")
    _render_shape_embedding(
        proj, labels_arr, sources, axes[0], axes[1],
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved shape PCA  → {output_path}")


def plot_shape_umap(
    all_hists: np.ndarray,
    all_labels: list[str],
    output_path: str = "shape_umap.png",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    sources    = list(dict.fromkeys(all_labels))
    reducer    = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                      random_state=42)
    proj       = reducer.fit_transform(all_hists)
    labels_arr = np.array(all_labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Column-Shape Histogram UMAP — Per-column Projection "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist})",
        fontsize=13, fontweight="bold",
    )
    _render_shape_embedding(
        proj, labels_arr, sources, axes[0], axes[1],
        xlabel="UMAP-1", ylabel="UMAP-2",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved shape UMAP → {output_path}")


def plot_shape_mean_histograms(
    source_hists: dict[str, np.ndarray],
    output_path: str = "shape_mean.png",
) -> None:
    """Mean ± 1 std column-shape PMF per source, overlaid on one axis."""
    sources = [s for s in source_hists if len(source_hists[s]) > 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Mean Column-Shape Distribution per Source", fontsize=12, fontweight="bold")

    for src in sources:
        hists = source_hists[src]
        mean  = hists.mean(0)
        std   = hists.std(0)
        ax.plot(SHAPE_BIN_CTR, mean, color=COLORS[src], label=src, linewidth=1.8)
        ax.fill_between(SHAPE_BIN_CTR, mean - std, mean + std, color=COLORS[src], alpha=0.18)

    ax.set_xlabel("z-score (σ from column mean)")
    ax.set_ylabel("Proportion of values")
    ax.set_xlim(-3, 3)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved shape mean → {output_path}")


def plot_shape_distances(
    sources: list[str],
    js_matrix: np.ndarray,
    output_path: str = "shape_distances.png",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.suptitle("Pairwise JS Divergence — Column-Shape PMFs",
                 fontsize=12, fontweight="bold")

    finite = js_matrix[np.isfinite(js_matrix)]
    vmax   = finite.max() if len(finite) else 1.0
    im     = ax.imshow(js_matrix, cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(sources))); ax.set_xticklabels(sources, rotation=30, ha="right")
    ax.set_yticks(range(len(sources))); ax.set_yticklabels(sources)
    ax.set_title("Mean JS Divergence (per-column PMFs)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(len(sources)):
        for j in range(len(sources)):
            v    = js_matrix[i, j]
            text = f"{v:.4f}" if np.isfinite(v) else "—"
            col  = "white" if (np.isfinite(v) and v > vmax * 0.6) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=col)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved shape dist → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Characterise table sources.")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--metrics-only",  action="store_true",
                       help="Run only the per-table structural/statistical metrics.")
    group.add_argument("--shapes-only",   action="store_true",
                       help="Run only the per-column shape-histogram analysis.")
    args = parser.parse_args()

    run_metrics = not args.shapes_only
    run_shapes  = not args.metrics_only

    run = wandb.init(
        entity  = "hierarchical-diffusion",
        project = "TabFMs-Characterise",
        name    = "-vs-".join(SOURCES.keys()),
        config  = {"max_tables": MAX_TABLES, "max_rows": MAX_ROWS,
                   "sources": list(SOURCES.keys())},
    )

    log_dict: dict = {}

    # ── Per-table structural/statistical metrics ──────────────────────────────
    if run_metrics:
        dfs = []
        for source_name, parquet_dir in SOURCES.items():
            if not parquet_dir.exists():
                print(f"[WARN] {parquet_dir} not found — skipping {source_name}")
                continue
            print(f"\nLoading {source_name} …")
            df = load_source_metrics(parquet_dir, source_name, MAX_TABLES, MAX_ROWS)
            print(f"  → {len(df)} tables")
            dfs.append(df)

        if not dfs:
            print("No data loaded. Exiting.")
            run.finish()
            return

        all_metrics = pd.concat(dfs, ignore_index=True)

        print_summary(all_metrics)

        dist_path     = "characterisation.png"
        pairwise_path = "characterisation_pairwise.png"
        plot_distributions(all_metrics, output_path=dist_path)
        plot_pairwise(all_metrics, output_path=pairwise_path)

        csv_path = "characterisation_metrics.csv"
        all_metrics.to_csv(csv_path, index=False)
        print(f"  Saved raw data  → {csv_path}")

        sources     = list(SOURCES.keys())
        metric_cols = [m for m in METRIC_META if m in all_metrics.columns]
        for metric in metric_cols:
            label, _ = METRIC_META[metric]
            for src in sources:
                vals = all_metrics.loc[all_metrics["source"] == src, metric].dropna()
                if len(vals):
                    log_dict[f"{src}/{label}/mean"] = float(vals.mean())
                    log_dict[f"{src}/{label}/std"]  = float(vals.std())

        log_dict["distributions"] = wandb.Image(dist_path)
        log_dict["pairwise"]      = wandb.Image(pairwise_path)
        log_dict["metrics_table"] = wandb.Table(dataframe=all_metrics)

    # ── Column-shape histograms ───────────────────────────────────────────────
    if run_shapes:
        print("\nComputing column-shape histograms …")
        source_shape_hists: dict[str, np.ndarray] = {}
        for source_name, parquet_dir in SOURCES.items():
            if not parquet_dir.exists():
                continue
            hists = load_source_shape_histograms(parquet_dir, source_name, MAX_TABLES, MAX_ROWS)
            print(f"  → {len(hists)} column PMFs for {source_name}")
            source_shape_hists[source_name] = hists

        active_shape_sources = [s for s in source_shape_hists if len(source_shape_hists[s]) > 0]
        if active_shape_sources:
            all_shape_hists  = np.concatenate([source_shape_hists[s] for s in active_shape_sources])
            all_shape_labels = [s for s in active_shape_sources
                                for _ in range(len(source_shape_hists[s]))]

            shape_pca_path  = "shape_pca.png"
            shape_umap_path = "shape_umap.png"
            shape_mean_path = "shape_mean.png"
            shape_dist_path = "shape_distances.png"

            plot_shape_pca(all_shape_hists, all_shape_labels, output_path=shape_pca_path)
            plot_shape_umap(all_shape_hists, all_shape_labels, output_path=shape_umap_path)
            plot_shape_mean_histograms(source_shape_hists, output_path=shape_mean_path)

            n = len(active_shape_sources)
            js_mat = np.zeros((n, n))
            for i, sa in enumerate(active_shape_sources):
                for j, sb in enumerate(active_shape_sources):
                    js_mat[i, j] = _mean_js_divergence(source_shape_hists[sa], source_shape_hists[sb])
            print("\n── Mean JS Divergence (column-shape PMFs) ──")
            print(pd.DataFrame(js_mat, index=active_shape_sources,
                               columns=active_shape_sources).round(4).to_string())
            plot_shape_distances(active_shape_sources, js_mat, output_path=shape_dist_path)

            log_dict["shape_dist/pca"]       = wandb.Image(shape_pca_path)
            log_dict["shape_dist/umap"]      = wandb.Image(shape_umap_path)
            log_dict["shape_dist/mean_hist"] = wandb.Image(shape_mean_path)
            log_dict["shape_dist/distances"] = wandb.Image(shape_dist_path)

    run.log(log_dict)
    run.finish()


if __name__ == "__main__":
    main()
