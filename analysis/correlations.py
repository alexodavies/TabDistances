#!/usr/bin/env python3
"""
correlations.py — Inter-column correlation structure comparison across table sources.

Pipeline:
  1. For each table, select only numeric columns and compute the pairwise Pearson
     correlation matrix.  Upper-triangle values (excl. diagonal) give the full set
     of inter-column relationships for that table.
  2. Bin those values into a normalised histogram (PMF) over [-1, 1] with N_BINS
     bins — a compact "correlation fingerprint" per table.
  3. PCA over all per-table histograms, coloured by source — reveals structural
     clusters in correlation-distribution space.
  4. Two pairwise distance metrics between sources:
       • Wasserstein-1  on pooled raw correlation values
       • Mean Jensen-Shannon divergence between per-table histogram PMFs
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.patches import Ellipse
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from umap import UMAP
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

SOURCES = {
    # "CleanTabLib": Path("tables/parquet"),
    "TabICL": Path("tables-tabicl/parquet"),
    "TabFM":  Path("tables-tabfm/parquet"),
    "T4":     Path("tables-t4/parquet"),
}
COLORS = {
    "CleanTabLib": "#4C72B0",
    "TabICL":      "#DD8452",
    "TabFM":       "#55A868",
    "T4":          "#C44E52",
}

N_BINS            = 50       # histogram bins over [-1, 1]
MAX_TABLES        = 100_000     # cap per source; None = use all
MAX_ROWS          = 100_000  # rows sampled per table
MIN_NUMERIC_COLS  = 2        # skip tables with fewer numeric columns

BIN_EDGES   = np.linspace(-1.0, 1.0, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTH   = 2.0 / N_BINS


# ── Per-table histogram ────────────────────────────────────────────────────────

def corr_pmf(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Select numeric columns only, compute Pearson correlation matrix, extract
    upper-triangle values, and return (pmf, raw_vals).

    pmf      — (N_BINS,) float32, sums to 1
    raw_vals — 1-D array of correlation values before binning
    Returns (None, None) if fewer than MIN_NUMERIC_COLS numeric columns remain.
    """
    numeric = df.select_dtypes(include="number")
    # Drop low-cardinality numeric columns (unique ratio < 0.2 → effectively categorical)
    numeric = numeric.loc[:, (numeric.nunique() / max(len(numeric), 1)) >= 0.2]
    if numeric.shape[1] < MIN_NUMERIC_COLS:
        return None, None

    corr  = numeric.corr(method="pearson").to_numpy()
    n     = corr.shape[0]
    idx   = np.triu_indices(n, k=1)
    vals  = corr[idx]
    vals  = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None, None

    hist = np.histogram(vals, bins=BIN_EDGES)[0].astype(np.float32)
    s    = hist.sum()
    if s > 0:
        hist /= s   # normalise to PMF

    return hist, vals


# ── Data loading ───────────────────────────────────────────────────────────────

def load_source(
    parquet_dir: Path,
    source_name: str,
    max_tables: int | None,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
      histograms : (N_tables, N_BINS) float32 — per-table correlation PMFs
      pooled     : 1-D float64            — all raw upper-triangle correlations
    """
    files = sorted(parquet_dir.glob("*.parquet"))
    if max_tables and len(files) > max_tables:
        rng   = np.random.default_rng(42)
        files = rng.choice(files, size=max_tables, replace=False).tolist()

    hists, pooled = [], []
    for f in tqdm(files, desc=f"  {source_name}", unit="tbl"):
        df = pd.read_parquet(f)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)

        h, vals = corr_pmf(df)
        if h is None:
            continue
        hists.append(h)
        pooled.append(vals)

    histograms = np.stack(hists)       if hists  else np.zeros((0, N_BINS), dtype=np.float32)
    pooled_arr = np.concatenate(pooled) if pooled else np.array([])
    return histograms, pooled_arr


# ── Distance metrics ───────────────────────────────────────────────────────────

def wasserstein_pooled(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return wasserstein_distance(a, b)


def mean_js_divergence(hists_a: np.ndarray, hists_b: np.ndarray, n_sample: int = 300) -> float:
    """
    Mean pairwise Jensen-Shannon divergence between per-table histograms.
    Samples up to n_sample tables from each source to keep runtime bounded.
    scipy.spatial.distance.jensenshannon returns the JS *distance* (sqrt of JS divergence);
    we square it to get JS divergence in [0, log(2)].
    """
    if len(hists_a) == 0 or len(hists_b) == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    ia  = rng.choice(len(hists_a), min(n_sample, len(hists_a)), replace=False)
    ib  = rng.choice(len(hists_b), min(n_sample, len(hists_b)), replace=False)
    dists = [jensenshannon(hists_a[i], hists_b[j]) ** 2 for i in ia for j in ib]
    return float(np.nanmean(dists))


# ── Plots ──────────────────────────────────────────────────────────────────────

def _render_embedding(
    proj: np.ndarray,
    labels_arr: np.ndarray,
    sources: list[str],
    ax_sc: plt.Axes,
    ax_el: plt.Axes,
    xlabel: str,
    ylabel: str,
) -> None:
    """Shared renderer: scatter on ax_sc, covariance ellipses on ax_el."""
    for src in sources:
        mask = labels_arr == src
        ax_sc.scatter(proj[mask, 0], proj[mask, 1],
                      s=8, alpha=0.30, color=COLORS[src], label=src, linewidths=0)
    ax_sc.set_xlabel(xlabel); ax_sc.set_ylabel(ylabel)
    ax_sc.set_title("Scatter")
    ax_sc.legend(fontsize=8, markerscale=2)

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


def plot_pca(
    all_hists: np.ndarray,
    all_labels: list[str],
    output_path: str = "correlations_pca.png",
) -> None:
    sources    = list(dict.fromkeys(all_labels))
    pca        = PCA(n_components=2, random_state=42)
    proj       = pca.fit_transform(all_hists)
    labels_arr = np.array(all_labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Correlation Histogram PCA — Per-table Projection",
                 fontsize=13, fontweight="bold")
    _render_embedding(
        proj, labels_arr, sources, axes[0], axes[1],
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved PCA       → {output_path}")


def plot_umap(
    all_hists: np.ndarray,
    all_labels: list[str],
    output_path: str = "correlations_umap.png",
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
        f"Correlation Histogram UMAP — Per-table Projection "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist})",
        fontsize=13, fontweight="bold",
    )
    _render_embedding(
        proj, labels_arr, sources, axes[0], axes[1],
        xlabel="UMAP-1", ylabel="UMAP-2",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved UMAP      → {output_path}")


def plot_mean_histograms(
    source_hists: dict[str, np.ndarray],
    output_path: str = "correlations_mean.png",
) -> None:
    """Mean ± 1 std correlation PMF per source, all overlaid on one axis."""
    sources = [s for s in source_hists if len(source_hists[s]) > 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Mean Correlation Distribution per Source", fontsize=12, fontweight="bold")

    for src in sources:
        hists = source_hists[src]
        mean  = hists.mean(0)
        std   = hists.std(0)
        ax.plot(BIN_CENTERS, mean, color=COLORS[src], label=src, linewidth=1.8)
        ax.fill_between(BIN_CENTERS, mean - std, mean + std, color=COLORS[src], alpha=0.18)

    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Proportion of column pairs")
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved mean hist → {output_path}")


def plot_distance_matrix(
    sources: list[str],
    w_matrix: np.ndarray,
    js_matrix: np.ndarray,
    output_path: str = "correlations_distances.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        "Pairwise Distances — Correlation Distributions",
        fontsize=12, fontweight="bold",
    )

    for ax, mat, title in zip(
        axes,
        [w_matrix, js_matrix],
        ["Wasserstein-1 (pooled correlations)", "Mean JS Divergence (per-table PMFs)"],
    ):
        finite = mat[np.isfinite(mat)]
        vmax   = finite.max() if len(finite) else 1.0
        im     = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(sources))); ax.set_xticklabels(sources, rotation=30, ha="right")
        ax.set_yticks(range(len(sources))); ax.set_yticklabels(sources)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(len(sources)):
            for j in range(len(sources)):
                v    = mat[i, j]
                text = f"{v:.4f}" if np.isfinite(v) else "—"
                col  = "white" if (np.isfinite(v) and v > vmax * 0.6) else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=col)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved distances → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    run = wandb.init(
        entity  = "hierarchical-diffusion",
        project = "TabFMs-Characterise",
        name    = "correlations-" + "-vs-".join(SOURCES.keys()),
        config  = {
            "max_tables": MAX_TABLES,
            "max_rows":   MAX_ROWS,
            "n_bins":     N_BINS,
            "sources":    list(SOURCES.keys()),
        },
    )

    source_hists:  dict[str, np.ndarray] = {}
    source_pooled: dict[str, np.ndarray] = {}

    for source_name, parquet_dir in SOURCES.items():
        if not parquet_dir.exists():
            print(f"[WARN] {parquet_dir} not found — skipping {source_name}")
            continue
        print(f"\nLoading {source_name} …")
        hists, pooled = load_source(parquet_dir, source_name, MAX_TABLES, MAX_ROWS)
        print(f"  → {len(hists)} tables with ≥{MIN_NUMERIC_COLS} numeric columns")
        source_hists[source_name]  = hists
        source_pooled[source_name] = pooled

    if not source_hists:
        print("No data loaded. Exiting.")
        run.finish()
        return

    sources = list(source_hists.keys())
    n       = len(sources)

    # ── Distance matrices ─────────────────────────────────────────────────────
    w_mat  = np.zeros((n, n))
    js_mat = np.zeros((n, n))
    for i, sa in enumerate(sources):
        for j, sb in enumerate(sources):
            w_mat[i, j]  = wasserstein_pooled(source_pooled[sa], source_pooled[sb])
            js_mat[i, j] = mean_js_divergence(source_hists[sa], source_hists[sb])

    print("\n── Wasserstein-1 (pooled raw correlations) ──")
    print(pd.DataFrame(w_mat, index=sources, columns=sources).round(4).to_string())
    print("\n── Mean JS Divergence (per-table PMFs) ──")
    print(pd.DataFrame(js_mat, index=sources, columns=sources).round(4).to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    pca_path  = "correlations_pca.png"
    umap_path = "correlations_umap.png"
    mean_path = "correlations_mean.png"
    dist_path = "correlations_distances.png"

    all_hists  = np.concatenate([source_hists[s] for s in sources])
    all_labels = [s for s in sources for _ in range(len(source_hists[s]))]

    plot_pca(all_hists, all_labels, output_path=pca_path)
    plot_umap(all_hists, all_labels, output_path=umap_path)
    plot_mean_histograms(source_hists, output_path=mean_path)
    plot_distance_matrix(sources, w_mat, js_mat, output_path=dist_path)

    # ── wandb logging ─────────────────────────────────────────────────────────
    run.log({
        "correlations/pca":       wandb.Image(pca_path),
        "correlations/umap":      wandb.Image(umap_path),
        "correlations/mean_hist": wandb.Image(mean_path),
        "correlations/distances": wandb.Image(dist_path),
        "correlations/wasserstein_table": wandb.Table(
            data=[[sources[i], sources[j], float(w_mat[i, j])]
                  for i in range(n) for j in range(n)],
            columns=["source_a", "source_b", "wasserstein"],
        ),
        "correlations/js_table": wandb.Table(
            data=[[sources[i], sources[j], float(js_mat[i, j])]
                  for i in range(n) for j in range(n)],
            columns=["source_a", "source_b", "js_divergence"],
        ),
    })
    run.finish()


if __name__ == "__main__":
    main()
