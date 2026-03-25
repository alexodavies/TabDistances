"""Column-level coverage analysis: TALENT vs FM vs T4 vs ICL.

Operates on individual numeric columns rather than whole tables.  The feature
for each column is the 30-bin normalised cumulative histogram produced by
get_histogram().

Workflow
--------
1. Load every numeric column from every TALENT dataset (train+val+test).
2. Build a flat (file, col_name) index of all numeric columns in FM, T4, ICL.
3. Sample exactly N_talent_cols uniformly from each index — genuinely random
   across all available columns, not just random tables.
4. Compute histograms for all four column pools.
5. Produce three outputs:
     a. Mean ± SD histogram band plot (TALENT vs FM vs T4 vs ICL).
     b. PCA of all column histograms coloured by source.
     c. Pairwise NN-distance density histograms (TALENT $\\in$ ICL, TALENT $\\in$ FM,
        TALENT $\\in$ T4) using the same compute_nn_distances approach as
        rank_vs_coverage.py.

Usage
-----
    python analysis/column_coverage.py [--talent-dir data/]
                                       [--fm-dir tables-tabfm/parquet]
                                       [--t4-dir tables-t4/parquet]
                                       [--icl-dir tables-tabicl/parquet]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tabdistances.columns.column_features import get_histogram

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
_ROOT        = Path(__file__).resolve().parent.parent

RANKINGS_CSV          = _ROOT / "results/tabicl-rankings-full-context.csv"
AUTOGLUON_RESULTS_CSV = _ROOT / "results/results-autogluon-512-context.csv"
TALENT_DIR            = _ROOT / "data"
FM_DIR       = _ROOT / "tables-tabfm/parquet"
T4_DIR       = _ROOT / "tables-t4/parquet"
ICL_DIR      = _ROOT / "tables-tabicl/parquet"

CACHE_DIR    = _ROOT / "cache"
OUTPUT_DIR   = _ROOT / "results/column-coverage"

MAX_ROWS     = 512
KNN_K        = 5
RANDOM_STATE = 42
WANDB_PROJECT = "tabicl-column-coverage"

_PAPER_RC = {
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 0.9,
    "figure.dpi":      300,
}

_COLORS = {"talent": "#2980b9", "fm": "#e67e22", "t4": "#27ae60", "icl": "#8e44ad"}
_LABELS = {"talent": "TALENT", "fm": "Curated (FM)", "t4": "Web-Scraped (T4)", "icl": "TabICL Prior"}


def _stairs(ax, density, bins, color, label, alpha_fill=0.2, lw=1.5):
    """Step-filled histogram with opaque edge and transparent fill."""
    patch = ax.stairs(density, bins, fill=True, label=label)
    patch.set_facecolor((*mpl.colors.to_rgb(color), alpha_fill))
    patch.set_edgecolor(color)
    patch.set_linewidth(lw)
    return patch


# ------------------------------------------------------------------
# TALENT column loader
# ------------------------------------------------------------------

def load_talent_col_hists(
    talent_dir: Path,
    datasets: list[str],
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (hists, col_datasets) where hists is (N_cols, 30) and
    col_datasets is (N_cols,) string array of the parent dataset name."""
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        hists = data["hists"]
        col_datasets = data["col_datasets"]
        print(f"  Loaded {len(hists)} TALENT column histograms from cache")
        return hists, col_datasets

    hists, col_datasets = [], []
    skipped_datasets = 0
    for ds in tqdm(datasets, desc="TALENT columns"):
        ds_path = talent_dir / ds
        if not ds_path.exists():
            skipped_datasets += 1
            continue
        parts = []
        for split in ("train", "val", "test"):
            npy = ds_path / f"N_{split}.npy"
            if npy.exists():
                arr = np.load(npy, allow_pickle=True).astype(float)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                parts.append(arr)
        if not parts:
            skipped_datasets += 1
            continue
        data = np.concatenate(parts, axis=0)[:MAX_ROWS]
        n = len(data)
        for i in range(data.shape[1]):
            col = pd.Series(data[:, i]).dropna()
            if n > 0 and col.nunique() / n >= 0.2:
                hists.append(get_histogram(col))
                col_datasets.append(ds)

    print(f"  Computed {len(hists)} TALENT column histograms ({skipped_datasets} datasets skipped)")
    hists_arr = np.array(hists)
    col_datasets_arr = np.array(col_datasets)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, hists=hists_arr, col_datasets=col_datasets_arr)
    print(f"  Cached -> {cache_path}")
    return hists_arr, col_datasets_arr


# ------------------------------------------------------------------
# Parquet column index builder + sampler
# ------------------------------------------------------------------

def _build_col_index(parquet_dir: Path) -> list[tuple[Path, str]]:
    """Return flat list of (parquet_path, col_name) for every numeric column."""
    index = []
    paths = sorted(parquet_dir.glob("*.parquet"))
    for p in tqdm(paths, desc=f"Indexing {parquet_dir.name}", leave=False):
        try:
            df = pd.read_parquet(p)
            num_cols = df.select_dtypes(include=["float"]).columns.tolist()
            n = len(df)
            for col in num_cols:
                if n > 0 and df[col].nunique() / n >= 0.2:
                    index.append((p, col))
        except Exception:
            pass
    return index


def _compute_sampled_hists(
    col_index: list[tuple[Path, str]],
    sampled_idx: np.ndarray,
) -> np.ndarray:
    """Load only the files needed for sampled_idx and compute histograms."""
    # Group selected (path, col) pairs by file to minimise parquet reads
    file_to_cols: dict[Path, list[tuple[int, str]]] = defaultdict(list)
    for rank_i, idx in enumerate(sampled_idx):
        path, col = col_index[idx]
        file_to_cols[path].append((rank_i, col))

    hists = np.zeros((len(sampled_idx), 30))
    for path, items in tqdm(file_to_cols.items(), desc="Computing histograms", leave=False):
        try:
            needed_cols = [col for _, col in items]
            df = pd.read_parquet(path, columns=needed_cols)
            df = df.iloc[:MAX_ROWS]
            for rank_i, col in items:
                if col in df.columns:
                    hists[rank_i] = get_histogram(df[col])
        except Exception:
            pass
    return hists


def load_parquet_col_hists(
    parquet_dir: Path,
    n: int,
    rng: np.random.Generator,
    cache_path: Path,
) -> np.ndarray:
    """Return (n, 30) array of column histograms sampled uniformly from parquet_dir."""
    if cache_path.exists():
        data = np.load(cache_path)
        arr = data["hists"]
        # If we need fewer than cached, subsample; if more, warn and use all
        if len(arr) >= n:
            idx = rng.choice(len(arr), size=n, replace=False)
            print(f"  Loaded {n} column histograms from cache ({cache_path.name})")
            return arr[idx]
        print(f"  Cache has {len(arr)} columns (< {n} requested), using all")
        return arr

    index = _build_col_index(parquet_dir)
    if not index:
        print(f"  WARNING: no numeric columns found in {parquet_dir}")
        return np.zeros((0, 30))

    print(f"  Total numeric columns in {parquet_dir.name}: {len(index)}")
    sampled_idx = rng.choice(len(index), size=min(n, len(index)), replace=False)
    hists = _compute_sampled_hists(index, sampled_idx)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, hists=hists)
    print(f"  Cached {len(hists)} histograms -> {cache_path}")
    return hists


# ------------------------------------------------------------------
# NN distances
# ------------------------------------------------------------------

def compute_nn_distances(
    query: np.ndarray,
    ref: np.ndarray,
    k: int,
) -> np.ndarray:
    """Mean k-NN distance from each query column to the reference pool."""
    query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
    ref   = np.nan_to_num(ref,   nan=0.0, posinf=0.0, neginf=0.0)
    scaler = MinMaxScaler().fit(ref)
    q_sc = scaler.transform(query)
    r_sc = scaler.transform(ref)
    tree = BallTree(r_sc, metric="euclidean")
    dists, _ = tree.query(q_sc, k=min(k, len(r_sc)))
    return dists.mean(axis=1)


# ------------------------------------------------------------------
# Plot 1 — mean ± SD histogram band
# ------------------------------------------------------------------

def plot_histogram_bands(hists: dict[str, np.ndarray], output_dir: Path, run=None) -> None:
    """Mean ± 1 SD band of cumulative histogram values across columns, per source."""
    bins = np.arange(30)
    with mpl.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for key, arr in hists.items():
            if len(arr) == 0:
                continue
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0)
            color = _COLORS[key]
            ax.plot(bins, mu, color=color, label=_LABELS[key], lw=1.2)
            ax.fill_between(bins, mu - sd, mu + sd, color=color, alpha=0.2)
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title("Column histogram distributions")
        ax.legend()
        plt.tight_layout()
        out = output_dir / "column_histogram_bands.png"
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({"column_histogram_bands": wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------------
# Plot 2 — PCA
# ------------------------------------------------------------------

def plot_pca(hists: dict[str, np.ndarray], output_dir: Path, run=None) -> None:
    """2-D PCA of all column histograms coloured by source."""
    arrays, labels = [], []
    for key, arr in hists.items():
        if len(arr) == 0:
            continue
        arrays.append(arr)
        labels.extend([key] * len(arr))

    X = np.nan_to_num(np.concatenate(arrays), nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    with mpl.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for key in hists:
            mask = np.array(labels) == key
            if mask.sum() == 0:
                continue
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       color=_COLORS[key], label=_LABELS[key],
                       alpha=0.4, s=8, linewidths=0)
        ax.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var[1]:.1%})")
        ax.set_title("PCA of column histograms")
        ax.legend()
        plt.tight_layout()
        out = output_dir / "column_pca.png"
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({"column_pca": wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------------
# Plot 3 — NN distance density histograms
# ------------------------------------------------------------------

def plot_nn_distance_densities(
    nn_pairs: dict[str, np.ndarray],
    output_dir: Path,
    run=None,
    n_bins: int = 40,
) -> None:
    """Overlaid density histograms of pairwise NN distances."""
    pair_colors = {
        "talent_to_icl": _COLORS["icl"],
        "talent_to_fm":  _COLORS["fm"],
        "talent_to_t4":  _COLORS["t4"],
        "fm_to_t4":      "#7f8c8d",
    }
    pair_labels = {
        "talent_to_icl": f"TALENT  $\\in$  {_LABELS['icl']}",
        "talent_to_fm":  f"TALENT  $\\in$  {_LABELS['fm']}",
        "talent_to_t4":  f"TALENT  $\\in$  {_LABELS['t4']}",
        "fm_to_t4":      f"{_LABELS['fm']}  $\\in$  {_LABELS['t4']}",
    }

    with mpl.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        all_dists = np.concatenate([d for d in nn_pairs.values() if d is not None and len(d)])
        bins = np.linspace(all_dists.min(), all_dists.max(), n_bins + 1)
        bin_w = bins[1] - bins[0]
        for key, dists in nn_pairs.items():
            if dists is None or len(dists) == 0:
                continue
            idx = np.clip(np.digitize(dists, bins) - 1, 0, n_bins - 1)
            counts = np.array([(idx == i).sum() for i in range(n_bins)])
            density = counts / (counts.sum() * bin_w)
            _stairs(ax, density, bins, pair_colors.get(key, "grey"), pair_labels.get(key, key))
        ax.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
        ax.set_ylabel("Density")
        ax.set_title("Column-level NN distances")
        ax.legend()
        plt.tight_layout()
        out = output_dir / "column_nn_distances.png"
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({"column_nn_distances": wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

_BLUE   = "#2980b9"
_ORANGE = "#e67e22"
_GREEN  = "#27ae60"
_PURPLE = "#8e44ad"
_RED    = "#ef0e0e"


def _plot_metric_histogram(
    nn_dists: np.ndarray,
    y_vals: np.ndarray,
    y_label: str,
    filename: str,
    output_dir: Path,
    run=None,
    n_bins: int = 30,
    invert_y: bool = False,
    nn_dists_fm: np.ndarray | None = None,
    nn_dists_icl_self: np.ndarray | None = None,
) -> None:
    """NN-distance density histogram (left y) with mean metric ± 1 SD per bin (right y).

    Optionally overlays FM and ICL self-distance distributions on the left axis.
    nn_dists is the TALENT $\\in$ ICL distribution used for the metric binning.
    """
    all_parts = [nn_dists]
    if nn_dists_fm       is not None: all_parts.append(nn_dists_fm)
    if nn_dists_icl_self is not None: all_parts.append(nn_dists_icl_self)
    all_dists = np.concatenate(all_parts)
    bins = np.linspace(all_dists.min(), all_dists.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_w = bins[1] - bins[0]

    idx = np.clip(np.digitize(nn_dists, bins) - 1, 0, n_bins - 1)
    counts = np.array([(idx == i).sum() for i in range(n_bins)])
    density = counts / counts.max()
    bin_means = np.array([y_vals[idx == i].mean() if (idx == i).sum() > 0 else np.nan for i in range(n_bins)])
    bin_stds  = np.array([y_vals[idx == i].std()  if (idx == i).sum() > 1 else 0.0   for i in range(n_bins)])

    with mpl.rc_context(_PAPER_RC):
        fig, ax1 = plt.subplots(figsize=(5, 3.5))
        _stairs(ax1, density, bins, _BLUE, "TALENT columns", alpha_fill=0.15)
        if nn_dists_fm is not None:
            fm_idx = np.clip(np.digitize(nn_dists_fm, bins) - 1, 0, n_bins - 1)
            fm_counts = np.array([(fm_idx == i).sum() for i in range(n_bins)])
            _stairs(ax1, fm_counts / fm_counts.max(), bins, _ORANGE, _LABELS["fm"], alpha_fill=0.15)
        if nn_dists_icl_self is not None:
            si_idx = np.clip(np.digitize(nn_dists_icl_self, bins) - 1, 0, n_bins - 1)
            si_counts = np.array([(si_idx == i).sum() for i in range(n_bins)])
            _stairs(ax1, si_counts / si_counts.max(), bins, _PURPLE, "TabICL Prior (self)", alpha_fill=0.15)
        ax1.set_xlabel("Mean k-NN distance (normalised)")
        ax1.set_ylabel("Relative density")
        ax2 = ax1.twinx()
        mask = ~np.isnan(bin_means)
        lo = np.clip(bin_means - bin_stds, 1, None) if invert_y else bin_means - bin_stds
        hi = bin_means + bin_stds
        ax2.plot(bin_centers[mask], bin_means[mask], color=_RED, lw=1.5, marker="o", ms=4, label=f"Mean {y_label}")
        ax2.fill_between(bin_centers[mask], lo[mask], hi[mask], color=_RED, alpha=0.25)
        if invert_y:
            ax2.invert_yaxis()
        ax2.set_ylabel(f"TabICL mean {y_label}", color=_RED)
        ax2.tick_params(axis="y", labelcolor=_RED)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", bbox_to_anchor=(1.12, 1), borderaxespad=0)
        plt.title(f"Column NN Distance vs {y_label}")
        plt.tight_layout()
        out = output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({filename.rsplit(".", 1)[0]: wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


def _plot_metric_scatter(
    nn_dists: np.ndarray,
    y_vals: np.ndarray,
    y_label: str,
    filename: str,
    output_dir: Path,
    run=None,
) -> dict:
    """Scatter of metric vs NN distance with Pearson r and Spearman ρ."""
    r_p, p_p = stats.pearsonr(nn_dists, y_vals)
    r_s, p_s = stats.spearmanr(nn_dists, y_vals)
    m, b = np.polyfit(nn_dists, y_vals.astype(float), 1)
    xs = np.linspace(nn_dists.min(), nn_dists.max(), 200)
    with mpl.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.scatter(nn_dists, y_vals, alpha=0.3, s=8, color=_BLUE, linewidths=0, zorder=3)
        ax.plot(xs, m * xs + b, color=_ORANGE, lw=1.5, label="Linear fit", zorder=4)
        ax.set_xlabel("Mean k-NN distance (normalised)")
        ax.set_ylabel(f"TabICL {y_label}")
        ax.set_title(
            f"{y_label} vs column NN distance\n"
            f"Pearson r = {r_p:.3f} (p = {p_p:.3f})   Spearman ρ = {r_s:.3f} (p = {p_s:.3f})"
        )
        ax.legend()
        plt.tight_layout()
        out = output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({filename.rsplit(".", 1)[0]: wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")
    return {"pearson_r": r_p, "pearson_p": p_p, "spearman_r": r_s, "spearman_p": p_s}


def _plot_metric_prediction(
    nn_dists: np.ndarray,
    y_vals: np.ndarray,
    y_label: str,
    filename: str,
    output_dir: Path,
    run=None,
) -> dict:
    """Linear regression: metric ~ column NN distance."""
    X = nn_dists.reshape(-1, 1)
    y = y_vals.astype(float)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    residuals = y - y_pred
    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
        lo, hi = y.min(), y.max()
        axes[0].scatter(y, y_pred, alpha=0.3, s=8, color=_GREEN, linewidths=0)
        axes[0].plot([lo, hi], [lo, hi], "k--", lw=0.8, label="Perfect fit")
        axes[0].set_xlabel(f"Actual {y_label}")
        axes[0].set_ylabel(f"Predicted {y_label}")
        axes[0].set_title(f"Linear Regression  R² = {r2:.3f}")
        axes[0].legend()
        axes[1].scatter(y_pred, residuals, alpha=0.3, s=8, color=_ORANGE, linewidths=0)
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xlabel(f"Predicted {y_label}")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals")
        plt.tight_layout()
        out = output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({filename.rsplit(".", 1)[0]: wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")
    return {"r2": r2, "coef": float(model.coef_[0]), "intercept": float(model.intercept_)}


# ------------------------------------------------------------------
# Plot — example column histograms (4 sources × 6 examples)
# ------------------------------------------------------------------

def plot_example_histograms(
    hists: dict[str, np.ndarray],
    output_dir: Path,
    run=None,
    n_examples: int = 6,
    seed: int = RANDOM_STATE,
) -> None:
    """4-column × n_examples-row grid of randomly sampled column histograms per source."""
    sources = [k for k in ("talent", "fm", "t4", "icl") if k in hists and len(hists[k])]
    n_cols = len(sources)
    rng = np.random.default_rng(seed)
    bins = np.arange(30)

    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(n_examples, n_cols, figsize=(2.0 * n_cols, 1.4 * n_examples),
                                 sharex=True, sharey=True)
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for col_i, src in enumerate(sources):
            arr = hists[src]
            chosen = rng.choice(len(arr), size=min(n_examples, len(arr)), replace=False)
            color = _COLORS[src]
            for row_i in range(n_examples):
                ax = axes[row_i, col_i]
                if row_i < len(chosen):
                    ax.plot(bins, arr[chosen[row_i]], color=color, lw=0.9)
                    ax.fill_between(bins, arr[chosen[row_i]], alpha=0.15, color=color)
                else:
                    ax.set_visible(False)
                ax.set_ylim(0, 1)
                if row_i == 0:
                    ax.set_title(_LABELS[src], color=color)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        fig.supxlabel("Bin index", fontsize=8)
        fig.supylabel("Cumulative fraction", fontsize=8)
        plt.suptitle("Example column histograms by source", fontsize=9)
        plt.tight_layout()
        out = output_dir / "column_examples.png"
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({"column_examples": wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


def _load_ag_ranks(results_csv: Path) -> pd.DataFrame:
    """Load autogluon and tabicl-with-ag ranks from the autogluon results CSV."""
    df = pd.read_csv(results_csv)
    ranking_rows = df[df["metric"] == "ranking"][["dataset", "model", "value"]].copy()
    ranking_rows["value"] = ranking_rows["value"].astype(int)
    ag = (
        ranking_rows[ranking_rows["model"] == "autogluon"]
        .rename(columns={"value": "ag_rank"})[["dataset", "ag_rank"]]
    )
    tc = (
        ranking_rows[ranking_rows["model"] == "tabicl"]
        .rename(columns={"value": "tabicl_rank_with_ag"})[["dataset", "tabicl_rank_with_ag"]]
    )
    return ag.merge(tc, on="dataset", how="outer")


def _plot_comparison_scatter(
    nn_dists: np.ndarray,
    y_tabicl: np.ndarray,
    y_autogluon: np.ndarray,
    y_label: str,
    filename: str,
    output_dir: Path,
    run=None,
) -> None:
    """Overlay TabICL and AutoGluon rank/metric vs column NN distance on one plot."""
    with mpl.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        for y_vals, color, label in [
            (y_tabicl,    _BLUE,   "TabICL"),
            (y_autogluon, _ORANGE, "AutoGluon"),
        ]:
            mask = ~np.isnan(y_vals)
            if mask.sum() < 2:
                continue
            xm, ym = nn_dists[mask], y_vals[mask]
            r_p, _ = stats.pearsonr(xm, ym)
            r_s, _ = stats.spearmanr(xm, ym)
            m, b   = np.polyfit(xm, ym.astype(float), 1)
            xs     = np.linspace(xm.min(), xm.max(), 200)
            ax.scatter(xm, ym, alpha=0.25, s=6, color=color, linewidths=0, zorder=3,
                       label=f"{label}  r={r_p:.2f}  ρ={r_s:.2f}")
            ax.plot(xs, m * xs + b, color=color, lw=1.5, zorder=4)
        ax.set_xlabel("Mean k-NN distance (normalised)")
        ax.set_ylabel(y_label)
        ax.set_title(f"TabICL vs AutoGluon — {y_label} vs column NN distance")
        ax.legend()
        plt.tight_layout()
        out = output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({filename.rsplit(".", 1)[0]: wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


def main(
    talent_dir: Path,
    fm_dir: Path,
    t4_dir: Path,
    icl_dir: Path,
    output_dir: Path,
    k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    # ── Rankings ───────────────────────────────────────────────────
    rankings = pd.read_csv(RANKINGS_CSV)
    mean_aucs = rankings.groupby("dataset")["AUC"].mean().rename("mean_AUC")
    tabicl_ranks = (
        rankings[rankings["model"] == "tabicl"][["dataset", "rank", "AUC"]]
        .copy()
    )
    tabicl_ranks["rank"] = tabicl_ranks["rank"].astype(int)
    tabicl_ranks = tabicl_ranks.merge(mean_aucs, on="dataset")
    tabicl_ranks["relative_auc"] = tabicl_ranks["AUC"] / tabicl_ranks["mean_AUC"]
    datasets = tabicl_ranks["dataset"].tolist()
    rank_map = tabicl_ranks.set_index("dataset")[["rank", "AUC", "relative_auc"]].to_dict("index")
    print(f"\n=== TALENT: {len(datasets)} datasets ===")

    # ── Autogluon rankings ─────────────────────────────────────────
    print("=== Loading AutoGluon ranks ===")
    ag_df = _load_ag_ranks(AUTOGLUON_RESULTS_CSV)
    ag_rank_map     = ag_df.set_index("dataset")["ag_rank"].to_dict()
    tc_ag_rank_map  = ag_df.set_index("dataset")["tabicl_rank_with_ag"].to_dict()

    # ── TALENT column histograms ───────────────────────────────────
    talent_cache = CACHE_DIR / "talent_col_hists.npz"
    talent_hists, col_datasets = load_talent_col_hists(talent_dir, datasets, talent_cache)
    n_talent = len(talent_hists)
    print(f"  {n_talent} TALENT numeric columns")

    # ── FM column histograms ───────────────────────────────────────
    print(f"\n=== FM columns (sampling {n_talent}) ===")
    fm_cache = CACHE_DIR / f"fm_col_hists_{n_talent}.npz"
    fm_hists = load_parquet_col_hists(fm_dir, n_talent, rng, fm_cache) if fm_dir.exists() else np.zeros((0, 30))

    # ── T4 column histograms ───────────────────────────────────────
    print(f"\n=== T4 columns (sampling {n_talent}) ===")
    t4_cache = CACHE_DIR / f"t4_col_hists_{n_talent}.npz"
    t4_hists = load_parquet_col_hists(t4_dir, n_talent, rng, t4_cache) if t4_dir.exists() else np.zeros((0, 30))

    # ── ICL column histograms ──────────────────────────────────────
    print(f"\n=== ICL columns (sampling {n_talent}) ===")
    icl_cache = CACHE_DIR / f"icl_col_hists_{n_talent}.npz"
    icl_hists = load_parquet_col_hists(icl_dir, n_talent, rng, icl_cache) if icl_dir.exists() else np.zeros((0, 30))

    hists = {"talent": talent_hists, "fm": fm_hists, "t4": t4_hists, "icl": icl_hists}
    for key, arr in hists.items():
        print(f"  {_LABELS[key]}: {len(arr)} columns")

    # ── wandb ──────────────────────────────────────────────────────
    run = None
    if _WANDB_AVAILABLE and WANDB_PROJECT:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"column_coverage_k{k}",
            config={"k": k, "n_talent": n_talent,
                    "n_fm": len(fm_hists), "n_t4": len(t4_hists), "n_icl": len(icl_hists)},
            reinit=True,
        )

    # ── Plots ──────────────────────────────────────────────────────
    print("\n=== Generating figures ===")
    plot_histogram_bands(hists, output_dir, run)
    plot_pca(hists, output_dir, run)
    plot_example_histograms(hists, output_dir, run)

    print("\n=== Computing NN distances ===")
    nn_pairs = {}
    if len(icl_hists):
        print("  TALENT  $\\in$  ICL")
        nn_pairs["talent_to_icl"] = compute_nn_distances(talent_hists, icl_hists, k)
    if len(fm_hists):
        print("  TALENT  $\\in$  FM")
        nn_pairs["talent_to_fm"] = compute_nn_distances(talent_hists, fm_hists, k)
    if len(t4_hists):
        print("  TALENT  $\\in$  T4")
        nn_pairs["talent_to_t4"] = compute_nn_distances(talent_hists, t4_hists, k)
    if len(fm_hists) and len(t4_hists):
        print("  FM  $\\in$  T4")
        nn_pairs["fm_to_t4"] = compute_nn_distances(fm_hists, t4_hists, k)

    plot_nn_distance_densities(nn_pairs, output_dir, run)

    # ── Rank / AUC analysis at column level ────────────────────────
    # Each column inherits its parent dataset's tabicl rank and AUC.
    # Use TALENT $\\in$ ICL distances as primary (fall back to FM, then T4).
    _icl = nn_pairs.get("talent_to_icl")
    _fm  = nn_pairs.get("talent_to_fm")
    _t4  = nn_pairs.get("talent_to_t4")
    col_nn = _icl if _icl is not None else (_fm if _fm is not None else _t4)
    ref_label = "ICL" if _icl is not None else ("FM" if _fm is not None else "T4")
    if col_nn is not None:
        print("\n=== Rank / AUC analysis (column level) ===")
        col_ranks         = np.array([rank_map[ds]["rank"]         for ds in col_datasets
                                      if ds in rank_map], dtype=float)
        col_aucs          = np.array([rank_map[ds]["AUC"]          for ds in col_datasets
                                      if ds in rank_map], dtype=float)
        col_relative_aucs = np.array([rank_map[ds]["relative_auc"] for ds in col_datasets
                                      if ds in rank_map], dtype=float)
        col_ag_ranks      = np.array([ag_rank_map.get(ds, np.nan)    for ds in col_datasets
                                      if ds in rank_map], dtype=float)
        col_tc_ag_ranks   = np.array([tc_ag_rank_map.get(ds, np.nan) for ds in col_datasets
                                      if ds in rank_map], dtype=float)
        mask = np.array([ds in rank_map for ds in col_datasets])
        col_nn_masked = col_nn[mask]

        fm_masked = _fm[mask] if _fm is not None else None

        # ICL self-distances: split ICL columns in half to avoid self-match
        icl_self_dists = None
        if len(icl_hists) >= 2:
            half = len(icl_hists) // 2
            icl_self_dists = compute_nn_distances(icl_hists[:half], icl_hists[half:], k)

        # TabICL plots (7-model field)
        _plot_metric_histogram(col_nn_masked, col_ranks, "rank (lower = better)",
                               "col_rank_vs_nndist_histogram.png", output_dir, run,
                               invert_y=True,
                               nn_dists_fm=fm_masked, nn_dists_icl_self=icl_self_dists)
        _plot_metric_histogram(col_nn_masked, col_aucs, "AUC",
                               "col_auc_vs_nndist_histogram.png", output_dir, run,
                               nn_dists_fm=fm_masked, nn_dists_icl_self=icl_self_dists)
        _plot_metric_histogram(col_nn_masked, col_relative_aucs, "relative AUC",
                               "col_relative_auc_vs_nndist_histogram.png", output_dir, run,
                               nn_dists_fm=fm_masked, nn_dists_icl_self=icl_self_dists)
        corr = _plot_metric_scatter(col_nn_masked, col_ranks, f"rank (TALENT $\\in$ {ref_label})",
                                    "col_rank_vs_nndist_scatter.png", output_dir, run)
        _plot_metric_scatter(col_nn_masked, col_aucs, f"AUC (TALENT $\\in$ {ref_label})",
                             "col_auc_vs_nndist_scatter.png", output_dir, run)
        corr_rel = _plot_metric_scatter(col_nn_masked, col_relative_aucs,
                                        f"relative AUC (TALENT $\\in$ {ref_label})",
                                        "col_relative_auc_vs_nndist_scatter.png", output_dir, run)
        pred = _plot_metric_prediction(col_nn_masked, col_ranks, "rank",
                                       "col_rank_prediction.png", output_dir, run)
        _plot_metric_prediction(col_nn_masked, col_aucs, "AUC",
                                "col_auc_prediction.png", output_dir, run)
        pred_rel = _plot_metric_prediction(col_nn_masked, col_relative_aucs, "relative AUC",
                                           "col_relative_auc_prediction.png", output_dir, run)

        # AutoGluon plots (8-model field, datasets with autogluon results only)
        ag_mask = ~np.isnan(col_ag_ranks)
        if ag_mask.sum() >= 2:
            nn_ag = col_nn_masked[ag_mask]
            _plot_metric_histogram(nn_ag, col_ag_ranks[ag_mask],
                                   "AutoGluon rank (lower = better)",
                                   "col_ag_rank_vs_nndist_histogram.png", output_dir, run,
                                   invert_y=True, nn_dists_icl_self=icl_self_dists)
            corr_ag = _plot_metric_scatter(nn_ag, col_ag_ranks[ag_mask],
                                           f"AutoGluon rank (TALENT $\\in$ {ref_label})",
                                           "col_ag_rank_vs_nndist_scatter.png", output_dir, run)
            pred_ag = _plot_metric_prediction(nn_ag, col_ag_ranks[ag_mask], "AutoGluon rank",
                                              "col_ag_rank_prediction.png", output_dir, run)
            _plot_comparison_scatter(nn_ag, col_tc_ag_ranks[ag_mask], col_ag_ranks[ag_mask],
                                     "rank (lower = better, 8-model field)",
                                     "col_tabicl_vs_ag_rank_scatter.png", output_dir, run)
        else:
            corr_ag = pred_ag = None
            print("  Skipping autogluon column plots (insufficient matched datasets)")

        print(f"\n=== Results (TabICL, 7-model field, n={mask.sum()} cols) ===")
        print(f"  Pearson  r (rank)         : {corr['pearson_r']:.3f}  (p = {corr['pearson_p']:.4f})")
        print(f"  Spearman ρ (rank)         : {corr['spearman_r']:.3f}  (p = {corr['spearman_p']:.4f})")
        print(f"  Linear R²  (rank)         : {pred['r2']:.3f}")
        print(f"  Pearson  r (relative AUC) : {corr_rel['pearson_r']:.3f}  (p = {corr_rel['pearson_p']:.4f})")
        print(f"  Spearman ρ (relative AUC) : {corr_rel['spearman_r']:.3f}  (p = {corr_rel['spearman_p']:.4f})")
        print(f"  Linear R²  (relative AUC) : {pred_rel['r2']:.3f}")
        if corr_ag is not None:
            print(f"\n=== Results (AutoGluon, 8-model field, n={ag_mask.sum()} cols) ===")
            print(f"  Pearson  r (rank) : {corr_ag['pearson_r']:.3f}  (p = {corr_ag['pearson_p']:.4f})")
            print(f"  Spearman ρ (rank) : {corr_ag['spearman_r']:.3f}  (p = {corr_ag['spearman_p']:.4f})")
            print(f"  Linear R²  (rank) : {pred_ag['r2']:.3f}")

        if run is not None:
            log_dict = {
                "col_pearson_r":      corr["pearson_r"],
                "col_spearman_r":     corr["spearman_r"],
                "col_linear_r2":      pred["r2"],
                "col_rel_pearson_r":  corr_rel["pearson_r"],
                "col_rel_spearman_r": corr_rel["spearman_r"],
                "col_rel_linear_r2":  pred_rel["r2"],
            }
            if corr_ag is not None:
                log_dict.update({
                    "col_ag_pearson_r":  corr_ag["pearson_r"],
                    "col_ag_spearman_r": corr_ag["spearman_r"],
                    "col_ag_linear_r2":  pred_ag["r2"],
                })
            run.log(log_dict)

    if run is not None:
        run.finish()

    print(f"\nOutputs in: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Column-level coverage analysis: TALENT vs FM vs T4 vs ICL."
    )
    parser.add_argument("--talent-dir", type=Path, default=TALENT_DIR)
    parser.add_argument("--fm-dir",     type=Path, default=FM_DIR)
    parser.add_argument("--t4-dir",     type=Path, default=T4_DIR)
    parser.add_argument("--icl-dir",    type=Path, default=ICL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--k",          type=int,  default=KNN_K)
    args = parser.parse_args()

    main(
        talent_dir=args.talent_dir,
        fm_dir=args.fm_dir,
        t4_dir=args.t4_dir,
        icl_dir=args.icl_dir,
        output_dir=args.output_dir,
        k=args.k,
    )
