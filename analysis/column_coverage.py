"""Column-level coverage analysis: TALENT vs FM vs T4.

Operates on individual numeric columns rather than whole tables.  The feature
for each column is the 30-bin normalised cumulative histogram produced by
get_histogram().

Workflow
--------
1. Load every numeric column from every TALENT dataset (train+val+test).
2. Build a flat (file, col_name) index of all numeric columns in FM and T4.
3. Sample exactly N_talent_cols uniformly from that index — genuinely random
   across all available columns, not just random tables.
4. Compute histograms for all three column pools.
5. Produce three outputs:
     a. Mean ± SD histogram band plot (TALENT vs FM vs T4).
     b. PCA of all column histograms coloured by source.
     c. Pairwise NN-distance density histograms (TALENT→FM, TALENT→T4,
        FM→T4) using the same compute_nn_distances approach as
        rank_vs_coverage.py.

Usage
-----
    python analysis/column_coverage.py [--talent-dir data/]
                                       [--fm-dir tables-tabfm/parquet]
                                       [--t4-dir tables-t4/parquet]
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
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tabdistances.columns.node_features import get_histogram

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
_ROOT        = Path(__file__).resolve().parent.parent

RANKINGS_CSV = _ROOT / "results/tabicl-rankings.csv"
TALENT_DIR   = _ROOT / "data"
FM_DIR       = _ROOT / "tables-tabfm/parquet"
T4_DIR       = _ROOT / "tables-t4/parquet"

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

_COLORS = {"talent": "#2980b9", "fm": "#e67e22", "t4": "#27ae60"}
_LABELS = {"talent": "TALENT", "fm": "Curated (FM)", "t4": "Web-Scraped (T4)"}


# ------------------------------------------------------------------
# TALENT column loader
# ------------------------------------------------------------------

def load_talent_col_hists(
    talent_dir: Path,
    datasets: list[str],
    cache_path: Path,
) -> np.ndarray:
    """Return (N_cols, 30) histogram array for all TALENT numeric columns."""
    if cache_path.exists():
        arr = np.load(cache_path)["hists"]
        print(f"  Loaded {len(arr)} TALENT column histograms from cache")
        return arr

    hists = []
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
        for i in range(data.shape[1]):
            h = get_histogram(pd.Series(data[:, i]))
            hists.append(h)

    print(f"  Computed {len(hists)} TALENT column histograms ({skipped_datasets} datasets skipped)")
    arr = np.array(hists)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, hists=arr)
    print(f"  Cached -> {cache_path}")
    return arr


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
            num_cols = df.select_dtypes(include=[float, int, "number"]).columns.tolist()
            for col in num_cols:
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
        "talent_to_fm": _COLORS["fm"],
        "talent_to_t4": _COLORS["t4"],
        "fm_to_t4":     "#8e44ad",
    }
    pair_labels = {
        "talent_to_fm": f"TALENT → {_LABELS['fm']}",
        "talent_to_t4": f"TALENT → {_LABELS['t4']}",
        "fm_to_t4":     f"{_LABELS['fm']} → {_LABELS['t4']}",
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
            ax.stairs(density, bins, fill=True, alpha=0.2,
                      color=pair_colors.get(key, "grey"),
                      linewidth=2.0, label=pair_labels.get(key, key))
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

def main(
    talent_dir: Path,
    fm_dir: Path,
    t4_dir: Path,
    output_dir: Path,
    k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    # ── Rankings → dataset list ────────────────────────────────────
    rankings = pd.read_csv(RANKINGS_CSV)
    datasets = rankings["dataset"].unique().tolist()
    print(f"\n=== TALENT: {len(datasets)} datasets ===")

    # ── TALENT column histograms ───────────────────────────────────
    talent_cache = CACHE_DIR / "talent_col_hists.npz"
    talent_hists = load_talent_col_hists(talent_dir, datasets, talent_cache)
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

    hists = {"talent": talent_hists, "fm": fm_hists, "t4": t4_hists}
    for key, arr in hists.items():
        print(f"  {_LABELS[key]}: {len(arr)} columns")

    # ── wandb ──────────────────────────────────────────────────────
    run = None
    if _WANDB_AVAILABLE and WANDB_PROJECT:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"column_coverage_k{k}",
            config={"k": k, "n_talent": n_talent,
                    "n_fm": len(fm_hists), "n_t4": len(t4_hists)},
            reinit=True,
        )

    # ── Plots ──────────────────────────────────────────────────────
    print("\n=== Generating figures ===")
    plot_histogram_bands(hists, output_dir, run)
    plot_pca(hists, output_dir, run)

    print("\n=== Computing NN distances ===")
    nn_pairs = {}
    if len(fm_hists):
        print("  TALENT → FM")
        nn_pairs["talent_to_fm"] = compute_nn_distances(talent_hists, fm_hists, k)
    if len(t4_hists):
        print("  TALENT → T4")
        nn_pairs["talent_to_t4"] = compute_nn_distances(talent_hists, t4_hists, k)
    if len(fm_hists) and len(t4_hists):
        print("  FM → T4")
        nn_pairs["fm_to_t4"] = compute_nn_distances(fm_hists, t4_hists, k)

    plot_nn_distance_densities(nn_pairs, output_dir, run)

    if run is not None:
        run.finish()

    print(f"\nOutputs in: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Column-level coverage analysis: TALENT vs FM vs T4."
    )
    parser.add_argument("--talent-dir", type=Path, default=TALENT_DIR)
    parser.add_argument("--fm-dir",     type=Path, default=FM_DIR)
    parser.add_argument("--t4-dir",     type=Path, default=T4_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--k",          type=int,  default=KNN_K)
    args = parser.parse_args()

    main(
        talent_dir=args.talent_dir,
        fm_dir=args.fm_dir,
        t4_dir=args.t4_dir,
        output_dir=args.output_dir,
        k=args.k,
    )
