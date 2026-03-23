"""Pairwise distribution comparison using column-level and correlation histograms.

Two complementary feature representations are compared across all three
dataset pairs (icl vs fm, t4 vs fm, icl vs t4):

  col_hist  — 30-bin normalised cumulative histograms of individual numeric
              columns (one row per column, sampled across all tables).
              Captures the marginal distribution shape of each column.

  corr_hist — 30-bin normalised cumulative histogram of all upper-triangle
              Pearson correlations within each table (one row per table).
              Captures the inter-column correlation structure.

For each pair × feature type the script runs:
  1. Bootstrap AUC (discriminability)
  2. Coverage recall  — does A cover B?
  3. Coverage precision — does B cover A?
  4. PCA projection

Caches are compatible with those produced by rank_vs_coverage_columns.py and
correlation_coverage.py so existing caches are reused automatically.

Logs to the same wandb project as the original pairwise comparisons:
  tabicl-coverage-pairwise

Usage
-----
    python analysis/pairwise_col_corr.py [--tables-dir tables-tabfm/parquet]
                                         [--icl-dir tables-tabicl/parquet]
                                         [--n-sample 25000]
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
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tabdistances.columns.node_features import get_histogram
import analysis.coverage_analysis as _ca
from analysis.coverage_analysis import (
    _PAPER_RC, KNN_K, RANDOM_STATE,
    analyse_bootstrap_auc, analyse_coverage, analyse_pca,
    _draw_bootstrap_auc, _draw_coverage_dists, _draw_coverage_curve,
    _draw_pca, _save_and_log,
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
_ROOT         = Path(__file__).resolve().parent.parent
FM_DIR        = _ROOT / "tables-tabfm/parquet"
T4_DIR        = _ROOT / "tables-t4/parquet"
ICL_DIR       = _ROOT / "tables-tabicl/parquet"
CACHE_DIR     = _ROOT / "cache"

OUTPUT_DIR    = Path("coverage-results-pairwise-colhists")
WANDB_PROJECT = "tabicl-coverage-pairwise"

N_SAMPLE  = 25_000
MAX_ROWS  = 512
N_BINS    = 30
CORR_BINS = np.linspace(-1, 1, N_BINS + 1)
UNIQUENESS_THRESHOLD = 0.2

PAIRS = [
    ("icl", "fm"),
    ("t4",  "fm"),
    ("icl", "t4"),
]

_DATASET_LABELS = {
    "fm":  "Curated",
    "t4":  "Web-Scraped",
    "icl": "Synthetic",
}
_DATASET_COLORS = {
    "fm":  "#e67e22",
    "t4":  "#2980b9",
    "icl": "#27ae60",
}
_PAIR_COLORS = {
    "icl_vs_fm": "#2980b9",
    "t4_vs_fm":  "#e67e22",
    "icl_vs_t4": "#27ae60",
}
_FEAT_LABELS = {
    "col_hist":  "Column Histograms",
    "corr_hist": "Correlation Histograms",
}


def _cdf_to_density(hists: np.ndarray) -> np.ndarray:
    """Convert (N, 30) CDF histograms to relative densities (PMF) via finite differences."""
    return np.diff(hists, prepend=0.0, axis=1)


# ------------------------------------------------------------------
# Column histogram loaders  (compatible with rank_vs_coverage_columns)
# ------------------------------------------------------------------

def _build_col_index(parquet_dir: Path) -> list[tuple[Path, str]]:
    """Return flat list of (path, col_name) for every qualifying numeric column."""
    index = []
    for p in tqdm(sorted(parquet_dir.glob("*.parquet")),
                  desc=f"Indexing {parquet_dir.name}", leave=False):
        try:
            df = pd.read_parquet(p)
            num = df.select_dtypes(include=["float"])
            n = len(df)
            for col in num.columns:
                if n > 0 and num[col].nunique() / n >= UNIQUENESS_THRESHOLD:
                    index.append((p, col))
        except Exception:
            pass
    return index


def _compute_sampled_hists(
    col_index: list[tuple[Path, str]],
    sampled_idx: np.ndarray,
) -> np.ndarray:
    file_to_cols: dict[Path, list[tuple[int, str]]] = defaultdict(list)
    for rank_i, idx in enumerate(sampled_idx):
        path, col = col_index[idx]
        file_to_cols[path].append((rank_i, col))

    hists = np.zeros((len(sampled_idx), N_BINS))
    for path, items in tqdm(file_to_cols.items(), desc="Computing col histograms", leave=False):
        try:
            needed = [col for _, col in items]
            df = pd.read_parquet(path, columns=needed).iloc[:MAX_ROWS]
            for rank_i, col in items:
                if col in df.columns:
                    hists[rank_i] = get_histogram(df[col])
        except Exception:
            pass
    return hists


def _find_existing_cache(stem_pattern: str, cache_dir: Path) -> Path | None:
    """Return any existing .npz whose name matches stem_pattern (glob), or None."""
    matches = sorted(cache_dir.glob(stem_pattern + "*.npz"))
    return matches[0] if matches else None


def load_col_hists(
    parquet_dir: Path,
    n: int,
    rng: np.random.Generator,
    cache_path: Path,
) -> np.ndarray:
    """Return (n, 30) array of column histograms from parquet_dir, with caching.

    If cache_path does not exist, also checks for any existing
    ``{stem}_*.npz`` file in the same directory (compatible with caches
    produced by rank_vs_coverage_columns.py which embed n_talent in the name).
    """
    # Try requested cache first, then any compatible existing cache
    for cp in [cache_path, _find_existing_cache(cache_path.stem.rsplit("_", 1)[0] + "_",
                                                  cache_path.parent)]:
        if cp is not None and cp.exists():
            arr = np.load(cp)["hists"]
            if len(arr) >= n:
                idx = rng.choice(len(arr), size=n, replace=False)
                print(f"  Loaded {n} col hists from cache ({cp.name})")
                return arr[idx]
            print(f"  Cache {cp.name} has {len(arr)} cols (< {n} requested), using all")
            return arr

    index = _build_col_index(parquet_dir)
    if not index:
        print(f"  WARNING: no qualifying numeric columns in {parquet_dir}")
        return np.zeros((0, N_BINS))

    print(f"  Total qualifying columns in {parquet_dir.name}: {len(index)}")
    sampled_idx = rng.choice(len(index), size=min(n, len(index)), replace=False)
    hists = _compute_sampled_hists(index, sampled_idx)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, hists=hists)
    print(f"  Cached {len(hists)} col hists -> {cache_path}")
    return hists


# ------------------------------------------------------------------
# Correlation histogram loaders  (compatible with correlation_coverage)
# ------------------------------------------------------------------

def _table_corr_hist(df: pd.DataFrame) -> np.ndarray | None:
    """Return 30-bin normalised cumulative correlation histogram for one table."""
    num = df.select_dtypes(include=["float"]).iloc[:MAX_ROWS]
    n = len(num)
    if n == 0:
        return None
    num = num[[c for c in num.columns if num[c].nunique() / n >= UNIQUENESS_THRESHOLD]]
    if num.shape[1] < 2:
        return None
    corr = num.corr().values
    i, j = np.triu_indices(corr.shape[0], k=1)
    vals = corr[i, j]
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None
    counts, _ = np.histogram(vals, bins=CORR_BINS)
    total = counts.sum()
    if total == 0:
        return None
    return np.cumsum(counts) / total


def load_corr_hists(
    parquet_dir: Path,
    n: int,
    rng: np.random.Generator,
    cache_path: Path,
) -> np.ndarray:
    """Return (n, 30) array of correlation histograms from parquet_dir, with caching.

    Falls back to any compatible ``{stem}_*.npz`` in the same directory
    (compatible with caches produced by correlation_coverage.py).
    """
    for cp in [cache_path, _find_existing_cache(cache_path.stem.rsplit("_", 1)[0] + "_",
                                                  cache_path.parent)]:
        if cp is not None and cp.exists():
            arr = np.load(cp)["hists"]
            if len(arr) >= n:
                idx = rng.choice(len(arr), size=n, replace=False)
                print(f"  Loaded {n} corr hists from cache ({cp.name})")
                return arr[idx]
            print(f"  Cache {cp.name} has {len(arr)} hists (< {n} requested), using all")
            return arr

    paths = rng.permutation(np.array(sorted(parquet_dir.glob("*.parquet")), dtype=object))
    hists, errors = [], 0
    for p in tqdm(paths, desc=f"Corr hists {parquet_dir.name}"):
        try:
            h = _table_corr_hist(pd.read_parquet(p))
            if h is not None:
                hists.append(h)
        except Exception:
            errors += 1
        if len(hists) >= n:
            break

    print(f"  Computed {len(hists)} corr hists from {parquet_dir.name} ({errors} errors)")
    arr = np.array(hists)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, hists=arr)
    print(f"  Cached -> {cache_path}")
    return arr


# ------------------------------------------------------------------
# Figure helpers
# ------------------------------------------------------------------

def _collect_legend_below(fig, axes) -> None:
    nrows, ncols = axes.shape
    for r in range(nrows):
        for c in range(ncols):
            if r < nrows - 1:
                axes[r, c].set_xlabel("")
            if c > 0:
                axes[r, c].set_ylabel("")
    all_handles, all_labels = [], []
    seen: set[str] = set()
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                all_handles.append(h)
                all_labels.append(l)
                seen.add(l)
        if ax.get_legend():
            ax.get_legend().remove()
    plt.tight_layout()
    fig.legend(all_handles, all_labels, loc="lower center", ncols=3,
               bbox_to_anchor=(0.5, 0), fontsize=7)
    plt.subplots_adjust(bottom=0.14)


# ------------------------------------------------------------------
# Per-pair combined figures (columns = feature types)
# ------------------------------------------------------------------

def _plot_combined_per_pair(pair_key: str, feat_data: dict) -> None:
    """2-column combined figures for a single pair (col = col_hist / corr_hist)."""
    feat_types = ["col_hist", "corr_hist"]
    tag = pair_key

    with mpl.rc_context(_PAPER_RC):
        # Bootstrap AUC
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
        for ax, ft in zip(axes, feat_types):
            d = feat_data[ft]["boot"]
            _draw_bootstrap_auc(ax, d["boot_aucs"], d["ci_lo"], d["ci_hi"])
            ax.set_title(_FEAT_LABELS[ft])
        plt.tight_layout()
        _save_and_log(fig, "combined_bootstrap_auc", None, tag)

        # Recall
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0))
        for col_i, ft in enumerate(feat_types):
            d = feat_data[ft]["recall"]
            _draw_coverage_dists(axes[0, col_i], d["real_to_synth"], d["synth_internal"],
                                 d["threshold"], label_ref=d["label_ref"], label_cand=d["label_cand"])
            _draw_coverage_curve(axes[1, col_i], d["real_to_synth"], d["threshold"],
                                 d["covered_fraction"], label_ref=d["label_ref"])
            axes[0, col_i].set_title(_FEAT_LABELS[ft])
            axes[1, col_i].set_title("")
        _collect_legend_below(fig, axes)
        _save_and_log(fig, "combined_recall", None, tag)

        # Precision
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0))
        for col_i, ft in enumerate(feat_types):
            d = feat_data[ft]["precision"]
            _draw_coverage_dists(axes[0, col_i], d["real_to_synth"], d["synth_internal"],
                                 d["threshold"], label_ref=d["label_ref"], label_cand=d["label_cand"])
            _draw_coverage_curve(axes[1, col_i], d["real_to_synth"], d["threshold"],
                                 d["covered_fraction"], label_ref=d["label_ref"])
            axes[0, col_i].set_title(_FEAT_LABELS[ft])
            axes[1, col_i].set_title("")
        _collect_legend_below(fig, axes)
        _save_and_log(fig, "combined_precision", None, tag)

        # Recall/precision CDF overlay
        fig, (ax_rec, ax_pre) = plt.subplots(1, 2, figsize=(6.5, 2.4))
        ft_colors = {"col_hist": "#2980b9", "corr_hist": "#e67e22"}
        for ft in feat_types:
            d_rec = feat_data[ft]["recall"]
            s = np.sort(d_rec["real_to_synth"])
            ax_rec.plot(s, np.linspace(0, 1, len(s)), color=ft_colors[ft],
                        label=_FEAT_LABELS[ft], linewidth=0.9)
            ax_rec.axvline(d_rec["threshold"], color=ft_colors[ft], linestyle="--",
                           linewidth=0.6, alpha=0.7)
            d_pre = feat_data[ft]["precision"]
            s = np.sort(d_pre["real_to_synth"])
            ax_pre.plot(s, np.linspace(0, 1, len(s)), color=ft_colors[ft],
                        label=_FEAT_LABELS[ft], linewidth=0.9)
            ax_pre.axvline(d_pre["threshold"], color=ft_colors[ft], linestyle="--",
                           linewidth=0.6, alpha=0.7)
        ax_rec.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
        ax_rec.set_ylabel("CDF (Recall)")
        ax_rec.set_title("Recall")
        ax_rec.legend()
        ax_pre.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
        ax_pre.set_ylabel("CDF (Precision)")
        ax_pre.set_title("Precision")
        ax_pre.legend()
        plt.tight_layout()
        _save_and_log(fig, "combined_recall_precision_curves", None, tag)

        # PCA
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
        for ax, ft in zip(axes, feat_types):
            d = feat_data[ft]["pca"]
            _draw_pca(ax, d["X_2d"], d["labels"], d["var"], "",
                      label_ref=d["label_ref"], label_cand=d["label_cand"])
        plt.tight_layout()
        _save_and_log(fig, "combined_pca", None, tag)


# ------------------------------------------------------------------
# Cross-pair combined figures (columns = pairs, one per feature type)
# ------------------------------------------------------------------

def _plot_combined_across_pairs(
    all_pair_data: dict,
    feat_type: str,
    summary_run,
) -> None:
    """1×3-col figures where each column is a pair, for a single feature type."""
    pairs_ordered = [f"{a}_vs_{b}" for a, b in PAIRS]
    pairs_display = {
        f"{a}_vs_{b}": f"{_DATASET_LABELS[a]} vs {_DATASET_LABELS[b]}"
        for a, b in PAIRS
    }
    tag = f"all_pairs_{feat_type}"

    with mpl.rc_context(_PAPER_RC):
        # Bootstrap AUC
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
        for ax, pk in zip(axes, pairs_ordered):
            d = all_pair_data[pk][feat_type]["boot"]
            _draw_bootstrap_auc(ax, d["boot_aucs"], d["ci_lo"], d["ci_hi"])
            ax.set_title(pairs_display[pk])
        plt.tight_layout()
        _save_and_log(fig, f"xpair_{feat_type}_bootstrap_auc", summary_run, tag)

        # Recall
        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
        for col_i, pk in enumerate(pairs_ordered):
            d = all_pair_data[pk][feat_type]["recall"]
            _draw_coverage_dists(axes[0, col_i], d["real_to_synth"], d["synth_internal"],
                                 d["threshold"], label_ref=d["label_ref"], label_cand=d["label_cand"])
            _draw_coverage_curve(axes[1, col_i], d["real_to_synth"], d["threshold"],
                                 d["covered_fraction"], label_ref=d["label_ref"])
            axes[0, col_i].set_title(pairs_display[pk])
            axes[1, col_i].set_title("")
        _collect_legend_below(fig, axes)
        _save_and_log(fig, f"xpair_{feat_type}_recall", summary_run, tag)

        # Precision
        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
        for col_i, pk in enumerate(pairs_ordered):
            d = all_pair_data[pk][feat_type]["precision"]
            _draw_coverage_dists(axes[0, col_i], d["real_to_synth"], d["synth_internal"],
                                 d["threshold"], label_ref=d["label_ref"], label_cand=d["label_cand"])
            _draw_coverage_curve(axes[1, col_i], d["real_to_synth"], d["threshold"],
                                 d["covered_fraction"], label_ref=d["label_ref"])
            axes[0, col_i].set_title(pairs_display[pk])
            axes[1, col_i].set_title("")
        _collect_legend_below(fig, axes)
        _save_and_log(fig, f"xpair_{feat_type}_precision", summary_run, tag)

        # Overlaid recall/precision CDFs
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
        ax_rec, ax_pre = axes
        for pk in pairs_ordered:
            c = _PAIR_COLORS[pk]
            lbl = pairs_display[pk]
            d_rec = all_pair_data[pk][feat_type]["recall"]
            s = np.sort(d_rec["real_to_synth"])
            ax_rec.plot(s, np.linspace(0, 1, len(s)), color=c, label=lbl, linewidth=0.9)
            ax_rec.axvline(d_rec["threshold"], color=c, linestyle="--", linewidth=0.6, alpha=0.7)
            d_pre = all_pair_data[pk][feat_type]["precision"]
            s = np.sort(d_pre["real_to_synth"])
            ax_pre.plot(s, np.linspace(0, 1, len(s)), color=c, label=lbl, linewidth=0.9)
            ax_pre.axvline(d_pre["threshold"], color=c, linestyle="--", linewidth=0.6, alpha=0.7)
        ax_rec.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
        ax_rec.set_ylabel("CDF (Recall)")
        ax_rec.set_title("Recall")
        ax_rec.legend()
        ax_pre.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
        ax_pre.set_ylabel("CDF (Precision)")
        ax_pre.set_title("Precision")
        ax_pre.legend()
        plt.tight_layout()
        _save_and_log(fig, f"xpair_{feat_type}_recall_precision_curves", summary_run, tag)

        # PCA
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
        for ax, pk in zip(axes, pairs_ordered):
            d = all_pair_data[pk][feat_type]["pca"]
            _draw_pca(ax, d["X_2d"], d["labels"], d["var"], "",
                      label_ref=d["label_ref"], label_cand=d["label_cand"])
        plt.tight_layout()
        _save_and_log(fig, f"xpair_{feat_type}_pca", summary_run, tag)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(tables_dir: Path, icl_dir: Path, n_sample: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _ca.OUTPUT_DIR = OUTPUT_DIR

    rng = np.random.default_rng(RANDOM_STATE)

    # ── Load column histograms ───────────────────────────────────────
    print("\n=== Loading column histograms ===")
    col_hists = {}
    for name, d in [("fm", tables_dir), ("t4", T4_DIR), ("icl", icl_dir)]:
        cache = CACHE_DIR / f"{name}_col_hists_{n_sample}.npz"
        col_hists[name] = _cdf_to_density(load_col_hists(d, n_sample, rng, cache))
        print(f"  {name}: {col_hists[name].shape}")

    # ── Load correlation histograms ──────────────────────────────────
    print("\n=== Loading correlation histograms ===")
    corr_hists = {}
    for name, d in [("fm", tables_dir), ("t4", T4_DIR), ("icl", icl_dir)]:
        cache = CACHE_DIR / f"{name}_corr_hists_{n_sample}.npz"
        corr_hists[name] = _cdf_to_density(load_corr_hists(d, n_sample, rng, cache))
        print(f"  {name}: {corr_hists[name].shape}")

    feat_sets = {"col_hist": col_hists, "corr_hist": corr_hists}

    # ── Run analyses ─────────────────────────────────────────────────
    all_pair_data: dict[str, dict] = {}

    for a_name, b_name in PAIRS:
        pair_key = f"{a_name}_vs_{b_name}"
        all_pair_data[pair_key] = {}

        print(f"\n{'='*60}")
        print(f"  Pair: {pair_key}")
        print(f"{'='*60}")

        for feat_type, hists in feat_sets.items():
            a_full = hists[a_name]
            b_full = hists[b_name]
            n_pair = min(len(a_full), len(b_full))
            a_m = a_full[rng.choice(len(a_full), size=n_pair, replace=False)]
            b_m = b_full[rng.choice(len(b_full), size=n_pair, replace=False)]

            tag = f"{pair_key}_{feat_type}"
            print(f"\n  ── feat={feat_type}  shape=({n_pair}, {a_m.shape[1]})  [{tag}] ──")

            run = None
            if _WANDB_AVAILABLE and WANDB_PROJECT:
                run = wandb.init(
                    project=WANDB_PROJECT,
                    name=tag,
                    config={
                        "pair": pair_key, "A": a_name, "B": b_name,
                        "feat_type": feat_type,
                        "n_A": len(a_m), "n_B": len(b_m),
                        "n_features": a_m.shape[1],
                    },
                    reinit=True,
                )

            # B = reference, A = candidate  (recall: does A cover B?)
            _, boot_data      = analyse_bootstrap_auc(b_m, a_m, run, tag)
            _, recall_data    = analyse_coverage(b_m, a_m, run, tag,
                                                 label_ref=_DATASET_LABELS[b_name],
                                                 label_cand=_DATASET_LABELS[a_name])
            prec_result, precision_data = analyse_coverage(a_m, b_m, None,
                                                            tag + "_precision",
                                                            label_ref=_DATASET_LABELS[a_name],
                                                            label_cand=_DATASET_LABELS[b_name])
            if run is not None:
                run.log({f"precision/{k}": v for k, v in prec_result.items()})
            _, pca_data       = analyse_pca(b_m, a_m, run, tag,
                                            label_ref=_DATASET_LABELS[b_name],
                                            label_cand=_DATASET_LABELS[a_name])

            all_pair_data[pair_key][feat_type] = {
                "boot": boot_data, "recall": recall_data,
                "precision": precision_data, "pca": pca_data,
            }

            if run is not None:
                run.finish()

        # Combined per-pair figures (columns = col_hist / corr_hist)
        print(f"\n  ── Combined figures for {pair_key} ──")
        _plot_combined_per_pair(pair_key, all_pair_data[pair_key])

    # ── Cross-pair combined figures ───────────────────────────────────
    print("\n── Cross-pair combined figures ──")
    for feat_type in feat_sets:
        summary_run = None
        if _WANDB_AVAILABLE and WANDB_PROJECT:
            summary_run = wandb.init(
                project=WANDB_PROJECT,
                name=f"combined_all_pairs_{feat_type}",
                config={"summary": True, "feat_type": feat_type,
                        "pairs": [f"{a}_vs_{b}" for a, b in PAIRS]},
                reinit=True,
            )
        _plot_combined_across_pairs(all_pair_data, feat_type, summary_run)
        if summary_run is not None:
            summary_run.finish()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=FM_DIR,
                        help="Directory of tabfm .parquet files (default: %(default)s)")
    parser.add_argument("--icl-dir", type=Path, default=ICL_DIR,
                        help="Directory of tabicl .parquet files (default: %(default)s)")
    parser.add_argument("--n-sample", type=int, default=N_SAMPLE,
                        help="Number of samples per dataset (default: %(default)s)")
    args = parser.parse_args()
    main(args.tables_dir, args.icl_dir, args.n_sample)
