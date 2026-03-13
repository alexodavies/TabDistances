"""Pairwise distribution comparison: tabfm / tabICL prior / t4.

Loads or generates feature vectors for three datasets and runs all five
coverage analyses for every pairwise combination:

    icl vs fm   — does the tabICL prior cover the tabfm distribution?
    t4  vs fm   — how similar are the t4 and tabfm real distributions?
    icl vs t4   — does the tabICL prior cover the t4 distribution?

For each pair the first named dataset is treated as "A" (the candidate /
synthetic side) and the second as "B" (the reference / real side).

Analyses per pair × feature mode
  1. Per-feature KS tests
  2. Feature importance (XGBoost discriminator gain)
  3. Bootstrap AUC CI
  4. Coverage (k-NN)
  5. PCA projection

When all three feature modes are run (the default) a combined 1×3-column
figure is produced per pair (columns = modes).  A second set of combined
figures shows all three pairs side-by-side (columns = pairs) for the "full"
feature mode.

All combined figures are sized for US letter-paper inclusion (6.5 in wide).

Usage
-----
    python coverage_analysis_pairwise.py [--tables-dir tables-tabfm/parquet]
                                         [--mode full scalar_only hist_only]
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

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

# Patch coverage_analysis OUTPUT_DIR before it is used by imported functions
import coverage_analysis as _ca
OUTPUT_DIR = Path("coverage-results-pairwise")
_ca.OUTPUT_DIR = OUTPUT_DIR

from coverage_analysis import (
    FEATURE_NAMES, FEATURE_MODES,
    _select_features as _ca_select_features,
    _PAPER_RC, KNN_K, COVERAGE_THRESHOLD_QUANTILE,
    N_BOOT, BOOT_ALPHA, TEST_FRAC, RANDOM_STATE, XGB_DEVICE,
    analyse_per_feature, analyse_bootstrap_auc, analyse_coverage,
    analyse_feature_importance, analyse_pca,
    _draw_per_feature_ks, _draw_bootstrap_auc,
    _draw_coverage_dists, _draw_coverage_curve,
    _draw_feature_importance, _draw_pca,
    _save_and_log,
)
from tabdistances.discriminator.features import df_to_vec


def _select_features(feats: np.ndarray, mode: str) -> tuple[np.ndarray, list[str]]:
    """Wrapper around coverage_analysis._select_features that drops missing_rate."""
    arr, names = _ca_select_features(feats, mode)
    if "missing_rate" in names:
        i = names.index("missing_rate")
        arr = np.delete(arr, i, axis=1)
        names = [n for n in names if n != "missing_rate"]
    return arr, names


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
TABLES_DIR    = Path("tables-tabfm/parquet")
TABICL_DIR    = Path("tables-tabicl/parquet")
T4_CACHE      = Path("cache/real_feats_t4_5000_512_v2.npz")

N_SAMPLE      = 1000     # samples drawn from each dataset for analysis
MAX_ROWS      = 512
WANDB_PROJECT = "tabicl-coverage-pairwise"

# Pairs: (A_name, B_name) — A is "candidate", B is "reference"
PAIRS = [
    ("icl", "fm"),
    ("t4",  "fm"),
    ("icl", "t4"),
]

_DATASET_LABELS = {
    "fm":  "Collated",
    "t4":  "Web-Scraped",
    "icl": "Synthetic",
}

_PAIR_COLORS = {
    "icl_vs_fm": "#2980b9",
    "t4_vs_fm":  "#e67e22",
    "icl_vs_t4": "#27ae60",
}
_MODE_COLORS = {
    "full":        "#2980b9",
    "scalar_only": "#e67e22",
    "hist_only":   "#27ae60",
}
_MODE_LABELS = {
    "full":        "All Feats.",
    "scalar_only": "Aggregates",
    "hist_only":   "Col. Hists.",
}
_DATASET_COLORS = {
    "fm":  "#e67e22",
    "t4":  "#2980b9",
    "icl": "#27ae60",
}


# ------------------------------------------------------------------
# Dataset loaders
# ------------------------------------------------------------------
def _load_fm(tables_dir: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    paths = sorted(tables_dir.glob("*.parquet"))
    if not paths:
        print(f"ERROR: no .parquet files in {tables_dir}")
        sys.exit(1)
    feats, errors = [], 0
    for p in tqdm(paths, desc="featurising tabfm tables", unit="table"):
        try:
            df = pd.read_parquet(p).select_dtypes(include=[float, int, "number"])
            if df.shape[0] == 0 or df.shape[1] == 0:
                errors += 1
                continue
            feats.append(df_to_vec(df.iloc[:MAX_ROWS]))
        except Exception:
            errors += 1
    print(f"  Loaded {len(feats)} tabfm tables ({errors} skipped)")
    arr = np.array(feats)
    idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
    return arr[idx]


def _load_t4(cache_path: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    if not cache_path.exists():
        print(f"ERROR: t4 cache not found at {cache_path}. Run basic-grid-search.py first.")
        sys.exit(1)
    arr = np.load(cache_path)["feats"]
    idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
    return arr[idx]


def _load_icl(tabicl_dir: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    paths = sorted(tabicl_dir.glob("*.parquet"))
    if not paths:
        print(f"ERROR: no .parquet files in {tabicl_dir}")
        sys.exit(1)
    rng.shuffle(paths := np.array(paths))  # random order so the sample is unbiased
    feats, errors = [], 0
    for p in tqdm(paths, desc="featurising tabicl tables", unit="table"):
        try:
            df = pd.read_parquet(p).select_dtypes(include=[float, int, "number"])
            if df.shape[0] == 0 or df.shape[1] == 0:
                errors += 1
                continue
            feats.append(df_to_vec(df.iloc[:MAX_ROWS]))
        except Exception:
            errors += 1
        if len(feats) >= n:
            break
    print(f"  Loaded {len(feats)} tabicl tables ({errors} skipped)")
    return np.array(feats)


# ------------------------------------------------------------------
# Figure helper
# ------------------------------------------------------------------
def _collect_legend_below(fig, axes) -> None:
    """Move all per-axes legends to a single figure legend below the grid (3 cols).
    Also strips x-labels from non-bottom rows and y-labels from non-left columns."""
    nrows, ncols = axes.shape
    for r in range(nrows):
        for c in range(ncols):
            if r < nrows - 1:
                axes[r, c].set_xlabel("")
            if c > 0:
                axes[r, c].set_ylabel("")
    all_handles, all_labels = [], []
    seen = set()
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
# Combined figures — per pair (columns = modes)
# ------------------------------------------------------------------
def _plot_combined_per_pair(pair_key: str, mode_data: dict, modes_ordered: list) -> None:
    """1×3-col combined figures for a single pair, one column per feature mode."""
    tag = pair_key

    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.5))
        for ax, mode in zip(axes, modes_ordered):
            _draw_per_feature_ks(ax, mode_data[mode]["ks"]["df_ks"], top_n=10)
            ax.set_title(_MODE_LABELS[mode])
        plt.tight_layout()
        _save_and_log(fig, "combined_per_feature_ks", None, tag)

        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
        for ax, mode in zip(axes, modes_ordered):
            d = mode_data[mode]["boot"]
            _draw_bootstrap_auc(ax, d["boot_aucs"], d["ci_lo"], d["ci_hi"])
            ax.set_title(_MODE_LABELS[mode])
        plt.tight_layout()
        _save_and_log(fig, "combined_bootstrap_auc", None, tag)

        d0 = mode_data[modes_ordered[0]]["cov"]
        fwd_title = f"{d0['label_ref']} covered by {d0['label_cand']}"
        d0r = mode_data[modes_ordered[0]]["cov_rev"]
        rev_title = f"{d0r['label_ref']} covered by {d0r['label_cand']}"

        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
        for col_i, mode in enumerate(modes_ordered):
            d = mode_data[mode]["cov"]
            _draw_coverage_dists(axes[0, col_i], d["real_to_synth"], d["synth_internal"],
                                 d["threshold"], label_ref=d["label_ref"], label_cand=d["label_cand"])
            _draw_coverage_curve(axes[1, col_i], d["real_to_synth"], d["threshold"],
                                 d["uncovered"], label_ref=d["label_ref"])
            axes[0, col_i].set_title(_MODE_LABELS[mode])
            axes[1, col_i].set_title("")
        _collect_legend_below(fig, axes)
        _save_and_log(fig, "combined_coverage", None, tag)

        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
        for col_i, mode in enumerate(modes_ordered):
            d = mode_data[mode]["cov_rev"]
            _draw_coverage_dists(axes[0, col_i], d["real_to_synth"], d["synth_internal"],
                                 d["threshold"], label_ref=d["label_ref"], label_cand=d["label_cand"])
            _draw_coverage_curve(axes[1, col_i], d["real_to_synth"], d["threshold"],
                                 d["uncovered"], label_ref=d["label_ref"])
            axes[0, col_i].set_title(_MODE_LABELS[mode])
            axes[1, col_i].set_title("")
        _collect_legend_below(fig, axes)
        _save_and_log(fig, "combined_coverage_rev", None, tag)

        fig, (ax_fwd, ax_rev) = plt.subplots(1, 2, figsize=(6.5, 2.4))
        for mode in modes_ordered:
            d = mode_data[mode]["cov"]
            sorted_d = np.sort(d["real_to_synth"])
            ax_fwd.plot(sorted_d, np.linspace(0, 1, len(sorted_d)),
                        color=_MODE_COLORS[mode], label=_MODE_LABELS[mode], linewidth=0.9)
            ax_fwd.axvline(d["threshold"], color=_MODE_COLORS[mode],
                           linestyle="--", linewidth=0.6, alpha=0.7)
        ax_fwd.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
        ax_fwd.set_ylabel(f"CDF of {d0['label_ref']} samples")
        ax_fwd.set_title(fwd_title)
        ax_fwd.legend()

        for mode in modes_ordered:
            d = mode_data[mode]["cov_rev"]
            sorted_d = np.sort(d["real_to_synth"])
            ax_rev.plot(sorted_d, np.linspace(0, 1, len(sorted_d)),
                        color=_MODE_COLORS[mode], label=_MODE_LABELS[mode], linewidth=0.9)
            ax_rev.axvline(d["threshold"], color=_MODE_COLORS[mode],
                           linestyle="--", linewidth=0.6, alpha=0.7)
        ax_rev.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
        ax_rev.set_ylabel(f"CDF of {d0r['label_ref']} samples")
        ax_rev.set_title(rev_title)
        ax_rev.legend()

        plt.tight_layout()
        _save_and_log(fig, "combined_coverage_curves", None, tag)

        fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.5))
        for ax, mode in zip(axes, modes_ordered):
            d = mode_data[mode]["imp"]
            _draw_feature_importance(ax, d["imp"], d["auc_val"], top_n=10)
            ax.set_xlabel("Normalised gain")
            ax.set_title(_MODE_LABELS[mode])
        plt.tight_layout()
        _save_and_log(fig, "combined_feature_importance", None, tag)

        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
        for ax, mode in zip(axes, modes_ordered):
            d = mode_data[mode]["pca"]
            _draw_pca(ax, d["X_2d"], d["labels"], d["var"], "",
                      label_ref=d["label_ref"], label_cand=d["label_cand"])
        plt.tight_layout()
        _save_and_log(fig, "combined_pca", None, tag)


# ------------------------------------------------------------------
# Combined figures — across pairs (columns = pairs, full mode only)
# ------------------------------------------------------------------
def _plot_combined_across_pairs(all_pair_data: dict, summary_run) -> None:
    """1×3-col figures where each column is a pair, using the 'full' feature mode."""
    pairs_ordered      = [f"{a}_vs_{b}" for a, b in PAIRS]
    pairs_display      = {f"{a}_vs_{b}": f"{_DATASET_LABELS[a]} vs {_DATASET_LABELS[b]}"
                          for a, b in PAIRS}
    mode = "full"
    tag  = "all_pairs"

    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.5))
        for ax, pk in zip(axes, pairs_ordered):
            _draw_per_feature_ks(ax, all_pair_data[pk][mode]["ks"]["df_ks"], top_n=10)
            ax.set_title(pairs_display[pk])
        plt.tight_layout()
        _save_and_log(fig, "xpair_per_feature_ks", summary_run, tag)

        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
        for ax, pk in zip(axes, pairs_ordered):
            d = all_pair_data[pk][mode]["boot"]
            _draw_bootstrap_auc(ax, d["boot_aucs"], d["ci_lo"], d["ci_hi"])
            ax.set_title(pairs_display[pk])
        plt.tight_layout()
        _save_and_log(fig, "xpair_bootstrap_auc", summary_run, tag)

        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
        for col_i, pk in enumerate(pairs_ordered):
            d = all_pair_data[pk][mode]["cov"]
            _draw_coverage_dists(axes[0, col_i], d["real_to_synth"], d["synth_internal"],
                                 d["threshold"], label_ref=d["label_ref"], label_cand=d["label_cand"])
            _draw_coverage_curve(axes[1, col_i], d["real_to_synth"], d["threshold"],
                                 d["uncovered"], label_ref=d["label_ref"])
            axes[0, col_i].set_title(pairs_display[pk])
            axes[1, col_i].set_title("")
        _collect_legend_below(fig, axes)
        _save_and_log(fig, "xpair_coverage", summary_run, tag)

        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
        for col_i, pk in enumerate(pairs_ordered):
            d = all_pair_data[pk][mode]["cov_rev"]
            _draw_coverage_dists(axes[0, col_i], d["real_to_synth"], d["synth_internal"],
                                 d["threshold"], label_ref=d["label_ref"], label_cand=d["label_cand"])
            _draw_coverage_curve(axes[1, col_i], d["real_to_synth"], d["threshold"],
                                 d["uncovered"], label_ref=d["label_ref"])
            axes[0, col_i].set_title(pairs_display[pk])
            axes[1, col_i].set_title("")
        _collect_legend_below(fig, axes)
        _save_and_log(fig, "xpair_coverage_rev", summary_run, tag)

        # Overlaid coverage curves — 2×3: row 0 = forward, row 1 = reverse; cols = feature modes
        all_modes_ordered = list(FEATURE_MODES.keys())
        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
        for col_i, feat_mode in enumerate(all_modes_ordered):
            for pk in pairs_ordered:
                d_fwd = all_pair_data[pk][feat_mode]["cov"]
                sorted_fwd = np.sort(d_fwd["real_to_synth"])
                axes[0, col_i].plot(sorted_fwd, np.linspace(0, 1, len(sorted_fwd)),
                                    color=_PAIR_COLORS[pk], label=pairs_display[pk],
                                    linewidth=0.9)
                axes[0, col_i].axvline(d_fwd["threshold"], color=_PAIR_COLORS[pk],
                                       linestyle="--", linewidth=0.6, alpha=0.7)

                d_rev = all_pair_data[pk][feat_mode]["cov_rev"]
                sorted_rev = np.sort(d_rev["real_to_synth"])
                axes[1, col_i].plot(sorted_rev, np.linspace(0, 1, len(sorted_rev)),
                                    color=_PAIR_COLORS[pk], label=pairs_display[pk],
                                    linewidth=0.9)
                axes[1, col_i].axvline(d_rev["threshold"], color=_PAIR_COLORS[pk],
                                       linestyle="--", linewidth=0.6, alpha=0.7)

            axes[0, col_i].set_title(_MODE_LABELS[feat_mode])
            axes[0, col_i].set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
            axes[0, col_i].set_ylabel("CDF (A covered by B)")
            axes[1, col_i].set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
            axes[1, col_i].set_ylabel("CDF (B covered by A)")
        _collect_legend_below(fig, axes)
        _save_and_log(fig, "xpair_coverage_curves", summary_run, tag)

        fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.5))
        for ax, pk in zip(axes, pairs_ordered):
            d = all_pair_data[pk][mode]["imp"]
            _draw_feature_importance(ax, d["imp"], d["auc_val"], top_n=10)
            ax.set_xlabel("Normalised gain")
            ax.set_title(pairs_display[pk])
        plt.tight_layout()
        _save_and_log(fig, "xpair_feature_importance", summary_run, tag)

        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
        for ax, pk in zip(axes, pairs_ordered):
            d = all_pair_data[pk][mode]["pca"]
            _draw_pca(ax, d["X_2d"], d["labels"], d["var"], "",
                      label_ref=d["label_ref"], label_cand=d["label_cand"])
        plt.tight_layout()
        _save_and_log(fig, "xpair_pca", summary_run, tag)


# ------------------------------------------------------------------
# Histogram feature visualisation
# ------------------------------------------------------------------
_HIST_MEAN_SLICE = slice(10, 40)   # hist_mean[0..29] in the raw feature vector
_HIST_STD_SLICE  = slice(40, 70)   # hist_std[0..29]


def _plot_hist_features(datasets: dict, run=None) -> None:
    """Mean ± 1 SD band plot of histogram features for each dataset."""
    bins = np.arange(30)
    with mpl.rc_context(_PAPER_RC):
        fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(6.5, 2.4))
        for name, arr in datasets.items():
            label = _DATASET_LABELS[name]
            color = _DATASET_COLORS[name]
            for ax, sl in [(ax_mean, _HIST_MEAN_SLICE), (ax_std, _HIST_STD_SLICE)]:
                vals = arr[:, sl]          # (n_tables, 30)
                mu   = vals.mean(axis=0)
                sd   = vals.std(axis=0)
                ax.plot(bins, mu, color=color, label=label, linewidth=0.9)
                ax.fill_between(bins, mu - sd, mu + sd, color=color, alpha=0.2)
        ax_mean.set_xlabel("Bin index")
        ax_mean.set_ylabel("Mean across tables")
        ax_mean.set_title("Col. histogram mean")
        ax_mean.legend()
        ax_std.set_xlabel("Bin index")
        ax_std.set_ylabel("Mean across tables")
        ax_std.set_title("Col. histogram std")
        ax_std.legend()
        plt.tight_layout()
        _save_and_log(fig, "hist_features", run, "all_datasets")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(tables_dir: Path, icl_dir: Path, modes: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    # ── Load all three datasets ──────────────────────────────────────
    print("\n=== Loading datasets ===")
    fm_feats  = _load_fm(tables_dir, N_SAMPLE, rng)
    t4_feats  = _load_t4(T4_CACHE, N_SAMPLE, rng)
    icl_feats = _load_icl(icl_dir, N_SAMPLE, rng)

    datasets = {"fm": fm_feats, "t4": t4_feats, "icl": icl_feats}
    for name, arr in datasets.items():
        print(f"  {name}: {arr.shape}")

    print("\n── Histogram feature distributions ──")
    hist_run = None
    if _WANDB_AVAILABLE and WANDB_PROJECT:
        hist_run = wandb.init(
            project=WANDB_PROJECT,
            name="hist_features",
            config={"summary": True, "datasets": list(datasets.keys())},
            reinit=True,
        )
    _plot_hist_features(datasets, hist_run)
    if hist_run is not None:
        hist_run.finish()

    # ── Run analyses for each pair × mode ───────────────────────────
    all_modes_run  = set(modes) == set(FEATURE_MODES.keys())
    modes_ordered  = [m for m in FEATURE_MODES if m in modes]
    all_pair_data: dict[str, dict] = {}

    for a_name, b_name in PAIRS:
        pair_key  = f"{a_name}_vs_{b_name}"
        a_full    = datasets[a_name]   # "synthetic" / candidate
        b_full    = datasets[b_name]   # "real"      / reference
        pair_mode_data: dict[str, dict] = {}

        print(f"\n{'='*60}")
        print(f"  Pair: {pair_key}  (A={a_name} vs B={b_name})")
        print(f"{'='*60}")

        for mode in modes:
            tag      = f"{pair_key}_{mode}"
            a_m, fn  = _select_features(a_full, mode)
            b_m, _   = _select_features(b_full, mode)
            print(f"\n  ── mode={mode}  features={a_m.shape[1]}  [{tag}] ──")

            run = None
            if _WANDB_AVAILABLE and WANDB_PROJECT:
                run = wandb.init(
                    project=WANDB_PROJECT,
                    name=f"coverage_{pair_key}_{mode}",
                    config={"pair": pair_key, "A": a_name, "B": b_name,
                            "feature_mode": mode, "n_features": a_m.shape[1],
                            "n_A": len(a_m), "n_B": len(b_m)},
                    reinit=True,
                )

            # A = candidate, B = reference
            _, ks_data   = analyse_per_feature(b_m, a_m, run, tag, fn)
            _, imp_data  = analyse_feature_importance(b_m, a_m, run, tag, fn)
            _, boot_data = analyse_bootstrap_auc(b_m, a_m, run, tag)
            _, cov_data          = analyse_coverage(b_m, a_m, run, tag,
                                                    label_ref=_DATASET_LABELS[b_name],
                                                    label_cand=_DATASET_LABELS[a_name])
            cov_rev_result, cov_rev_data = analyse_coverage(a_m, b_m, None, tag + "_rev",
                                                            label_ref=_DATASET_LABELS[a_name],
                                                            label_cand=_DATASET_LABELS[b_name])
            if run is not None:
                run.log({f"coverage_rev/{k}": v for k, v in cov_rev_result.items()})
            _, pca_data     = analyse_pca(b_m, a_m, run, tag,
                                          label_ref=_DATASET_LABELS[b_name],
                                          label_cand=_DATASET_LABELS[a_name])

            pair_mode_data[mode] = {
                "ks": ks_data, "imp": imp_data, "boot": boot_data,
                "cov": cov_data, "cov_rev": cov_rev_data, "pca": pca_data,
            }
            if run is not None:
                run.finish()

        all_pair_data[pair_key] = pair_mode_data

        # Combined per-pair figures (columns = modes)
        if all_modes_run:
            print(f"\n  ── Combined mode figures for {pair_key} ──")
            _plot_combined_per_pair(pair_key, pair_mode_data, modes_ordered)

    # ── Cross-pair combined figures (columns = pairs) ────────────────
    if all_modes_run:
        print("\n── Cross-pair combined figures (full mode) ──")
        summary_run = None
        if _WANDB_AVAILABLE and WANDB_PROJECT:
            summary_run = wandb.init(
                project=WANDB_PROJECT,
                name="combined_all_pairs",
                config={"summary": True, "pairs": [f"{a}_vs_{b}" for a, b in PAIRS]},
                reinit=True,
            )
        _plot_combined_across_pairs(all_pair_data, summary_run)
        if summary_run is not None:
            summary_run.finish()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR,
                        help="Directory of tabfm .parquet files (default: %(default)s)")
    parser.add_argument("--icl-dir", type=Path, default=TABICL_DIR,
                        help="Directory of tabicl .parquet files (default: %(default)s)")
    parser.add_argument("--mode", nargs="+", choices=list(FEATURE_MODES.keys()),
                        default=list(FEATURE_MODES.keys()),
                        help="Feature mode(s) to run (default: all)")
    args = parser.parse_args()
    main(args.tables_dir, args.icl_dir, args.mode)
