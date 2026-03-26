"""Ablation study: effect of k (neighbours) and threshold quantile on coverage.

For each of the three dataset pairs (icl vs fm, t4 vs fm, icl vs t4) and using
the full feature set only, sweeps a grid of:

    KNN_K                      : [1, 3, 5, 10, 20, 50]
    COVERAGE_THRESHOLD_QUANTILE: [0.80, 0.85, 0.90, 0.95, 0.99]

and records the resulting recall and precision covered_fraction.

Outputs
-------
coverage-results-ablation/ablation_results.csv   — full numeric grid
coverage-results-ablation/ablation_recall_<pair>.png
coverage-results-ablation/ablation_precision_<pair>.png
  Heatmaps: rows = k, columns = quantile, cell value = covered_fraction.

A combined 3-column figure (one column per pair) is also produced for each
direction (recall / precision).

Usage
-----
    python analysis/coverage_ablation.py [--tables-dir tables-tabfm/parquet]
                                         [--icl-dir   tables-tabicl/parquet]
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import analysis.coverage_analysis as _ca
from analysis.coverage_analysis import (
    _PAPER_RC, RANDOM_STATE, MAX_ROWS,
    _select_features as _ca_select_features,
    FEATURE_NAMES,
)
from tabdistances.discriminator.features import df_to_vec

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
TABLES_DIR = Path("tables-tabfm/parquet")
TABICL_DIR = Path("tables-tabicl/parquet")
T4_DIR     = Path("tables-t4/parquet")
T4_CACHE   = Path("cache/real_feats_t4_5000_512_v2.npz")
ICL_CACHE  = Path("cache/real_feats_icl_25000_512_v2.npz")

N_SAMPLE   = 25000
OUTPUT_DIR = Path("coverage-results-ablation")

# Ablation grid
K_VALUES         = [1, 3, 5, 10, 20, 50]
QUANTILE_VALUES  = [0.80, 0.85, 0.90, 0.95, 0.99]

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


# ------------------------------------------------------------------
# Dataset loaders (mirrors pairwise-tabes.py)
# ------------------------------------------------------------------
def _load_fm(tables_dir: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    paths = np.array(sorted(tables_dir.glob("*.parquet")))
    if not len(paths):
        print(f"ERROR: no .parquet files in {tables_dir}")
        sys.exit(1)
    rng.shuffle(paths)
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
        if len(feats) >= n:
            break
    print(f"  Loaded {len(feats)} tabfm tables ({errors} skipped)")
    return np.array(feats)


def _load_t4(cache_path: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    if not cache_path.exists():
        paths = np.array(sorted(T4_DIR.glob("*.parquet")))
        if not len(paths):
            print(f"ERROR: no .parquet files in {T4_DIR}")
            sys.exit(1)
        rng.shuffle(paths)
        feats, errors = [], 0
        for p in tqdm(paths, desc="featurising t4 tables", unit="table"):
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
        print(f"  Loaded {len(feats)} t4 tables ({errors} skipped)")
        arr = np.array(feats)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, feats=arr)
        print(f"  Cached -> {cache_path}")
    else:
        arr = np.load(cache_path)["feats"]
        idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
        arr = arr[idx]
    return arr


def _load_icl(icl_dir: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    if not ICL_CACHE.exists():
        paths = sorted(icl_dir.glob("*.parquet"))
        if not paths:
            print(f"ERROR: no .parquet files in {icl_dir}")
            sys.exit(1)
        rng.shuffle(paths := np.array(paths))
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
        arr = np.array(feats)
        ICL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ICL_CACHE, feats=arr)
        print(f"  Cached -> {ICL_CACHE}")
    else:
        arr = np.load(ICL_CACHE)["feats"]
        idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
        arr = arr[idx]
    return arr


# ------------------------------------------------------------------
# Feature selection: full mode + n_cols prepended, missing_rate dropped
# ------------------------------------------------------------------
def _select_full(feats: np.ndarray) -> tuple[np.ndarray, list[str]]:
    arr, names = _ca_select_features(feats, "full")
    if "missing_rate" in names:
        i = names.index("missing_rate")
        arr = np.delete(arr, i, axis=1)
        names = [n for n in names if n != "missing_rate"]
    arr   = np.concatenate([feats[:, [0]], arr], axis=1)
    names = ["n_cols"] + names
    return arr, names


# ------------------------------------------------------------------
# Parameterised coverage computation (no global state)
# ------------------------------------------------------------------
def _coverage_at(
    ref: np.ndarray,
    cand: np.ndarray,
    k: int,
    quantile: float,
) -> dict:
    """Compute coverage of *ref* by *cand* for a specific (k, quantile) pair.

    Returns a dict with keys: threshold, covered_fraction, median_dist.
    Mirrors the normalisation and k-NN logic in coverage_analysis.analyse_coverage.
    """
    lo  = np.minimum(ref.min(0), cand.min(0))
    hi  = np.maximum(ref.max(0), cand.max(0))
    rng = hi - lo
    rng[rng == 0] = 1.0
    ref_n  = (ref  - lo) / rng
    cand_n = (cand - lo) / rng

    # Within-candidate distances (for threshold)
    nn_cc = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(cand_n)
    dists_cc, _ = nn_cc.kneighbors(cand_n)
    cand_internal = dists_cc[:, 1:]            # exclude self; shape (n_cand, k)
    threshold = np.percentile(cand_internal.mean(axis=1), quantile * 100)

    # Reference → candidate distances
    nn_rc = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(cand_n)
    dists_rc, _ = nn_rc.kneighbors(ref_n)
    ref_to_cand = dists_rc.mean(axis=1)

    covered_fraction = 1.0 - (ref_to_cand > threshold).mean()
    return {
        "threshold":        float(threshold),
        "covered_fraction": float(covered_fraction),
        "median_dist":      float(np.median(ref_to_cand)),
    }


# ------------------------------------------------------------------
# Heatmap helpers
# ------------------------------------------------------------------
def _heatmap(ax, matrix: np.ndarray, row_labels, col_labels,
             title: str, vmin: float = 0.0, vmax: float = 1.0) -> None:
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([f"{q:.2f}" for q in col_labels], rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([str(k) for k in row_labels])
    ax.set_xlabel("Threshold quantile")
    ax.set_ylabel("k (neighbours)")
    ax.set_title(title)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            text_color = "black" if 0.25 < val < 0.75 else "white"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=text_color)
    return im


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(tables_dir: Path, icl_dir: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    # ── Load datasets ──────────────────────────────────────────────
    print("\n=== Loading datasets ===")
    fm_feats  = _load_fm(tables_dir, N_SAMPLE, rng)
    t4_feats  = _load_t4(T4_CACHE,   N_SAMPLE, rng)
    icl_feats = _load_icl(icl_dir,   N_SAMPLE, rng)

    datasets = {"fm": fm_feats, "t4": t4_feats, "icl": icl_feats}
    for name, arr in datasets.items():
        print(f"  {name}: {arr.shape}")

    # ── Select full features for each dataset ──────────────────────
    print("\n=== Selecting full features ===")
    feat_data: dict[str, np.ndarray] = {}
    for name, arr in datasets.items():
        feat_data[name], feat_names = _select_full(arr)
        print(f"  {name}: {feat_data[name].shape}  ({len(feat_names)} features)")

    # ── Ablation grid ──────────────────────────────────────────────
    records = []
    # Pre-sample balanced subsets per pair (fixed across the entire grid for fair comparison)
    pair_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for a_name, b_name in PAIRS:
        n_pair = min(len(feat_data[a_name]), len(feat_data[b_name]))
        a_idx  = rng.choice(len(feat_data[a_name]), size=n_pair, replace=False)
        b_idx  = rng.choice(len(feat_data[b_name]), size=n_pair, replace=False)
        pair_arrays[f"{a_name}_vs_{b_name}"] = (
            feat_data[a_name][a_idx],
            feat_data[b_name][b_idx],
        )
        print(f"  {a_name}_vs_{b_name}: n={n_pair}")

    total_evals = len(PAIRS) * len(K_VALUES) * len(QUANTILE_VALUES)
    print(f"\n=== Ablation grid: {len(K_VALUES)} k × {len(QUANTILE_VALUES)} quantiles"
          f" × {len(PAIRS)} pairs = {total_evals} evaluations ===")

    with tqdm(total=total_evals, desc="ablation") as pbar:
        for a_name, b_name in PAIRS:
            pair_key = f"{a_name}_vs_{b_name}"
            a_feats, b_feats = pair_arrays[pair_key]

            for k in K_VALUES:
                for q in QUANTILE_VALUES:
                    # Recall: how well does A (candidate) cover B (reference)?
                    recall    = _coverage_at(b_feats, a_feats, k, q)
                    # Precision: how well does B (reference) cover A (candidate)?
                    precision = _coverage_at(a_feats, b_feats, k, q)

                    records.append({
                        "pair":             pair_key,
                        "a":                a_name,
                        "b":                b_name,
                        "k":                k,
                        "quantile":         q,
                        "recall_covered":   recall["covered_fraction"],
                        "recall_threshold": recall["threshold"],
                        "recall_median":    recall["median_dist"],
                        "prec_covered":     precision["covered_fraction"],
                        "prec_threshold":   precision["threshold"],
                        "prec_median":      precision["median_dist"],
                    })
                    pbar.update(1)

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved -> {csv_path}")

    # ── Heatmaps: 3 rows (pairs) × 2 cols (recall | precision) ───────
    pairs_ordered = [f"{a}_vs_{b}" for a, b in PAIRS]
    pair_titles   = {f"{a}_vs_{b}": f"{_DATASET_LABELS[a]} vs {_DATASET_LABELS[b]}"
                     for a, b in PAIRS}

    def _build_mat(pair_key: str, col: str) -> np.ndarray:
        sub = df[df["pair"] == pair_key]
        mat = np.zeros((len(K_VALUES), len(QUANTILE_VALUES)))
        for ri, k in enumerate(K_VALUES):
            for ci, q in enumerate(QUANTILE_VALUES):
                val = sub.loc[(sub["k"] == k) & (sub["quantile"] == q), col]
                mat[ri, ci] = val.iloc[0] if len(val) else np.nan
        return mat

    with mpl.rc_context(_PAPER_RC):
        n_rows = len(PAIRS)
        fig, axes = plt.subplots(n_rows, 2, figsize=(6.5, 2.8 * n_rows))
        ims = []
        for row_i, pair_key in enumerate(pairs_ordered):
            for col_i, (direction, col) in enumerate(
                [("Recall", "recall_covered"), ("Precision", "prec_covered")]
            ):
                ax  = axes[row_i, col_i]
                mat = _build_mat(pair_key, col)
                # Only show axis labels on the edges to reduce clutter
                xlabel = "Threshold quantile" if row_i == n_rows - 1 else ""
                ylabel = "k (neighbours)"     if col_i == 0           else ""
                im = _heatmap(ax, mat, K_VALUES, QUANTILE_VALUES,
                              f"{direction} — {pair_titles[pair_key]}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ims.append(im)

        # Single shared colorbar on the right
        fig.subplots_adjust(right=0.88, hspace=0.55, wspace=0.25)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
        fig.colorbar(ims[-1], cax=cbar_ax, label="Covered fraction")

        fname = OUTPUT_DIR / "ablation_combined.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved -> {fname}")

    # ── Print summary table ────────────────────────────────────────
    print("\n=== Summary (recall covered_fraction) ===")
    pivot = df.pivot_table(
        values="recall_covered", index="k", columns=["pair", "quantile"], aggfunc="first"
    )
    print(pivot.to_string())

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR,
                        help="Directory of tabfm .parquet files (default: %(default)s)")
    parser.add_argument("--icl-dir", type=Path, default=TABICL_DIR,
                        help="Directory of tabicl .parquet files (default: %(default)s)")
    args = parser.parse_args()
    main(args.tables_dir, args.icl_dir)
