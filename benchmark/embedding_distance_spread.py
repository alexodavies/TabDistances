"""Embedding distance spread: scale-free statistics for the compression question.

Answers McLn W2 / AC point 2: is the weak observed correlation (|r| <= 0.138)
between embedding distance and TabICL rank real, or an artifact of the
embedding space compressing TALENT tables into a narrow band close to the
prior? Raw distances aren't comparable across feature spaces of different
dimensionality/scaling, so this reports dimensionless quantities for every
feature space side by side:

  - CV of the TALENT -> prior mean 5-NN distance (std / mean)
  - IQR / median of that same distribution (robust companion to CV)
  - ratio of median TALENT -> prior distance to median within-prior
    self-distance (near 1 => compression, well above 1 => real separation)

plus the same CV / IQR-median for the within-prior self-distances as a
reference point, and a histogram grid comparing the two distributions in
each feature space.

Caches used (must exist — run the individual benchmark scripts first):
    cache/talent_feats_512.npz
    cache/real_feats_icl_25000_512_v2.npz
    cache/talent_col_hists.npz
    cache/icl_col_hists_<n>.npz
    cache/talent_corr_hists.npz
    cache/tabicl_corr_hists.npz
    cache/talent_tabicl_emb_colrow.npz
    cache/icl_tabicl_emb_25000_colrow.npz

Usage
-----
    python benchmark/embedding_distance_spread.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from tabdistances.analysis.distances import compute_nn_distances

_ROOT = Path(__file__).resolve().parent.parent
CACHE = _ROOT / "cache"
RESULTS = _ROOT / "results" / "embedding-distance-spread"

TALENT_FEATS_CACHE = CACHE / "talent_feats_512.npz"
ICL_FEATS_CACHE    = CACHE / "real_feats_icl_25000_512_v2.npz"
TALENT_COL_CACHE   = CACHE / "talent_col_hists.npz"
TALENT_CORR_CACHE  = CACHE / "talent_corr_hists.npz"
TABICL_CORR_CACHE  = CACHE / "tabicl_corr_hists.npz"
TALENT_EMB_CACHE   = CACHE / "talent_tabicl_emb_colrow.npz"
ICL_EMB_CACHE       = CACHE / "icl_tabicl_emb_25000_colrow.npz"

KNN_K        = 5
RANDOM_STATE = 42
N_ICL        = 25_000


def _self_nn_distances(feats: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Mean k-NN distance of each point in *feats* to its k nearest neighbours
    within the same set (self excluded), Min-Max scaled to itself."""
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = MinMaxScaler().fit_transform(feats)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(feats)
    dists, _ = nn.kneighbors(feats)
    return dists[:, 1:].mean(axis=1)


def _spread_stats(dists: np.ndarray) -> dict:
    mean = float(np.mean(dists))
    median = float(np.median(dists))
    q25, q75 = np.percentile(dists, [25, 75])
    return {
        "N": len(dists),
        "mean": mean,
        "median": median,
        "CV": float(np.std(dists) / mean) if mean else float("nan"),
        "IQR/median": float((q75 - q25) / median) if median else float("nan"),
    }


def _subsample_prior(feats: np.ndarray, rng: np.random.Generator, n: int = N_ICL) -> np.ndarray:
    n = min(n, len(feats))
    idx = rng.choice(len(feats), size=n, replace=False)
    return feats[idx]


def _load_table_feats(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    if not TALENT_FEATS_CACHE.exists() or not ICL_FEATS_CACHE.exists():
        print(f"  [skip Full] cache missing")
        return None
    talent = np.load(TALENT_FEATS_CACHE, allow_pickle=True)["feats"]
    icl_all = np.load(ICL_FEATS_CACHE)["feats"]
    return talent, _subsample_prior(icl_all, rng)


def _load_col_hists(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    if not TALENT_COL_CACHE.exists():
        print(f"  [skip Col. Hists] cache missing: {TALENT_COL_CACHE}")
        return None
    talent_hists = np.load(TALENT_COL_CACHE, allow_pickle=True)["hists"]
    icl_col_cache = CACHE / f"icl_col_hists_{len(talent_hists)}.npz"
    if not icl_col_cache.exists():
        print(f"  [skip Col. Hists] ICL cache missing: {icl_col_cache}")
        return None
    icl_all = np.load(icl_col_cache)["hists"]
    return talent_hists, _subsample_prior(icl_all, rng)


def _load_corr_hists(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    if not TALENT_CORR_CACHE.exists() or not TABICL_CORR_CACHE.exists():
        print(f"  [skip Corr. Hists] cache missing")
        return None
    talent_hists = np.load(TALENT_CORR_CACHE, allow_pickle=True)["hists"]
    icl_all = np.load(TABICL_CORR_CACHE)["hists"]
    return talent_hists, _subsample_prior(icl_all, rng)


def _load_tabicl_emb(rng: np.random.Generator, emb_type: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not TALENT_EMB_CACHE.exists() or not ICL_EMB_CACHE.exists():
        print(f"  [skip TabICL {emb_type}] cache missing")
        return None
    bd = np.load(TALENT_EMB_CACHE, allow_pickle=True)
    id_ = np.load(ICL_EMB_CACHE, allow_pickle=True)
    if "feats_col" not in bd or len(bd["feats_col"]) == 0:
        print(f"  [skip TabICL {emb_type}] TALENT cache empty")
        return None
    if "feats_col" not in id_ or len(id_["feats_col"]) == 0:
        print(f"  [skip TabICL {emb_type}] ICL cache empty")
        return None

    if emb_type == "row":
        talent_feats, icl_feats = bd["feats_row"], id_["feats_row"]
    elif emb_type == "col":
        talent_feats, icl_feats = bd["feats_col"], id_["feats_col"]
    else:  # combined
        talent_feats = np.concatenate([bd["feats_col"], bd["feats_row"]], axis=1)
        icl_feats = np.concatenate([id_["feats_col"], id_["feats_row"]], axis=1)

    return talent_feats, _subsample_prior(icl_feats, rng)


REPRESENTATIONS = [
    ("Full", _load_table_feats, {}),
    ("Col. Hists", _load_col_hists, {}),
    ("Corr. Hists", _load_corr_hists, {}),
    ("TabICL Rows", _load_tabicl_emb, {"emb_type": "row"}),
    ("TabICL Columns", _load_tabicl_emb, {"emb_type": "col"}),
    ("TabICL Concat", _load_tabicl_emb, {"emb_type": "combined"}),
]


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = []
    hist_data = {}

    for label, loader, kwargs in REPRESENTATIONS:
        print(f"--- {label} ---")
        loaded = loader(rng, **kwargs)
        if loaded is None:
            rows.append({"Representation": label, "N": 0})
            continue
        talent_feats, prior_feats = loaded

        between = compute_nn_distances(talent_feats, prior_feats, k=KNN_K)
        within = _self_nn_distances(prior_feats, k=KNN_K)

        between_stats = _spread_stats(between)
        within_stats = _spread_stats(within)

        rows.append({
            "Representation": label,
            "N": between_stats["N"],
            "CV": round(between_stats["CV"], 4),
            "IQR/median": round(between_stats["IQR/median"], 4),
            "ratio (between/within median)": round(
                between_stats["median"] / within_stats["median"], 4
            ) if within_stats["median"] else float("nan"),
            "CV (within-prior)": round(within_stats["CV"], 4),
            "IQR/median (within-prior)": round(within_stats["IQR/median"], 4),
        })
        hist_data[label] = (between, within)

    summary = pd.DataFrame(rows).set_index("Representation")
    out_csv = RESULTS / "summary.csv"
    summary.to_csv(out_csv)

    print()
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_columns", 20, "display.width", 200):
        print(summary.to_string())
    print(f"\n  -> saved to {out_csv}")

    if hist_data:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (label, _, _) in zip(axes.flat, REPRESENTATIONS):
            if label not in hist_data:
                ax.set_visible(False)
                continue
            between, within = hist_data[label]
            ax.hist(within, bins=30, density=True, alpha=0.5, label="within-prior (self)")
            ax.hist(between, bins=30, density=True, alpha=0.5, label="TALENT -> prior")
            ax.set_title(label)
            ax.set_xlabel("mean 5-NN distance")
            ax.legend(fontsize=8)
        fig.tight_layout()
        out_png = RESULTS / "histograms_grid.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  -> saved to {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
