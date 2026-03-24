"""3-panel combined histogram + metric figure for rank_vs_coverage.

Produces a single figure (rank_vs_coverage_combined.png) with three side-by-side
panels, each showing:
  - Left y-axis: normalised NN-distance density (TALENT datasets + TabICL Prior self)
  - Right y-axis: mean metric ± 1 SD per bin (rank, AUC, relative AUC)

Sized for US letter paper (full-width, portrait).

Usage
-----
    python analysis/rank_vs_coverage_combined.py [--output-dir results/rank-coverage]
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
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from tabdistances.discriminator.features import df_to_vec, select_features, FEATURE_MODES

# ------------------------------------------------------------------
# CONFIG  (shared with rank_vs_coverage.py)
# ------------------------------------------------------------------
_ROOT                 = Path(__file__).resolve().parent.parent
ICL_CACHE             = _ROOT / "cache/real_feats_icl_25000_512_v2.npz"
RANKINGS_CSV          = _ROOT / "results/tabicl-rankings-full-context.csv"
AUTOGLUON_RESULTS_CSV = _ROOT / "results/results-autogluon-512-context.csv"
TALENT_DIR            = _ROOT / "data"

MAX_ROWS     = 512
N_ICL        = 25_000
FEATURE_MODE = "full"
KNN_K        = 5
RANDOM_STATE = 42

_PAPER_RC = {
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.5,
    "figure.dpi":      300,
}

_BLUE   = "#2980b9"
_PURPLE = "#8e44ad"
_RED    = "#ef0e0e"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _stairs(ax, density, bins, color, label, alpha_fill=0.2, lw=1.5):
    patch = ax.stairs(density, bins, fill=True, label=label)
    patch.set_facecolor((*mpl.colors.to_rgb(color), alpha_fill))
    patch.set_edgecolor(color)
    patch.set_linewidth(lw)


def _load_talent_feats(talent_dir: Path, datasets: list[str], cache: Path) -> list[dict]:
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        feats_arr = data["feats"]
        names_arr = data["names"]
        by_name = {n: feats_arr[i] for i, n in enumerate(names_arr)}
        records = [{"dataset": d, "feats": by_name[d]} for d in datasets if d in by_name]
        print(f"  Loaded {len(records)} TALENT feature vectors from {cache}")
        return records

    print(f"  Cache not found at {cache}, computing features …")
    import glob as _glob, os
    records = []
    dataset_dirs = {p.name: p for p in talent_dir.iterdir() if p.is_dir()}
    for ds in tqdm(datasets, desc="TALENT feats"):
        d = dataset_dirs.get(ds)
        if d is None:
            continue
        csvs = list(d.glob("*.csv"))
        if not csvs:
            continue
        dfs = []
        for csv in csvs:
            try:
                dfs.append(pd.read_csv(csv, nrows=MAX_ROWS))
            except Exception:
                pass
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True).head(MAX_ROWS)
        try:
            vec = df_to_vec(df, mode=FEATURE_MODE)
        except Exception:
            continue
        records.append({"dataset": ds, "feats": vec})

    if records:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache,
                 feats=np.stack([r["feats"] for r in records]),
                 names=np.array([r["dataset"] for r in records]))
        print(f"  Cached {len(records)} vectors to {cache}")
    return records


def _load_icl_feats(cache: Path, rng: np.random.Generator) -> np.ndarray:
    data = np.load(cache)
    arr  = data["feats"]
    idx  = rng.choice(len(arr), size=min(N_ICL, len(arr)), replace=False)
    return arr[idx]


def _compute_nn_distances(query: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    scaler = MinMaxScaler().fit(ref)
    q = scaler.transform(query)
    r = scaler.transform(ref)
    tree = BallTree(r, metric="euclidean")
    dists, _ = tree.query(q, k=k)
    return dists.mean(axis=1)


def _bin_metric(nn_dists: np.ndarray, y_vals: np.ndarray, bins: np.ndarray):
    n_bins = len(bins) - 1
    idx = np.clip(np.digitize(nn_dists, bins) - 1, 0, n_bins - 1)
    counts = np.array([(idx == i).sum() for i in range(n_bins)])
    density = counts / counts.max()
    means = np.array([y_vals[idx == i].mean() if (idx == i).sum() > 0 else np.nan
                      for i in range(n_bins)])
    stds  = np.array([y_vals[idx == i].std()  if (idx == i).sum() > 1 else 0.0
                      for i in range(n_bins)])
    return density, means, stds


# ------------------------------------------------------------------
# Main figure
# ------------------------------------------------------------------
def plot_combined(
    nn_dists_talent: np.ndarray,
    nn_dists_self:   np.ndarray,
    rk:    np.ndarray,
    auc:   np.ndarray,
    rel_auc: np.ndarray,
    output_path: Path,
    n_bins: int = 30,
) -> None:
    """3-panel histogram + metric figure sized for US letter paper."""

    # Shared distance bins (cover both talent and self)
    all_dists = np.concatenate([nn_dists_talent, nn_dists_self])
    bins = np.linspace(all_dists.min(), all_dists.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Self-distance density (same in all panels)
    si_idx    = np.clip(np.digitize(nn_dists_self, bins) - 1, 0, n_bins - 1)
    si_counts = np.array([(si_idx == i).sum() for i in range(n_bins)])
    si_density = si_counts / si_counts.max()

    panels = [
        dict(y=rk,      label="rank",         ylabel="Mean rank (↓ better)", invert=True),
        dict(y=auc,     label="AUC",           ylabel="Mean AUC",             invert=False),
        dict(y=rel_auc, label="relative AUC",  ylabel="Mean relative AUC",    invert=False),
    ]

    # US letter full-width: 7.5" wide, height chosen to leave room for the legend
    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True,
                                 constrained_layout=False)
        fig.subplots_adjust(left=0.07, right=0.93, bottom=0.22, top=0.93,
                            wspace=0.08)

        right_axes = []
        for col, (ax1, p) in enumerate(zip(axes, panels)):
            talent_density, means, stds = _bin_metric(nn_dists_talent, p["y"], bins)

            # Left: density histograms
            _stairs(ax1, talent_density, bins, _BLUE,   "TALENT datasets",      alpha_fill=0.15)
            _stairs(ax1, si_density,     bins, _PURPLE, "TabICL Prior (self)",  alpha_fill=0.15)
            ax1.set_xlabel("Mean k-NN distance (normalised)")
            if col == 0:
                ax1.set_ylabel("Relative density")

            # Right: mean metric ± 1 SD
            ax2 = ax1.twinx()
            right_axes.append(ax2)
            mask = ~np.isnan(means)
            lo = np.clip(means - stds, 1, None) if p["invert"] else means - stds
            hi = means + stds
            ax2.plot(bin_centers[mask], means[mask],
                     color=_RED, lw=1.5, marker="o", ms=3,
                     label=p["ylabel"])
            ax2.fill_between(bin_centers[mask], lo[mask], hi[mask],
                             color=_RED, alpha=0.2)
            if p["invert"]:
                ax2.invert_yaxis()
            ax2.set_ylabel(p["ylabel"], color=_RED,
                           visible=(col == len(panels) - 1))
            ax2.tick_params(axis="y", labelcolor=_RED,
                            labelleft=False,
                            labelright=(col == len(panels) - 1))
            ax1.set_title(p["label"])

        # Shared legend below all axes (3 columns)
        # Collect: density handles from axes[0], metric handles from each right axis
        h_left,  l_left  = axes[0].get_legend_handles_labels()
        h_right = [ax.get_legend_handles_labels()[0][0] for ax in right_axes]
        l_right = [ax.get_legend_handles_labels()[1][0] for ax in right_axes]

        fig.legend(
            h_left + h_right,
            l_left + l_right,
            loc="lower center",
            ncols=3,
            bbox_to_anchor=(0.5, 0.01),
            frameon=True,
            borderaxespad=0.5,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {output_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=_ROOT / "results/rank-coverage")
    parser.add_argument("--n-bins", type=int, default=30)
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_STATE)

    # Rankings
    rankings  = pd.read_csv(RANKINGS_CSV)
    mean_aucs = rankings.groupby("dataset")["AUC"].mean().rename("mean_AUC")
    tabicl_ranks = (
        rankings[rankings["model"] == "tabicl"][["dataset", "rank", "AUC"]].copy()
    )
    tabicl_ranks["rank"] = tabicl_ranks["rank"].astype(int)
    tabicl_ranks = tabicl_ranks.merge(mean_aucs, on="dataset")
    tabicl_ranks["relative_auc"] = tabicl_ranks["AUC"] / tabicl_ranks["mean_AUC"]
    datasets = tabicl_ranks["dataset"].tolist()
    print(f"Rankings: {len(datasets)} datasets with tabicl entries")

    # TALENT features
    talent_cache = _ROOT / f"cache/talent_feats_{MAX_ROWS}.npz"
    records = _load_talent_feats(TALENT_DIR, datasets, talent_cache)
    if not records:
        print("ERROR: no TALENT features found — check --output-dir / data paths")
        sys.exit(1)

    talent_feats = np.stack([r["feats"] for r in records])
    talent_names = [r["dataset"] for r in records]
    print(f"Using {len(talent_names)} TALENT datasets, shape={talent_feats.shape}")

    # TabICL Prior features
    icl_feats = _load_icl_feats(ICL_CACHE, rng)
    print(f"Using {len(icl_feats)} TabICL feature vectors, shape={icl_feats.shape}")

    # NN distances
    print(f"Computing {KNN_K}-NN distances …")
    nn_dists = _compute_nn_distances(talent_feats, icl_feats, KNN_K)

    half = len(icl_feats) // 2
    nn_dists_self = _compute_nn_distances(icl_feats[:half], icl_feats[half:], KNN_K)

    # Merge with rankings
    df = pd.DataFrame({"dataset": talent_names, "nn_dist": nn_dists})
    df = df.merge(tabicl_ranks, on="dataset", how="inner")
    print(f"Matched {len(df)} datasets to rankings")

    plot_combined(
        nn_dists_talent = df["nn_dist"].values,
        nn_dists_self   = nn_dists_self,
        rk              = df["rank"].values.astype(float),
        auc             = df["AUC"].values.astype(float),
        rel_auc         = df["relative_auc"].values.astype(float),
        output_path     = args.output_dir / "rank_vs_coverage_combined.png",
        n_bins          = args.n_bins,
    )


if __name__ == "__main__":
    main()
