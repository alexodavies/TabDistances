"""3-panel combined histogram + mean-rank figure.

One panel per distance representation:
  (1) Table feature vectors   (rank_vs_coverage approach)
  (2) Column histograms       (rank_vs_coverage_columns approach)
  (3) Correlation histograms  (rank_vs_coverage_correlations approach)

Each panel:
  - Left y-axis:  normalised NN-distance density (TALENT datasets + TabICL Prior self)
  - Right y-axis: mean TabICL rank ± 1 SD per distance bin (inverted, lower = better)

Ranks are drawn exclusively from tabicl-rankings-full-context.csv.
Sized for US letter paper (full text width, portrait).

Usage
-----
    python analysis/rank_vs_coverage_combined.py [--output-dir results/rank-coverage]
                                                 [--n-bins 30]
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
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from tabdistances.discriminator.features import df_to_vec, FEATURE_MODES
from tabdistances.columns.node_features import get_histogram

# ------------------------------------------------------------------
# CONFIG  (mirrors paths used by the individual rank_vs_coverage* scripts)
# ------------------------------------------------------------------
_ROOT  = Path(__file__).resolve().parent.parent
CACHE  = _ROOT / "cache"

RANKINGS_CSV = _ROOT / "results/tabicl-rankings-full-context.csv"
TALENT_DIR   = _ROOT / "data"
ICL_DIR      = _ROOT / "tables-tabicl/parquet"

TALENT_FEATS_CACHE  = CACHE / "talent_feats_512.npz"
ICL_FEATS_CACHE     = CACHE / "real_feats_icl_25000_512_v2.npz"
TALENT_COL_CACHE    = CACHE / "talent_col_hists.npz"
TALENT_CORR_CACHE   = CACHE / "talent_corr_hists.npz"
TABICL_CORR_CACHE   = CACHE / "tabicl_corr_hists.npz"

MAX_ROWS     = 512
KNN_K        = 5
N_ICL        = 25_000
RANDOM_STATE = 42

# Correlation histogram config (matches rank_vs_coverage_correlations.py)
N_CORR_BINS  = 50
CORR_EDGES   = np.linspace(-1.0, 1.0, N_CORR_BINS + 1)
MIN_NUM_COLS = 2

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
# Shared utilities
# ------------------------------------------------------------------

def _stairs(ax, density, bins, color, label, alpha_fill=0.15, lw=1.5):
    patch = ax.stairs(density, bins, fill=True, label=label)
    patch.set_facecolor((*mpl.colors.to_rgb(color), alpha_fill))
    patch.set_edgecolor(color)
    patch.set_linewidth(lw)


def _nn_distances(query: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    """Mean k-NN distance from each query point to the reference pool."""
    query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
    ref   = np.nan_to_num(ref,   nan=0.0, posinf=0.0, neginf=0.0)
    scaler = MinMaxScaler().fit(ref)
    q = scaler.transform(query)
    r = scaler.transform(ref)
    tree = BallTree(r, metric="euclidean")
    dists, _ = tree.query(q, k=min(k, len(r)))
    return dists.mean(axis=1)


def _bin_metric(nn_dists: np.ndarray, ranks: np.ndarray,
                n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Bin TALENT data, zero out singleton bins, then rebin over trimmed range."""
    # Pass 1: find the non-singleton range
    bins1 = np.linspace(nn_dists.min(), nn_dists.max(), n_bins + 1)
    idx1  = np.clip(np.digitize(nn_dists, bins1) - 1, 0, n_bins - 1)
    cts1  = np.array([(idx1 == i).sum() for i in range(n_bins)])
    nz    = np.where(cts1 > 1)[0]
    x_lo  = bins1[nz[0]]      if len(nz) else bins1[0]
    x_hi  = bins1[nz[-1] + 1] if len(nz) else bins1[-1]

    # Pass 2: rebin only the data within [x_lo, x_hi]
    inner = (nn_dists >= x_lo) & (nn_dists <= x_hi)
    nd, rk = nn_dists[inner], ranks[inner]
    bins = np.linspace(x_lo, x_hi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    idx  = np.clip(np.digitize(nd, bins) - 1, 0, n_bins - 1)
    cts  = np.array([(idx == i).sum() for i in range(n_bins)])
    density = cts.astype(float)
    density /= density.max() if density.max() > 0 else 1.0
    means = np.array([rk[idx == i].mean() if cts[i] > 0 else np.nan for i in range(n_bins)])
    stds  = np.array([rk[idx == i].std()  if cts[i] > 1 else 0.0   for i in range(n_bins)])
    return bins, bin_centers, density, means, stds, x_lo, x_hi


# ------------------------------------------------------------------
# Rankings
# ------------------------------------------------------------------

def _load_rankings() -> tuple[dict[str, float], list[str]]:
    """Returns (rank_map {dataset: rank}, datasets list) from full-context CSV."""
    df = pd.read_csv(RANKINGS_CSV)
    tc = df[df["model"] == "tabicl"][["dataset", "rank"]].copy()
    tc["rank"] = tc["rank"].astype(int)
    rank_map = dict(zip(tc["dataset"], tc["rank"]))
    print(f"Rankings: {len(rank_map)} datasets (full-context TabICL)")
    return rank_map, tc["dataset"].tolist()


# ------------------------------------------------------------------
# Panel 1 — table feature vectors
# ------------------------------------------------------------------

def _load_table_panel(rank_map: dict, datasets: list[str], rng: np.random.Generator):
    """Returns (nn_dists_talent, nn_dists_self, ranks)."""
    # TALENT table features — load from cache, compute any missing entries
    cached: dict[str, np.ndarray] = {}
    if TALENT_FEATS_CACHE.exists():
        data = np.load(TALENT_FEATS_CACHE, allow_pickle=True)
        cached = {n: data["feats"][i] for i, n in enumerate(data["names"].tolist())
                  if n in set(datasets)}
        print(f"  Table feats: cache hit {len(cached)}/{len(datasets)} datasets")

    missing = [ds for ds in datasets if ds not in cached]
    if missing:
        print(f"  Table feats: computing {len(missing)} missing datasets …")
        skipped = 0
        for ds in tqdm(missing, desc="Featurising TALENT datasets"):
            ds_path = TALENT_DIR / ds
            if not ds_path.exists():
                skipped += 1
                continue
            parts = []
            for split in ("train", "val", "test"):
                npy = ds_path / f"N_{split}.npy"
                if npy.exists():
                    arr = np.load(npy, allow_pickle=True).astype(float)
                    parts.append(pd.DataFrame(
                        arr.reshape(-1, 1) if arr.ndim == 1 else arr,
                        columns=[f"num_{i}" for i in range(arr.shape[-1])],
                    ))
            if not parts:
                skipped += 1
                continue
            df = pd.concat(parts, ignore_index=True)
            if df.shape[0] == 0 or df.shape[1] == 0:
                skipped += 1
                continue
            cached[ds] = df_to_vec(df.iloc[:MAX_ROWS])
        print(f"  Table feats: computed {len(missing) - skipped} ({skipped} skipped)")
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(TALENT_FEATS_CACHE,
                 names=np.array(list(cached.keys())),
                 feats=np.array(list(cached.values())))
        print(f"  Table feats: cache updated ({len(cached)} total) -> {TALENT_FEATS_CACHE}")

    names        = [ds for ds in datasets if ds in cached]
    talent_feats = np.stack([cached[ds] for ds in names])
    ranks        = np.array([rank_map[n] for n in names], dtype=float)
    print(f"  Table feats: {len(names)} TALENT datasets")

    # ICL table features
    icl_all = np.load(ICL_FEATS_CACHE)["feats"]
    idx = rng.choice(len(icl_all), size=min(N_ICL, len(icl_all)), replace=False)
    icl_feats = icl_all[idx]
    print(f"  Table feats: {len(icl_feats)} ICL vectors")

    nn_dists = _nn_distances(talent_feats, icl_feats, KNN_K)
    half = len(icl_feats) // 2
    nn_self  = _nn_distances(icl_feats[:half], icl_feats[half:], KNN_K)
    return nn_dists, nn_self, ranks


# ------------------------------------------------------------------
# Panel 2 — column histograms
# ------------------------------------------------------------------

def _load_column_panel(rank_map: dict, rng: np.random.Generator):
    """Returns (nn_dists_talent, nn_dists_self, ranks) at column level."""
    # TALENT column hists
    if not TALENT_COL_CACHE.exists():
        raise FileNotFoundError(f"Cache missing: {TALENT_COL_CACHE}  "
                                f"(run rank_vs_coverage_columns.py first)")
    cdata = np.load(TALENT_COL_CACHE, allow_pickle=True)
    talent_hists   = cdata["hists"]
    col_datasets   = cdata["col_datasets"]
    print(f"  Col hists: {len(talent_hists)} TALENT columns")

    # ICL column hists (cache named after n_talent to match rank_vs_coverage_columns.py)
    n_talent = len(talent_hists)
    icl_col_cache = CACHE / f"icl_col_hists_{n_talent}.npz"
    if not icl_col_cache.exists():
        raise FileNotFoundError(f"Cache missing: {icl_col_cache}  "
                                f"(run rank_vs_coverage_columns.py first)")
    icl_hists = np.load(icl_col_cache)["hists"]
    idx = rng.choice(len(icl_hists), size=min(n_talent, len(icl_hists)), replace=False)
    icl_hists = icl_hists[idx]
    print(f"  Col hists: {len(icl_hists)} ICL columns")

    # Filter talent cols to those with rank data
    mask = np.array([ds in rank_map for ds in col_datasets])
    talent_hists_m = talent_hists[mask]
    ranks = np.array([rank_map[ds] for ds in col_datasets[mask]], dtype=float)

    nn_dists = _nn_distances(talent_hists_m, icl_hists, KNN_K)
    half = len(icl_hists) // 2
    nn_self  = _nn_distances(icl_hists[:half], icl_hists[half:], KNN_K)
    print(f"  Col hists: {mask.sum()} columns matched to ranks")
    return nn_dists, nn_self, ranks


# ------------------------------------------------------------------
# Panel 3 — correlation histograms
# ------------------------------------------------------------------

def _corr_pmf(arr2d: np.ndarray) -> np.ndarray | None:
    """(N, D) numeric array → 50-bin correlation PMF, or None if too few columns."""
    df = pd.DataFrame(arr2d)
    df = df.loc[:, (df.nunique() / max(len(df), 1)) >= 0.2]
    if df.shape[1] < MIN_NUM_COLS:
        return None
    corr = df.corr(method="pearson").to_numpy()
    n = corr.shape[0]
    vals = corr[np.triu_indices(n, k=1)]
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None
    hist = np.histogram(vals, bins=CORR_EDGES)[0].astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else hist


def _load_talent_corr_hists(datasets: list[str]) -> tuple[np.ndarray, list[str]]:
    if TALENT_CORR_CACHE.exists():
        d = np.load(TALENT_CORR_CACHE, allow_pickle=True)
        if d["hists"].shape[1] == N_CORR_BINS:
            print(f"  Corr hists: {len(d['hists'])} TALENT tables from cache")
            return d["hists"], d["names"].tolist()
        print(f"  Corr hists: cache has {d['hists'].shape[1]} bins (expected {N_CORR_BINS}), recomputing …")

    hists, names = [], []
    for ds in tqdm(datasets, desc="TALENT corr hists"):
        ds_path = TALENT_DIR / ds
        parts = []
        for split in ("train", "val", "test"):
            npy = ds_path / f"N_{split}.npy"
            if npy.exists():
                arr = np.load(npy, allow_pickle=True).astype(float)
                parts.append(arr.reshape(-1, 1) if arr.ndim == 1 else arr)
        if not parts:
            continue
        data = np.concatenate(parts, axis=0)
        h = _corr_pmf(data)
        if h is not None:
            hists.append(h)
            names.append(ds)

    hists_arr = np.array(hists)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(TALENT_CORR_CACHE, hists=hists_arr, names=np.array(names))
    print(f"  Corr hists: {len(hists_arr)} TALENT tables computed + cached")
    return hists_arr, names


def _load_tabicl_corr_hists() -> np.ndarray:
    if TABICL_CORR_CACHE.exists():
        arr = np.load(TABICL_CORR_CACHE)["hists"]
        if arr.shape[1] == N_CORR_BINS:
            print(f"  Corr hists: {len(arr)} TabICL tables from cache")
            return arr
        print(f"  Corr hists: cache has {arr.shape[1]} bins (expected {N_CORR_BINS}), recomputing …")

    if not ICL_DIR.exists():
        raise FileNotFoundError(f"TabICL parquet dir not found: {ICL_DIR}")

    all_files = sorted(ICL_DIR.glob("*.parquet"))
    rng = np.random.default_rng(RANDOM_STATE)
    if len(all_files) > 10_000:
        all_files = rng.choice(all_files, size=10_000, replace=False).tolist()

    hists = []
    for p in tqdm(all_files, desc="TabICL corr hists"):
        try:
            df = pd.read_parquet(p)
            num = df.select_dtypes(include="number")
            num = num.loc[:, (num.nunique() / max(len(num), 1)) >= 0.2]
            if num.shape[1] < MIN_NUM_COLS:
                continue
            h = _corr_pmf(num.to_numpy())
            if h is not None:
                hists.append(h)
        except Exception:
            pass

    arr = np.array(hists) if hists else np.zeros((0, N_CORR_BINS), dtype=np.float32)
    TABICL_CORR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TABICL_CORR_CACHE, hists=arr)
    print(f"  Corr hists: {len(arr)} TabICL tables computed + cached")
    return arr


def _load_corr_panel(rank_map: dict, datasets: list[str], rng: np.random.Generator):
    """Returns (nn_dists_talent, nn_dists_self, ranks) at table level."""
    talent_hists, names = _load_talent_corr_hists(datasets)
    tabicl_hists = _load_tabicl_corr_hists()

    idx = rng.choice(len(tabicl_hists),
                     size=min(N_ICL, len(tabicl_hists)), replace=False)
    tabicl_hists = tabicl_hists[idx]
    print(f"  Corr hists: {len(tabicl_hists)} TabICL tables used as reference")

    mask  = np.array([n in rank_map for n in names])
    hists_m = talent_hists[mask]
    ranks   = np.array([rank_map[n] for n in np.array(names)[mask]], dtype=float)

    nn_dists = _nn_distances(hists_m, tabicl_hists, KNN_K)
    half = len(tabicl_hists) // 2
    nn_self  = _nn_distances(tabicl_hists[:half], tabicl_hists[half:], KNN_K)
    print(f"  Corr hists: {mask.sum()} tables matched to ranks")
    return nn_dists, nn_self, ranks


# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

def plot_combined(panels: list[dict], output_path: Path, n_bins: int = 30) -> None:
    """panels: list of dicts with keys nn_dists, nn_self, ranks, title."""
    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True,
                                 constrained_layout=False)
        fig.subplots_adjust(left=0.07, right=0.93, bottom=0.26, top=0.91,
                            wspace=0.08)

        right_axes = []
        for col, (ax1, p) in enumerate(zip(axes, panels)):
            bins, bin_centers, talent_density, means, stds, x_lo, x_hi = \
                _bin_metric(p["nn_dists"], p["ranks"], n_bins)

            # Self-distance density (clipped to TALENT bins)
            si_idx     = np.clip(np.digitize(p["nn_self"], bins) - 1, 0, n_bins - 1)
            si_cts     = np.array([(si_idx == i).sum() for i in range(n_bins)])
            si_density = si_cts / si_cts.max()

            # Left: density histograms
            _stairs(ax1, talent_density, bins, _BLUE,   "TALENT datasets")
            _stairs(ax1, si_density,     bins, _PURPLE, "TabICL Prior (self)")
            ax1.set_xlim(x_lo, x_hi)
            ax1.set_xlabel("Mean k-NN distance (normalised)")
            if col == 0:
                ax1.set_ylabel("Relative density")

            ax1.set_title(p["title"])

            # Right: mean rank ± 1 SD (inverted — lower = better)
            ax2 = ax1.twinx()
            right_axes.append(ax2)
            mask = ~np.isnan(means)
            lo = np.clip(means - stds, 1, None)
            hi = means + stds
            ax2.plot(bin_centers[mask], means[mask],
                     color=_RED, lw=1.5, marker="o", ms=3,
                     label="Mean rank (↓ better)")
            ax2.fill_between(bin_centers[mask], lo[mask], hi[mask],
                             color=_RED, alpha=0.2)
            ax2.invert_yaxis()

            # Right-axis label and ticks only on the last panel
            ax2.set_ylabel("TabICL mean rank", color=_RED,
                           visible=(col == len(panels) - 1))
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True, prune="upper"))
            ax2.tick_params(axis="y", labelcolor=_RED,
                            labelleft=False, labelright=False,
                            direction="in")

            if mask.sum() >= 2:
                r = np.corrcoef(bin_centers[mask], means[mask])[0, 1]
                right_axes[-1]._corr_r = r   # stash for post-draw annotation

        # Single 3-column legend below all axes
        h_left, l_left = axes[0].get_legend_handles_labels()   # TALENT + self
        h_right = [right_axes[0].get_legend_handles_labels()[0][0]]
        l_right = ["Mean rank (↓ better)"]

        fig.legend(
            h_left + h_right,
            l_left + l_right,
            loc="lower center",
            ncols=3,
            bbox_to_anchor=(0.5, 0.01),
            frameon=True,
            borderaxespad=0.5,
        )

        fig.canvas.draw()
        _bbox = dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=1.0)
        _trans_yd = mpl.transforms.blended_transform_factory
        for ax2 in right_axes:
            trans = _trans_yd(ax2.transAxes, ax2.transData)
            ylim  = sorted(ax2.get_ylim())   # sort: inverted axis has ylim[0] > ylim[1]
            for loc in ax2.yaxis.get_majorticklocs():
                if ylim[0] <= loc <= ylim[1]:
                    ax2.text(0.97, loc, str(int(round(loc))),
                             transform=trans, ha="right", va="center",
                             fontsize=7, color=_RED, zorder=10, bbox=_bbox)
            if hasattr(ax2, "_corr_r"):
                ax2.text(0.5, 0.03, f"$r = {ax2._corr_r:.2f}$",
                         transform=ax2.transAxes, va="bottom", ha="center",
                         fontsize=7, color="black", zorder=10, bbox=_bbox)

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

    rank_map, datasets = _load_rankings()

    print("\n=== Panel 1: table feature vectors ===")
    nn1, self1, rk1 = _load_table_panel(rank_map, datasets, rng)

    print("\n=== Panel 2: column histograms ===")
    nn2, self2, rk2 = _load_column_panel(rank_map, rng)

    print("\n=== Panel 3: correlation histograms ===")
    nn3, self3, rk3 = _load_corr_panel(rank_map, datasets, rng)

    panels = [
        dict(nn_dists=nn1, nn_self=self1, ranks=rk1, title="Table features"),
        dict(nn_dists=nn2, nn_self=self2, ranks=rk2, title="Column histograms"),
        dict(nn_dists=nn3, nn_self=self3, ranks=rk3, title="Correlation histograms"),
    ]

    print("\n=== Generating figure ===")
    plot_combined(panels, args.output_dir / "rank_vs_coverage_combined.png", args.n_bins)


if __name__ == "__main__":
    main()
