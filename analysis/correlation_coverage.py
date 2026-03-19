"""Correlation-coverage: distribution of within-table pairwise correlations.

For each table (TALENT, FM, T4, ICL) we compute the histogram of all
upper-triangle Pearson correlations between continuous columns.  The
resulting 30-dim vector (fixed bins on [-1, 1]) is used as a table-level
feature for coverage analysis and rank/AUC prediction.

Only float columns whose uniqueness ratio >= 0.2 are included, matching
the filter used in column_coverage.py.  Tables with fewer than 2 such
columns are skipped (no correlations to compute).

Outputs
-------
  correlation_bands.png       — mean ± SD of the correlation histogram per source
  correlation_pca.png         — 2-D PCA of all histograms, coloured by source
  correlation_nn_distances.png — overlaid NN-distance densities
  corr_rank_vs_nndist_histogram.png / corr_auc_vs_nndist_histogram.png
  corr_rank_vs_nndist_scatter.png  / corr_auc_vs_nndist_scatter.png
  corr_rank_prediction.png         / corr_auc_prediction.png
  corr_rank_vs_nndist.csv

Usage
-----
    python analysis/correlation_coverage.py [--talent-dir data/]
                                            [--fm-dir tables-tabfm/parquet]
                                            [--t4-dir tables-t4/parquet]
                                            [--icl-dir tables-tabicl/parquet]
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

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
_ROOT        = Path(__file__).resolve().parent.parent

RANKINGS_CSV = _ROOT / "results/tabicl-rankings.csv"
TALENT_DIR   = _ROOT / "data"
FM_DIR       = _ROOT / "tables-tabfm/parquet"
T4_DIR       = _ROOT / "tables-t4/parquet"
ICL_DIR      = _ROOT / "tables-tabicl/parquet"

CACHE_DIR    = _ROOT / "cache"
OUTPUT_DIR   = _ROOT / "results/correlation-coverage"

N_BINS       = 30
CORR_BINS    = np.linspace(-1, 1, N_BINS + 1)
MAX_ROWS     = 512
KNN_K        = 5
UNIQUENESS_THRESHOLD = 0.2
RANDOM_STATE = 42
WANDB_PROJECT = "tabicl-correlation-coverage"

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

_BLUE   = "#2980b9"
_ORANGE = "#e67e22"
_GREEN  = "#27ae60"
_PURPLE = "#8e44ad"
_RED    = "#ef0e0e"


def _stairs(ax, density, bins, color, label, alpha_fill=0.2, lw=1.5):
    """Step-filled histogram with opaque edge and transparent fill."""
    patch = ax.stairs(density, bins, fill=True, label=label)
    patch.set_facecolor((*mpl.colors.to_rgb(color), alpha_fill))
    patch.set_edgecolor(color)
    patch.set_linewidth(lw)
    return patch


# ------------------------------------------------------------------
# Core feature: correlation histogram for one table
# ------------------------------------------------------------------

def _select_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only float columns with uniqueness ratio >= UNIQUENESS_THRESHOLD."""
    num = df.select_dtypes(include=["float"])
    n = len(num)
    if n == 0:
        return num
    return num[[c for c in num.columns if num[c].nunique() / n >= UNIQUENESS_THRESHOLD]]


def table_corr_hist(df: pd.DataFrame) -> np.ndarray | None:
    """Return a 30-bin normalised histogram of upper-triangle Pearson correlations.

    Returns None if the table has fewer than 2 continuous columns or all
    correlation values are NaN.
    """
    cont = _select_continuous(df.iloc[:MAX_ROWS])
    if cont.shape[1] < 2:
        return None
    corr = cont.corr().values
    i, j = np.triu_indices(corr.shape[0], k=1)
    values = corr[i, j]
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None
    counts, _ = np.histogram(values, bins=CORR_BINS)
    total = counts.sum()
    if total == 0:
        return None
    return counts / total  # sums to 1


# ------------------------------------------------------------------
# TALENT loader
# ------------------------------------------------------------------

def load_talent_corr_hists(
    talent_dir: Path,
    datasets: list[str],
    cache_path: Path,
) -> tuple[np.ndarray, list[str]]:
    """Return (hists, names) where hists is (N, 30) and names are dataset names."""
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        hists = data["hists"]
        names = data["names"].tolist()
        print(f"  Loaded {len(hists)} TALENT correlation histograms from cache")
        return hists, names

    hists, names = [], []
    skipped = 0
    for ds in tqdm(datasets, desc="TALENT corr hists"):
        ds_path = talent_dir / ds
        if not ds_path.exists():
            skipped += 1
            continue
        parts = []
        for split in ("train", "val", "test"):
            npy = ds_path / f"N_{split}.npy"
            if npy.exists():
                arr = np.load(npy, allow_pickle=True).astype(float)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                parts.append(pd.DataFrame(arr, columns=[f"num_{i}" for i in range(arr.shape[1])]))
        if not parts:
            skipped += 1
            continue
        df = pd.concat(parts, ignore_index=True)
        h = table_corr_hist(df)
        if h is None:
            skipped += 1
            continue
        hists.append(h)
        names.append(ds)

    print(f"  Computed {len(hists)} TALENT corr hists ({skipped} skipped)")
    hists_arr = np.array(hists)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, hists=hists_arr, names=np.array(names))
    print(f"  Cached -> {cache_path}")
    return hists_arr, names


# ------------------------------------------------------------------
# Parquet loader
# ------------------------------------------------------------------

def load_parquet_corr_hists(
    parquet_dir: Path,
    n: int,
    rng: np.random.Generator,
    cache_path: Path,
) -> np.ndarray:
    """Return up to n correlation histograms randomly sampled from parquet_dir."""
    if cache_path.exists():
        arr = np.load(cache_path)["hists"]
        if len(arr) >= n:
            idx = rng.choice(len(arr), size=n, replace=False)
            print(f"  Loaded {n} corr hists from cache ({cache_path.name})")
            return arr[idx]
        print(f"  Cache has {len(arr)} hists (< {n} requested), using all")
        return arr

    paths = np.array(sorted(parquet_dir.glob("*.parquet")), dtype=object)
    paths = rng.permutation(paths)

    hists, errors = [], 0
    for p in tqdm(paths, desc=f"Corr hists {parquet_dir.name}"):
        try:
            df = pd.read_parquet(p)
            h = table_corr_hist(df)
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
# NN distances
# ------------------------------------------------------------------

def compute_nn_distances(query: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    """Mean k-NN distance from each query to the reference pool."""
    query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
    ref   = np.nan_to_num(ref,   nan=0.0, posinf=0.0, neginf=0.0)
    scaler = MinMaxScaler().fit(ref)
    q_sc = scaler.transform(query)
    r_sc = scaler.transform(ref)
    tree = BallTree(r_sc, metric="euclidean")
    dists, _ = tree.query(q_sc, k=min(k, len(r_sc)))
    return dists.mean(axis=1)


# ------------------------------------------------------------------
# Plot 1 — mean ± SD band of the correlation histogram
# ------------------------------------------------------------------

def plot_histogram_bands(
    hists: dict[str, np.ndarray],
    output_dir: Path,
    run=None,
) -> None:
    """Mean ± 1 SD band of the correlation histogram per source."""
    bin_centers = (CORR_BINS[:-1] + CORR_BINS[1:]) / 2
    with mpl.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for key, arr in hists.items():
            if len(arr) == 0:
                continue
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0)
            color = _COLORS[key]
            ax.plot(bin_centers, mu, color=color, label=_LABELS[key], lw=1.2)
            ax.fill_between(bin_centers, mu - sd, mu + sd, color=color, alpha=0.2)
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.set_xlabel("Pearson correlation")
        ax.set_ylabel("Fraction of pairs")
        ax.set_title("Within-table correlation distributions")
        ax.legend()
        plt.tight_layout()
        out = output_dir / "correlation_bands.png"
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({"correlation_bands": wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------------
# Plot 2 — PCA
# ------------------------------------------------------------------

def plot_pca(
    hists: dict[str, np.ndarray],
    output_dir: Path,
    run=None,
) -> None:
    """2-D PCA of all correlation histograms coloured by source."""
    arrays, labels = [], []
    for key, arr in hists.items():
        if len(arr) == 0:
            continue
        arrays.append(arr)
        labels.extend([key] * len(arr))

    X = np.nan_to_num(np.concatenate(arrays), nan=0.0)
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
        ax.set_title("PCA of correlation histograms")
        ax.legend()
        plt.tight_layout()
        out = output_dir / "correlation_pca.png"
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({"correlation_pca": wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------------
# Plot 3 — NN-distance density histograms
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
        ax.set_title("Correlation-histogram NN distances")
        ax.legend()
        plt.tight_layout()
        out = output_dir / "correlation_nn_distances.png"
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({"correlation_nn_distances": wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------------
# Rank / AUC plots
# ------------------------------------------------------------------

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
    """NN-distance density histogram (left y) + mean metric ± 1 SD per bin (right y)."""
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
    bin_means = np.array([y_vals[idx == i].mean() if (idx == i).sum() > 0 else np.nan
                          for i in range(n_bins)])
    bin_stds  = np.array([y_vals[idx == i].std()  if (idx == i).sum() > 1 else 0.0
                          for i in range(n_bins)])

    with mpl.rc_context(_PAPER_RC):
        fig, ax1 = plt.subplots(figsize=(7.5, 3.5))
        _stairs(ax1, density, bins, _BLUE, "TALENT tables", alpha_fill=0.15)
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
        ax2.plot(bin_centers[mask], bin_means[mask], color=_RED, lw=1.5,
                 marker="o", ms=4, label=f"Mean {y_label}")
        ax2.fill_between(bin_centers[mask], lo[mask], hi[mask], color=_RED, alpha=0.25)
        if invert_y:
            ax2.invert_yaxis()
        ax2.set_ylabel(f"TabICL mean {y_label}", color=_RED)
        ax2.tick_params(axis="y", labelcolor=_RED)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", bbox_to_anchor=(1.12, 1), borderaxespad=0)
        plt.title(f"Correlation NN Distance vs {y_label}")
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
    r_p, p_p = stats.pearsonr(nn_dists, y_vals)
    r_s, p_s = stats.spearmanr(nn_dists, y_vals)
    m, b = np.polyfit(nn_dists, y_vals.astype(float), 1)
    xs = np.linspace(nn_dists.min(), nn_dists.max(), 200)
    with mpl.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.scatter(nn_dists, y_vals, alpha=0.5, s=20, color=_BLUE, linewidths=0, zorder=3)
        ax.plot(xs, m * xs + b, color=_ORANGE, lw=1.5, label="Linear fit", zorder=4)
        ax.set_xlabel("Mean k-NN distance (normalised)")
        ax.set_ylabel(f"TabICL {y_label}")
        ax.set_title(
            f"{y_label} vs correlation NN distance\n"
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
    X = nn_dists.reshape(-1, 1)
    y = y_vals.astype(float)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    residuals = y - y_pred
    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
        lo, hi = y.min(), y.max()
        axes[0].scatter(y, y_pred, alpha=0.5, s=20, color=_GREEN, linewidths=0)
        axes[0].plot([lo, hi], [lo, hi], "k--", lw=0.8, label="Perfect fit")
        axes[0].set_xlabel(f"Actual {y_label}")
        axes[0].set_ylabel(f"Predicted {y_label}")
        axes[0].set_title(f"Linear Regression  R² = {r2:.3f}")
        axes[0].legend()
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20, color=_ORANGE, linewidths=0)
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
# Main
# ------------------------------------------------------------------

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
    tabicl_ranks = (
        rankings[rankings["model"] == "tabicl"][["dataset", "rank", "AUC"]].copy()
    )
    tabicl_ranks["rank"] = tabicl_ranks["rank"].astype(int)
    datasets = tabicl_ranks["dataset"].tolist()
    rank_map = tabicl_ranks.set_index("dataset")[["rank", "AUC"]].to_dict("index")
    print(f"\n=== TALENT: {len(datasets)} datasets ===")

    # ── TALENT ─────────────────────────────────────────────────────
    talent_cache = CACHE_DIR / "talent_corr_hists.npz"
    talent_hists, talent_names = load_talent_corr_hists(talent_dir, datasets, talent_cache)
    n_talent = len(talent_hists)
    print(f"  {n_talent} TALENT tables with correlation histograms")

    # ── FM ─────────────────────────────────────────────────────────
    print(f"\n=== FM (sampling {n_talent} tables) ===")
    fm_cache = CACHE_DIR / f"fm_corr_hists_{n_talent}.npz"
    fm_hists = load_parquet_corr_hists(fm_dir, n_talent, rng, fm_cache) if fm_dir.exists() else np.zeros((0, N_BINS))

    # ── T4 ─────────────────────────────────────────────────────────
    print(f"\n=== T4 (sampling {n_talent} tables) ===")
    t4_cache = CACHE_DIR / f"t4_corr_hists_{n_talent}.npz"
    t4_hists = load_parquet_corr_hists(t4_dir, n_talent, rng, t4_cache) if t4_dir.exists() else np.zeros((0, N_BINS))

    # ── ICL ────────────────────────────────────────────────────────
    print(f"\n=== ICL (sampling {n_talent} tables) ===")
    icl_cache = CACHE_DIR / f"icl_corr_hists_{n_talent}.npz"
    icl_hists = load_parquet_corr_hists(icl_dir, n_talent, rng, icl_cache) if icl_dir.exists() else np.zeros((0, N_BINS))

    hists = {"talent": talent_hists, "fm": fm_hists, "t4": t4_hists, "icl": icl_hists}
    for key, arr in hists.items():
        print(f"  {_LABELS[key]}: {len(arr)} tables")

    # ── wandb ──────────────────────────────────────────────────────
    run = None
    if _WANDB_AVAILABLE and WANDB_PROJECT:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"corr_coverage_k{k}",
            config={"k": k, "n_talent": n_talent,
                    "n_fm": len(fm_hists), "n_t4": len(t4_hists), "n_icl": len(icl_hists)},
            reinit=True,
        )

    # ── Distribution plots ─────────────────────────────────────────
    print("\n=== Generating figures ===")
    plot_histogram_bands(hists, output_dir, run)
    plot_pca(hists, output_dir, run)

    # ── NN distances ───────────────────────────────────────────────
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

    # ── Rank / AUC analysis ────────────────────────────────────────
    _icl = nn_pairs.get("talent_to_icl")
    _fm  = nn_pairs.get("talent_to_fm")
    _t4  = nn_pairs.get("talent_to_t4")
    col_nn = _icl if _icl is not None else (_fm if _fm is not None else _t4)
    ref_label = "ICL" if _icl is not None else ("FM" if _fm is not None else "T4")

    if col_nn is not None:
        print("\n=== Rank / AUC analysis ===")

        # Align nn_dists with rankings
        nn_dict = dict(zip(talent_names, col_nn))
        fm_dict = dict(zip(talent_names, _fm)) if _fm is not None else {}

        matched = [(ds, nn_dict[ds], rank_map[ds]["rank"], rank_map[ds]["AUC"])
                   for ds in talent_names if ds in rank_map and ds in nn_dict]
        if not matched:
            print("  No datasets matched to rankings — skipping rank/AUC plots")
        else:
            ds_names, nn_vals, rk_vals, auc_vals = zip(*matched)
            nn_arr  = np.array(nn_vals)
            rk_arr  = np.array(rk_vals, dtype=float)
            auc_arr = np.array(auc_vals, dtype=float)
            fm_arr  = np.array([fm_dict[ds] for ds in ds_names]) if fm_dict else None

            # ICL self-distances
            icl_self_dists = None
            if len(icl_hists) >= 2:
                half = len(icl_hists) // 2
                icl_self_dists = compute_nn_distances(icl_hists[:half], icl_hists[half:], k)

            # Save CSV
            csv_path = output_dir / "corr_rank_vs_nndist.csv"
            pd.DataFrame({"dataset": ds_names, "nn_dist": nn_arr,
                          "rank": rk_arr, "AUC": auc_arr}).to_csv(csv_path, index=False)
            print(f"  Saved {csv_path}")

            _plot_metric_histogram(nn_arr, rk_arr, "rank (lower = better)",
                                   "corr_rank_vs_nndist_histogram.png", output_dir, run,
                                   invert_y=True, nn_dists_fm=fm_arr,
                                   nn_dists_icl_self=icl_self_dists)
            _plot_metric_histogram(nn_arr, auc_arr, "AUC",
                                   "corr_auc_vs_nndist_histogram.png", output_dir, run,
                                   nn_dists_fm=fm_arr, nn_dists_icl_self=icl_self_dists)

            corr = _plot_metric_scatter(nn_arr, rk_arr, f"rank (TALENT $\\in$ {ref_label})",
                                        "corr_rank_vs_nndist_scatter.png", output_dir, run)
            _plot_metric_scatter(nn_arr, auc_arr, f"AUC (TALENT $\\in$ {ref_label})",
                                 "corr_auc_vs_nndist_scatter.png", output_dir, run)
            pred = _plot_metric_prediction(nn_arr, rk_arr, "rank",
                                           "corr_rank_prediction.png", output_dir, run)
            _plot_metric_prediction(nn_arr, auc_arr, "AUC",
                                    "corr_auc_prediction.png", output_dir, run)

            print(f"\n=== Results (n={len(nn_arr)} datasets) ===")
            print(f"  Pearson  r : {corr['pearson_r']:.3f}  (p = {corr['pearson_p']:.4f})")
            print(f"  Spearman ρ : {corr['spearman_r']:.3f}  (p = {corr['spearman_p']:.4f})")
            print(f"  Linear R²  : {pred['r2']:.3f}")

            if run is not None:
                run.log({"corr_pearson_r": corr["pearson_r"],
                         "corr_spearman_r": corr["spearman_r"],
                         "corr_linear_r2": pred["r2"]})

    if run is not None:
        run.finish()

    print(f"\nOutputs in: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correlation-coverage analysis: TALENT vs FM vs T4 vs ICL."
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
