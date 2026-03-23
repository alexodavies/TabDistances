"""Rank-vs-coverage: does NN distance from a TALENT dataset to the TabICL
prior predict TabICL's ranking on that dataset?

For every TALENT benchmark dataset that appears in results/tabicl-rankings.csv
we compute a single 70-dim feature vector (via df_to_vec on the numeric columns
of the combined train/val/test splits), then measure the mean k-NN distance to
the TabICL prior's feature distribution (loaded from tables-tabicl/parquet/).

Primary outputs
---------------
1. rank_vs_nndist_histogram.png
       NN-distance histogram (count, left y-axis) with a dual right y-axis
       showing TabICL's mean rank ± 1 SD within each distance bin.

2. rank_vs_nndist_scatter.png
       Scatter plot of rank vs NN distance with Pearson r, Spearman ρ, and
       a linear regression line.

3. rank_prediction.png
       Linear regression diagnostics: actual-vs-predicted and residual plot.

4. rank_vs_nndist.csv
       Raw per-dataset table (dataset, nn_dist, rank, AUC).

Usage
-----
    python analysis/rank_vs_coverage.py [--talent-dir data/]
                                        [--tabicl-dir tables-tabicl/parquet]
                                        [--mode full]
                                        [--k 5]
                                        [--n-bins 15]
                                        [--output-dir results/rank-coverage]
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
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tabdistances.discriminator.features import (
    df_to_vec,
    select_features,
    FEATURE_MODES,
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
_ROOT        = Path(__file__).resolve().parent.parent   # project root

TABICL_DIR          = _ROOT / "tables-tabicl/parquet"
ICL_CACHE           = _ROOT / "cache/real_feats_icl_25000_512_v2.npz"
FM_CACHE            = _ROOT / "cache/real_feats_tabfm_5000_512.npz"
T4_CACHE            = _ROOT / "cache/real_feats_t4_5000_512_v2.npz"
RANKINGS_CSV        = _ROOT / "results/tabicl-rankings-full-context.csv"
AUTOGLUON_RESULTS_CSV = _ROOT / "results/results-autogluon-512-context.csv"
TALENT_DIR          = _ROOT / "data"

MAX_ROWS     = 512
N_ICL        = 25000
FEATURE_MODE = "full"
KNN_K        = 5
OUTPUT_DIR   = _ROOT / "results/rank-coverage"
RANDOM_STATE  = 42
WANDB_PROJECT = "tabicl-rank-coverage"

_PAPER_RC = {
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 4,
    "figure.dpi":      300,
}

_BLUE   = "#2980b9"
_ORANGE = "#e67e22"
_GREEN  = "#27ae60"
_PURPLE = "#8e44ad"
_RED    = "#ef0e0e"


def _stairs(ax, density, bins, color, label, alpha_fill=0.2, lw=1.5):
    """stairs() with opaque edges and transparent fill."""
    patch = ax.stairs(density, bins, fill=True, label=label)
    patch.set_facecolor((*mpl.colors.to_rgb(color), alpha_fill))
    patch.set_edgecolor(color)
    patch.set_linewidth(lw)
    return patch


# ------------------------------------------------------------------
# TALENT dataset loader
# ------------------------------------------------------------------

def _load_talent_dataset(ds_path: Path) -> pd.DataFrame | None:
    """Load numeric features from all splits of one TALENT dataset."""
    parts = []
    for split in ("train", "val", "test"):
        npy = ds_path / f"N_{split}.npy"
        if npy.exists():
            arr = np.load(npy, allow_pickle=True).astype(float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            df_split = pd.DataFrame(arr, columns=[f"num_{i}" for i in range(arr.shape[1])])
            parts.append(df_split)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def load_talent_feats(
    talent_dir: Path,
    datasets: list[str],
    cache_path: Path,
) -> list[dict]:
    """Return list of {dataset, feats} dicts, using a cache when available."""
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        names = data["names"].tolist()
        feats = data["feats"]
        print(f"  Loaded TALENT features from cache: {cache_path} ({len(names)} datasets)")
        return [{"dataset": n, "feats": f} for n, f in zip(names, feats)]

    records = []
    skipped = 0
    for ds in tqdm(datasets, desc="Featurising TALENT datasets"):
        ds_path = talent_dir / ds
        if not ds_path.exists():
            skipped += 1
            continue
        df = _load_talent_dataset(ds_path)
        if df is None or df.shape[0] == 0 or df.shape[1] == 0:
            skipped += 1
            continue
        vec = df_to_vec(df.iloc[:MAX_ROWS])
        records.append({"dataset": ds, "feats": vec})

    print(f"  Featurised {len(records)} TALENT datasets ({skipped} skipped / not found)")

    if records:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            names=np.array([r["dataset"] for r in records]),
            feats=np.array([r["feats"] for r in records]),
        )
        print(f"  Cached TALENT features -> {cache_path}")

    return records


# ------------------------------------------------------------------
# Autogluon rank loader
# ------------------------------------------------------------------

def load_autogluon_ranks(results_csv: Path) -> pd.DataFrame:
    """Load autogluon (and tabicl-with-autogluon) ranks from results CSV.

    Returns a DataFrame with columns [dataset, ag_rank, tabicl_rank_with_ag]
    where ranks are among all 8 models including autogluon.
    """
    df = pd.read_csv(results_csv)
    ranking_rows = df[df["metric"] == "ranking"][["dataset", "model", "value"]].copy()
    ranking_rows["value"] = ranking_rows["value"].astype(int)

    ag = (
        ranking_rows[ranking_rows["model"] == "autogluon"]
        .rename(columns={"value": "ag_rank"})
        [["dataset", "ag_rank"]]
    )
    tc = (
        ranking_rows[ranking_rows["model"] == "tabicl"]
        .rename(columns={"value": "tabicl_rank_with_ag"})
        [["dataset", "tabicl_rank_with_ag"]]
    )
    merged = ag.merge(tc, on="dataset", how="outer")
    print(f"  Autogluon ranks: {ag['dataset'].nunique()} datasets, "
          f"tabicl-with-ag ranks: {tc['dataset'].nunique()} datasets")
    return merged


# ------------------------------------------------------------------
# TabICL feature loader
# ------------------------------------------------------------------

def load_icl_feats(
    icl_cache: Path,
    tabicl_dir: Path,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Load (or compute and cache) tabICL feature vectors."""
    if icl_cache.exists():
        arr = np.load(icl_cache)["feats"]
        idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
        print(f"  Loaded {len(idx)} tabICL features from cache: {icl_cache}")
        return arr[idx]

    paths = np.array(sorted(tabicl_dir.glob("*.parquet")))
    if not len(paths):
        print(f"ERROR: no .parquet files in {tabicl_dir}")
        sys.exit(1)
    rng.shuffle(paths)

    feats, errors = [], 0
    for p in tqdm(paths, desc="Featurising tabICL tables"):
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

    print(f"  Featurised {len(feats)} tabICL tables ({errors} skipped)")
    arr = np.array(feats)
    icl_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(icl_cache, feats=arr)
    print(f"  Cached tabICL features -> {icl_cache}")
    return arr


# ------------------------------------------------------------------
# NN distance computation
# ------------------------------------------------------------------

def compute_nn_distances(
    query_feats: np.ndarray,
    ref_feats: np.ndarray,
    mode: str,
    k: int,
) -> np.ndarray:
    """Mean k-NN distance from each query point to the reference distribution.

    Features are selected by mode and scaled to [0, 1] using the reference
    distribution's range (consistent with pareto-coverage.py).
    """
    q_m, _ = select_features(query_feats, mode)
    r_m, _ = select_features(ref_feats, mode)

    q_m = np.nan_to_num(q_m, nan=0.0, posinf=0.0, neginf=0.0)
    r_m = np.nan_to_num(r_m, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = MinMaxScaler().fit(r_m)
    q_sc = scaler.transform(q_m)
    r_sc = scaler.transform(r_m)

    tree = BallTree(r_sc, metric="euclidean")
    dists, _ = tree.query(q_sc, k=min(k, len(r_sc)))
    return dists.mean(axis=1)   # (n_query,)




# ------------------------------------------------------------------
# Plot 1 — histogram + dual-axis rank
# ------------------------------------------------------------------

def plot_histogram_with_metric(
    nn_dists_icl: np.ndarray,
    y_vals: np.ndarray,
    y_label: str,
    filename: str,
    output_dir: Path,
    run=None,
    n_bins: int = 30,
    nn_dists_fm: np.ndarray | None = None,
    nn_dists_icl_self: np.ndarray | None = None,
    invert_y: bool = False,
) -> None:
    """NN-distance histogram (left y) with mean metric ± 1 SD per bin (right y).

    Optionally overlays FM and ICL self-distances on the left axis.
    """
    all_dists = np.concatenate(
        [nn_dists_icl]
        + ([nn_dists_fm] if nn_dists_fm is not None else [])
        + ([nn_dists_icl_self] if nn_dists_icl_self is not None else [])
    )
    bins = np.linspace(all_dists.min(), all_dists.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_w = bins[1] - bins[0]

    icl_idx = np.clip(np.digitize(nn_dists_icl, bins) - 1, 0, n_bins - 1)
    icl_counts = np.array([(icl_idx == i).sum() for i in range(n_bins)])
    icl_density = icl_counts / icl_counts.max()
    bin_means  = np.array([
        y_vals[icl_idx == i].mean() if (icl_idx == i).sum() > 0 else np.nan
        for i in range(n_bins)
    ])
    bin_stds   = np.array([
        y_vals[icl_idx == i].std()  if (icl_idx == i).sum() > 1 else 0.0
        for i in range(n_bins)
    ])

    with mpl.rc_context(_PAPER_RC):
        fig, ax1 = plt.subplots(figsize=(6.5, 3.5))

        _stairs(ax1, icl_density, bins, _BLUE, "TALENT datasets")

        if nn_dists_fm is not None:
            fm_idx = np.clip(np.digitize(nn_dists_fm, bins) - 1, 0, n_bins - 1)
            fm_counts = np.array([(fm_idx == i).sum() for i in range(n_bins)])
            _stairs(ax1, fm_counts / fm_counts.max(), bins, _ORANGE, "Curated (FM) tables")

        if nn_dists_icl_self is not None:
            si_idx = np.clip(np.digitize(nn_dists_icl_self, bins) - 1, 0, n_bins - 1)
            si_counts = np.array([(si_idx == i).sum() for i in range(n_bins)])
            _stairs(ax1, si_counts / si_counts.max(), bins, _PURPLE, "TabICL Prior (self)")

        ax1.set_xlabel("Mean k-NN distance (normalised)")
        ax1.set_ylabel("Relative density")

        ax2 = ax1.twinx()
        mask = ~np.isnan(bin_means)
        lo = np.clip(bin_means - bin_stds, 1, None) if invert_y else bin_means - bin_stds
        hi = bin_means + bin_stds
        ax2.plot(bin_centers[mask], bin_means[mask],
                 color=_RED, lw=1.5, marker="o", ms=4, label=f"Mean {y_label}")
        ax2.fill_between(bin_centers[mask], lo[mask], hi[mask], color=_RED, alpha=0.25)
        if invert_y:
            ax2.invert_yaxis()
        ax2.set_ylabel(f"TabICL mean {y_label}", color=_RED)
        ax2.tick_params(axis="y", labelcolor=_RED)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.title(f"NN Distance to TabICL Prior vs {y_label}")
        plt.tight_layout()
        out = output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({filename.rsplit(".", 1)[0]: wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------------
# Plot 2 — scatter with correlations
# ------------------------------------------------------------------

def plot_metric_vs_distance(
    nn_dists: np.ndarray,
    y_vals: np.ndarray,
    y_label: str,
    filename: str,
    output_dir: Path,
    run=None,
) -> dict:
    """Scatter: metric vs NN distance, with Pearson r and Spearman ρ."""
    r_p, p_p = stats.pearsonr(nn_dists, y_vals)
    r_s, p_s = stats.spearmanr(nn_dists, y_vals)

    m, b = np.polyfit(nn_dists, y_vals.astype(float), 1)
    xs = np.linspace(nn_dists.min(), nn_dists.max(), 200)

    with mpl.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.scatter(nn_dists, y_vals, alpha=0.65, s=25,
                   color=_BLUE, edgecolors="white", linewidths=0.3, zorder=3)
        ax.plot(xs, m * xs + b, color=_ORANGE, lw=1.5, label="Linear fit", zorder=4)
        ax.set_xlabel("Mean k-NN distance to TabICL distribution (normalised)")
        ax.set_ylabel(f"TabICL {y_label}")
        ax.set_title(
            f"{y_label} vs NN Distance\n"
            f"Pearson r = {r_p:.3f} (p = {p_p:.3f})   "
            f"Spearman ρ = {r_s:.3f} (p = {p_s:.3f})"
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


# ------------------------------------------------------------------
# Plot 3 — linear regression diagnostics
# ------------------------------------------------------------------

def plot_metric_prediction(
    nn_dists: np.ndarray,
    y_vals: np.ndarray,
    y_label: str,
    filename: str,
    output_dir: Path,
    run=None,
) -> dict:
    """Linear regression of metric ~ NN distance with actual-vs-predicted and residuals."""
    X = nn_dists.reshape(-1, 1)
    y = y_vals.astype(float)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    residuals = y - y_pred

    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))

        ax = axes[0]
        lo, hi = y.min(), y.max()
        ax.scatter(y, y_pred, alpha=0.6, s=25, color=_GREEN,
                   edgecolors="white", linewidths=0.3)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="Perfect fit")
        ax.set_xlabel(f"Actual {y_label}")
        ax.set_ylabel(f"Predicted {y_label}")
        ax.set_title(f"Linear Regression  R² = {r2:.3f}")
        ax.legend()

        ax = axes[1]
        ax.scatter(y_pred, residuals, alpha=0.6, s=25, color=_ORANGE,
                   edgecolors="white", linewidths=0.3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(f"Predicted {y_label}")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals")

        plt.tight_layout()
        out = output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({filename.rsplit(".", 1)[0]: wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")

    return {"r2": r2, "coef": float(model.coef_[0]), "intercept": float(model.intercept_)}


# ------------------------------------------------------------------
# Plot 4 — side-by-side model comparison scatter
# ------------------------------------------------------------------

def plot_comparison_scatter(
    nn_dists: np.ndarray,
    y_tabicl: np.ndarray,
    y_autogluon: np.ndarray,
    y_label: str,
    filename: str,
    output_dir: Path,
    run=None,
) -> None:
    """Overlay TabICL and AutoGluon rank/metric vs NN distance on one plot."""
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
            ax.scatter(xm, ym, alpha=0.50, s=20, color=color,
                       edgecolors="white", linewidths=0.3, zorder=3,
                       label=f"{label}  r={r_p:.2f}  ρ={r_s:.2f}")
            ax.plot(xs, m * xs + b, color=color, lw=1.5, zorder=4)

        ax.set_xlabel("Mean k-NN distance to TabICL distribution (normalised)")
        ax.set_ylabel(y_label)
        ax.set_title(f"TabICL vs AutoGluon — {y_label} vs NN Distance")
        ax.legend()
        plt.tight_layout()
        out = output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        if run is not None:
            run.log({filename.rsplit(".", 1)[0]: wandb.Image(str(out))})
        plt.close(fig)
    print(f"  Saved {out}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(
    talent_dir: Path,
    tabicl_dir: Path,
    mode: str,
    k: int,
    n_bins: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    talent_cache = _ROOT / f"cache/talent_feats_{MAX_ROWS}.npz"

    # ── Rankings (tabicl-rankings-full-context.csv) ────────────────
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
    print(f"\n=== Rankings: {len(datasets)} datasets with tabicl entries ===")

    # ── Autogluon rankings (results-autogluon-512-context.csv) ─────
    print("\n=== Loading AutoGluon ranks ===")
    ag_ranks = load_autogluon_ranks(AUTOGLUON_RESULTS_CSV)

    # ── TALENT feature vectors ─────────────────────────────────────
    print("\n=== Loading TALENT datasets ===")
    records = load_talent_feats(talent_dir, datasets, talent_cache)
    if not records:
        print("ERROR: no TALENT datasets could be featurised — check --talent-dir")
        sys.exit(1)

    talent_feats = np.stack([r["feats"] for r in records])
    talent_names = [r["dataset"] for r in records]
    print(f"  Using {len(talent_names)} TALENT datasets  shape={talent_feats.shape}")

    # ── TabICL feature vectors ─────────────────────────────────────
    print("\n=== Loading TabICL tables ===")
    icl_feats = load_icl_feats(ICL_CACHE, tabicl_dir, N_ICL, rng)
    print(f"  Using {len(icl_feats)} TabICL feature vectors  shape={icl_feats.shape}")

    # ── FM feature vectors ─────────────────────────────────────────
    fm_feats = None
    if FM_CACHE.exists():
        arr = np.load(FM_CACHE)["feats"]
        idx = rng.choice(len(arr), size=min(N_ICL, len(arr)), replace=False)
        fm_feats = arr[idx]
        print(f"  Loaded {len(fm_feats)} FM feature vectors from cache (of {len(arr)} total)")
    else:
        print(f"  FM cache not found at {FM_CACHE}, skipping FM distances")

    # ── NN distances ───────────────────────────────────────────────
    print(f"\n=== Computing {k}-NN distances (mode={mode}) ===")
    nn_dists = compute_nn_distances(talent_feats, icl_feats, mode, k)
    nn_dists_fm = compute_nn_distances(fm_feats, icl_feats, mode, k) if fm_feats is not None else None

    # ICL self-distances: split in half to avoid self-match
    half = len(icl_feats) // 2
    nn_dists_icl_self = compute_nn_distances(icl_feats[:half], icl_feats[half:], mode, k)
    print(f"  Computed ICL self-distances ({half} query vs {len(icl_feats) - half} ref)")

    # ── Merge with ranks ───────────────────────────────────────────
    df = pd.DataFrame({"dataset": talent_names, "nn_dist_icl": nn_dists})
    df = df.merge(tabicl_ranks, on="dataset", how="inner")
    df = df.merge(ag_ranks, on="dataset", how="left")
    print(f"  Matched {len(df)} datasets to rankings "
          f"({df['ag_rank'].notna().sum()} with autogluon rank)")

    csv_path = output_dir / "rank_vs_nndist.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # ── wandb ──────────────────────────────────────────────────────
    run = None
    if _WANDB_AVAILABLE and WANDB_PROJECT:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"rank_coverage_{mode}_k{k}",
            config={
                "mode": mode, "k": k, "n_bins": n_bins,
                "n_talent": len(df), "n_icl": len(icl_feats),
                "max_rows": MAX_ROWS,
            },
            reinit=True,
        )

    # ── Plots ──────────────────────────────────────────────────────
    print("\n=== Generating figures ===")
    nn       = df["nn_dist_icl"].values
    rk       = df["rank"].values.astype(float)
    auc      = df["AUC"].values.astype(float)
    rel_auc  = df["relative_auc"].values.astype(float)
    ag_rk    = df["ag_rank"].values.astype(float)       # NaN where autogluon absent
    tc_rk_ag = df["tabicl_rank_with_ag"].values.astype(float)  # tabicl rank in 8-model field

    # TabICL plots (7-model ranking)
    plot_histogram_with_metric(nn, rk, "rank (lower = better)", "rank_vs_nndist_histogram.png",
                               output_dir, run, n_bins, nn_dists_fm=nn_dists_fm,
                               nn_dists_icl_self=nn_dists_icl_self, invert_y=True)
    plot_histogram_with_metric(nn, auc, "AUC", "auc_vs_nndist_histogram.png",
                               output_dir, run, n_bins, nn_dists_fm=nn_dists_fm,
                               nn_dists_icl_self=nn_dists_icl_self, invert_y=False)
    plot_histogram_with_metric(nn, rel_auc, "relative AUC", "relative_auc_vs_nndist_histogram.png",
                               output_dir, run, n_bins, nn_dists_fm=nn_dists_fm,
                               nn_dists_icl_self=nn_dists_icl_self, invert_y=False)

    corr     = plot_metric_vs_distance(nn, rk, "rank", "rank_vs_nndist_scatter.png", output_dir, run)
    plot_metric_vs_distance(nn, auc, "AUC", "auc_vs_nndist_scatter.png", output_dir, run)
    corr_rel = plot_metric_vs_distance(nn, rel_auc, "relative AUC", "relative_auc_vs_nndist_scatter.png", output_dir, run)

    pred     = plot_metric_prediction(nn, rk, "rank", "rank_prediction.png", output_dir, run)
    plot_metric_prediction(nn, auc, "AUC", "auc_prediction.png", output_dir, run)
    pred_rel = plot_metric_prediction(nn, rel_auc, "relative AUC", "relative_auc_prediction.png", output_dir, run)

    # AutoGluon plots (8-model ranking, datasets with autogluon results only)
    ag_mask = ~np.isnan(ag_rk)
    if ag_mask.sum() >= 2:
        nn_ag = nn[ag_mask]
        plot_histogram_with_metric(
            nn_ag, ag_rk[ag_mask], "AutoGluon rank (lower = better)",
            "ag_rank_vs_nndist_histogram.png", output_dir, run, n_bins,
            nn_dists_icl_self=nn_dists_icl_self, invert_y=True,
        )
        corr_ag = plot_metric_vs_distance(
            nn_ag, ag_rk[ag_mask], "AutoGluon rank",
            "ag_rank_vs_nndist_scatter.png", output_dir, run,
        )
        pred_ag = plot_metric_prediction(
            nn_ag, ag_rk[ag_mask], "AutoGluon rank",
            "ag_rank_prediction.png", output_dir, run,
        )
        # Comparison scatter: tabicl-with-ag vs autogluon rank on common datasets
        plot_comparison_scatter(
            nn_ag, tc_rk_ag[ag_mask], ag_rk[ag_mask],
            "rank (lower = better, 8-model field)",
            "tabicl_vs_ag_rank_scatter.png", output_dir, run,
        )
    else:
        corr_ag = pred_ag = None
        print("  Skipping autogluon plots (insufficient matched datasets)")

    # ── Summary ────────────────────────────────────────────────────
    print("\n=== Results (TabICL rank, 7-model field) ===")
    print(f"  n datasets       : {len(df)}")
    print(f"  Pearson  r       : {corr['pearson_r']:.3f}  (p = {corr['pearson_p']:.4f})")
    print(f"  Spearman ρ       : {corr['spearman_r']:.3f}  (p = {corr['spearman_p']:.4f})")
    print(f"  Linear R²        : {pred['r2']:.3f}")
    print(f"  Linear coef      : {pred['coef']:.4f}  (intercept = {pred['intercept']:.4f})")
    print("\n=== Results (relative AUC = AUC(tabicl)/mean(all AUCs)) ===")
    print(f"  Pearson  r       : {corr_rel['pearson_r']:.3f}  (p = {corr_rel['pearson_p']:.4f})")
    print(f"  Spearman ρ       : {corr_rel['spearman_r']:.3f}  (p = {corr_rel['spearman_p']:.4f})")
    print(f"  Linear R²        : {pred_rel['r2']:.3f}")
    print(f"  Linear coef      : {pred_rel['coef']:.4f}  (intercept = {pred_rel['intercept']:.4f})")
    if corr_ag is not None:
        print("\n=== Results (AutoGluon rank, 8-model field) ===")
        print(f"  n datasets       : {ag_mask.sum()}")
        print(f"  Pearson  r       : {corr_ag['pearson_r']:.3f}  (p = {corr_ag['pearson_p']:.4f})")
        print(f"  Spearman ρ       : {corr_ag['spearman_r']:.3f}  (p = {corr_ag['spearman_p']:.4f})")
        print(f"  Linear R²        : {pred_ag['r2']:.3f}")
        print(f"  Linear coef      : {pred_ag['coef']:.4f}  (intercept = {pred_ag['intercept']:.4f})")

    if run is not None:
        log_dict = {
            "pearson_r":       corr["pearson_r"],
            "pearson_p":       corr["pearson_p"],
            "spearman_r":      corr["spearman_r"],
            "spearman_p":      corr["spearman_p"],
            "linear_r2":       pred["r2"],
            "linear_coef":     pred["coef"],
            "rel_pearson_r":   corr_rel["pearson_r"],
            "rel_pearson_p":   corr_rel["pearson_p"],
            "rel_spearman_r":  corr_rel["spearman_r"],
            "rel_spearman_p":  corr_rel["spearman_p"],
            "rel_linear_r2":   pred_rel["r2"],
            "rel_linear_coef": pred_rel["coef"],
        }
        if corr_ag is not None:
            log_dict.update({
                "ag_pearson_r":  corr_ag["pearson_r"],
                "ag_pearson_p":  corr_ag["pearson_p"],
                "ag_spearman_r": corr_ag["spearman_r"],
                "ag_spearman_p": corr_ag["spearman_p"],
                "ag_linear_r2":  pred_ag["r2"],
                "ag_linear_coef": pred_ag["coef"],
            })
        run.log(log_dict)
        run.finish()

    print(f"\nOutputs in: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correlation between NN distance to TabICL prior and TabICL benchmark rank."
    )
    parser.add_argument("--talent-dir", type=Path, default=TALENT_DIR,
                        help="Root directory of TALENT benchmark datasets (default: %(default)s)")
    parser.add_argument("--tabicl-dir", type=Path, default=TABICL_DIR,
                        help="Directory of tabICL .parquet files (default: %(default)s)")
    parser.add_argument("--mode", choices=list(FEATURE_MODES), default=FEATURE_MODE,
                        help="Feature mode for NN distance (default: %(default)s)")
    parser.add_argument("--k", type=int, default=KNN_K,
                        help="Number of nearest neighbours (default: %(default)s)")
    parser.add_argument("--n-bins", type=int, default=30,
                        help="Histogram bins for plot 1 (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory for figures and CSV (default: %(default)s)")
    args = parser.parse_args()

    main(
        talent_dir=args.talent_dir,
        tabicl_dir=args.tabicl_dir,
        mode=args.mode,
        k=args.k,
        n_bins=args.n_bins,
        output_dir=args.output_dir,
    )
