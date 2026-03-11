"""Coverage analysis: tabICL prior vs real t4 data.

Generates synthetic feature vectors for the top-k grid-search configurations
and runs five analyses per config, each logged as a separate wandb run:

  1. Per-feature KS tests   — which of the 70 df_to_vec dimensions differ most
  2. Feature importance      — XGBoost discriminator gain per feature
  3. Bootstrap AUC CI       — exact CI by resampling + retraining (no Hanley-McNeil assumptions)
  4. Coverage (k-NN)        — fraction of real samples with no nearby synthetic neighbour
  5. PCA projection         — 2-D visualisation of the real / synthetic separation

When all three feature modes are run (the default), a combined 3-row figure is
also produced for each measure so the modes can be compared side-by-side.

Usage
-----
    python coverage_analysis.py [--config grid-results-35275/combined_results_t4.csv]
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from xgboost import XGBClassifier

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.discriminator.features import df_to_vec

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
RESULTS_CSV   = "grid-results-35275/combined_results_t4.csv"
CACHE_DIR     = Path("cache")
SOURCE_NAME   = "t4"
N_REAL_CACHE  = 5000
MAX_ROWS      = 512

N_SYNTH       = 1000    # synthetic tables to generate for analysis
N_REAL_SAMPLE = 1000    # real tables to sample from cache (matched to N_SYNTH)
BATCH_SIZE    = 8
TEST_FRAC     = 0.3
RANDOM_STATE  = 42

N_BOOT        = 200     # bootstrap iterations for AUC CI
BOOT_ALPHA    = 0.005   # 99.5% CI

KNN_K         = 5       # neighbours for coverage analysis
# A real sample is "uncovered" if its distance to the nearest synthetic
# neighbour exceeds this quantile of within-synthetic k-NN distances.
COVERAGE_THRESHOLD_QUANTILE = 0.95

TOP_K         = 5       # number of top configs to analyse

WANDB_PROJECT = "tabicl-coverage-analysis"
OUTPUT_DIR    = Path("coverage-results")

# ------------------------------------------------------------------
# Feature names (matches tabdistances/discriminator/features.py layout)
# ------------------------------------------------------------------
_SCALAR_NAMES = [
    "n_cols", "type_ratio", "missing_rate",
    "mean_col_skew", "std_col_skew",
    "mean_col_kurt", "std_col_kurt",
    "mean_abs_corr", "std_abs_corr", "frac_high_corr",
]
FEATURE_NAMES = (
    _SCALAR_NAMES
    + [f"hist_mean[{i}]" for i in range(30)]
    + [f"hist_std[{i}]"  for i in range(30)]
)

# Feature modes: indices to keep per mode.
# "scalar_only" drops n_cols (idx 0) and all histogram features (idx 10-69),
# leaving the 9 distributional scalars that are independent of table size/shape.
# "hist_only" keeps only the 60 histogram features (idx 10-69).
_SCALAR_ONLY_IDX = list(range(1, 10))
_HIST_ONLY_IDX   = list(range(10, 70))
FEATURE_MODES = {
    "full":        list(range(len(FEATURE_NAMES))),
    "scalar_only": _SCALAR_ONLY_IDX,
    "hist_only":   _HIST_ONLY_IDX,
}


def _select_features(feats: np.ndarray, mode: str) -> tuple[np.ndarray, list[str]]:
    """Return (filtered_feats, filtered_names) for the given mode."""
    idx = FEATURE_MODES[mode]
    return feats[:, idx], [FEATURE_NAMES[i] for i in idx]


# ------------------------------------------------------------------
# GPU detection (mirrors basic-grid-search.py)
# ------------------------------------------------------------------
def _xgb_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

XGB_DEVICE = _xgb_device()


# ------------------------------------------------------------------
# Synthetic generation
# ------------------------------------------------------------------
def generate_synthetic_features(params: dict, n: int) -> np.ndarray:
    prior = SCMPrior(
        batch_size=BATCH_SIZE,
        min_features=int(params["min_features"]),
        max_features=int(params["max_features"]),
        max_classes=int(params["max_classes"]),
        max_seq_len=MAX_ROWS,
        log_seq_len=True,
        min_train_size=0.1,
        max_train_size=0.9,
        prior_type=params["prior_type"],
        replay_small=bool(params["replay_small"]),
        fixed_hp={
            "tree_model":               params["tree_model"],
            "mix_probas":               (float(params["mix_prob_mlp"]),
                                         1.0 - float(params["mix_prob_mlp"])),
            "cat_prob":                 float(params["cat_prob"]),
            "multiclass_ordered_prob":  float(params["multiclass_ordered_prob"]),
            "balanced":                 bool(params["balanced"]),
            "tree_depth_lambda":        float(params["tree_depth_lambda"]),
            "tree_n_estimators_lambda": float(params["tree_n_estimators_lambda"]),
        },
        n_jobs=1,
        device="cpu",
    )

    feats = []
    with tqdm(total=n, desc="generating synthetic tables", unit="table") as pbar:
        while len(feats) < n:
            X, y, d, seq_lens, _ = prior.get_batch()
            for i in range(X.shape[0]):
                n_feat  = int(d[i].item())
                seq_len = int(seq_lens[i].item())
                if n_feat < int(params["min_features"]) or seq_len == 0:
                    continue
                X_i = X[i, :seq_len, :n_feat].numpy().astype(float)
                y_i = y[i, :seq_len].numpy().astype(int)
                df_i = pd.DataFrame(X_i, columns=[f"f{j}" for j in range(n_feat)])
                df_i["target"] = y_i
                feats.append(df_to_vec(df_i))
                pbar.update(1)
                if len(feats) >= n:
                    break

    return np.array(feats)


# ------------------------------------------------------------------
# Drawing helpers — each renders onto caller-supplied axes
# ------------------------------------------------------------------

def _draw_per_feature_ks(ax, df_ks: pd.DataFrame, title_suffix: str = "") -> None:
    top20 = df_ks.head(20)
    colors = ["#e74c3c" if r > 0 else "#2ecc71" for r in top20["rank_biserial_r"]]
    bars = ax.barh(top20["feature"][::-1], top20["ks_stat"][::-1],
                   color=colors[::-1], alpha=0.85)
    ax.set_xlabel("KS statistic  (higher = larger distributional gap)")
    title = "Top-20 KS: real vs synthetic (red = real > synth)"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title)
    for bar, row in zip(bars, top20.iloc[::-1].itertuples()):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"r={row.rank_biserial_r:.2f}", va="center", fontsize=7)


def _draw_bootstrap_auc(ax, boot_aucs: np.ndarray, ci_lo: float, ci_hi: float,
                        title_suffix: str = "") -> None:
    ax.hist(boot_aucs, bins=30, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(ci_lo,  color="red",   linestyle="--",
               label=f"{100*(1-BOOT_ALPHA):.1f}% CI: ({ci_lo:.4f}, {ci_hi:.4f})")
    ax.axvline(ci_hi,  color="red",   linestyle="--")
    ax.axvline(boot_aucs.mean(), color="orange", linestyle="-",
               label=f"mean={boot_aucs.mean():.4f}")
    ax.axvline(0.5,    color="black", linestyle=":",  label="AUC=0.5 (chance)")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Bootstrap count")
    title = f"Bootstrap AUC  (n_boot={N_BOOT})"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title)
    ax.legend(fontsize=8)


def _draw_coverage_dists(ax, real_to_synth: np.ndarray, synth_internal: np.ndarray,
                         threshold: float, title_suffix: str = "") -> None:
    ax.hist(synth_internal.mean(axis=1), bins=50, alpha=0.6,
            color="steelblue", label="within-synthetic", edgecolor="white")
    ax.hist(real_to_synth, bins=50, alpha=0.6,
            color="salmon", label="real → nearest synthetic", edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"threshold (p{COVERAGE_THRESHOLD_QUANTILE*100:.0f}): {threshold:.3f}")
    ax.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
    ax.set_ylabel("Count")
    title = "k-NN distance distributions"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title)
    ax.legend(fontsize=8)


def _draw_coverage_curve(ax, real_to_synth: np.ndarray, threshold: float,
                         uncovered: float, title_suffix: str = "") -> None:
    ax.plot(np.sort(real_to_synth), np.linspace(0, 1, len(real_to_synth)), color="salmon")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold: {threshold:.3f}")
    ax.axhline(1 - uncovered, color="steelblue", linestyle=":",
               label=f"covered fraction: {1-uncovered:.3f}")
    ax.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
    ax.set_ylabel("Cumulative fraction of real samples")
    title = "Coverage curve"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title)
    ax.legend(fontsize=8)


def _draw_feature_importance(ax, imp: pd.DataFrame, auc_val: float,
                             title_suffix: str = "") -> None:
    top30 = imp.head(30)
    bars = ax.barh(top30["feature"][::-1], top30["gain_norm"][::-1],
                   color="steelblue", alpha=0.85)
    ax.set_xlabel("Normalised gain (fraction of total discriminator gain)")
    title = f"Feature importance — discriminator gain  (AUC={auc_val:.4f})"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title)
    for bar, val in zip(bars, top30["gain_norm"][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)


def _draw_pca(ax, X_2d: np.ndarray, labels: np.ndarray, var: np.ndarray,
              title_suffix: str = "") -> None:
    for label, color, alpha in [("real", "#e74c3c", 0.5), ("synthetic", "#2980b9", 0.35)]:
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=label, alpha=alpha, s=8, linewidths=0)
    ax.set_xlabel(f"PC1  ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2  ({var[1]:.1%} variance)")
    title = "PCA: real vs synthetic feature vectors"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=9)


# ------------------------------------------------------------------
# Analysis 1: per-feature KS tests
# ------------------------------------------------------------------
def analyse_per_feature(real: np.ndarray, synth: np.ndarray, run, tag: str = "",
                        feature_names: list[str] = FEATURE_NAMES
                        ) -> tuple[pd.DataFrame, dict]:
    print("\n── Per-feature KS tests ──")
    rows = []
    for i, name in enumerate(feature_names):
        ks_stat, p_val = stats.ks_2samp(real[:, i], synth[:, i])
        u_stat, _ = stats.mannwhitneyu(real[:, i], synth[:, i], alternative="two-sided")
        r = 1 - (2 * u_stat) / (len(real) * len(synth))
        rows.append({"feature": name, "ks_stat": ks_stat, "p_value": p_val,
                     "rank_biserial_r": r, "abs_r": abs(r)})

    df_ks = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_per_feature_ks(ax, df_ks, tag)
    plt.tight_layout()
    _save_and_log(fig, "per_feature_ks", run, tag)

    if run is not None:
        run.log({"per_feature/mean_ks": df_ks["ks_stat"].mean(),
                 "per_feature/max_ks":  df_ks["ks_stat"].max(),
                 "per_feature/n_significant_bonf": (df_ks["p_value"] * len(df_ks) < 0.005).sum(),
                 "per_feature/table": wandb.Table(dataframe=df_ks.round(4))})

    print(df_ks.head(10).to_string(index=False))
    return df_ks, {"df_ks": df_ks}


# ------------------------------------------------------------------
# Analysis 2: bootstrap AUC CI
# ------------------------------------------------------------------
def analyse_bootstrap_auc(real: np.ndarray, synth: np.ndarray, run,
                           tag: str = "") -> tuple[dict, dict]:
    print(f"\n── Bootstrap AUC CI  (n_boot={N_BOOT}, α={BOOT_ALPHA}) ──")
    rng = np.random.default_rng(RANDOM_STATE)
    boot_aucs = []

    for _ in tqdm(range(N_BOOT), desc="bootstrap iterations"):
        r_idx = rng.integers(0, len(real),  len(real),  endpoint=False)
        s_idx = rng.integers(0, len(synth), len(synth), endpoint=False)
        X_all = np.vstack([synth[s_idx], real[r_idx]])
        y_all = np.array([0] * len(synth) + [1] * len(real))

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=TEST_FRAC, random_state=int(rng.integers(1e6)),
            stratify=y_all)

        clf = XGBClassifier(n_estimators=300, device=XGB_DEVICE,
                            eval_metric="logloss", verbosity=0,
                            random_state=int(rng.integers(1e6)))
        clf.fit(X_tr, y_tr)
        boot_aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))

    boot_aucs = np.array(boot_aucs)
    pct_lo = 100 * BOOT_ALPHA / 2
    pct_hi = 100 * (1 - BOOT_ALPHA / 2)
    ci_lo, ci_hi = np.percentile(boot_aucs, [pct_lo, pct_hi])

    print(f"  Bootstrap {100*(1-BOOT_ALPHA):.1f}% CI: ({ci_lo:.4f}, {ci_hi:.4f})")
    print(f"  Mean: {boot_aucs.mean():.4f}   Std: {boot_aucs.std():.4f}")
    print(f"  Lower bound > 0.5: {ci_lo > 0.5}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    _draw_bootstrap_auc(ax, boot_aucs, ci_lo, ci_hi, tag)
    plt.tight_layout()
    _save_and_log(fig, "bootstrap_auc", run, tag)

    result = {"ci_lo": ci_lo, "ci_hi": ci_hi,
              "mean": boot_aucs.mean(), "std": boot_aucs.std(),
              "lower_bound_above_0.5": bool(ci_lo > 0.5)}
    if run is not None:
        run.log({f"bootstrap/{k}": v for k, v in result.items()})
    return result, {"boot_aucs": boot_aucs, "ci_lo": ci_lo, "ci_hi": ci_hi}


# ------------------------------------------------------------------
# Analysis 3: coverage (k-NN)
# ------------------------------------------------------------------
def analyse_coverage(real: np.ndarray, synth: np.ndarray, run,
                     tag: str = "") -> tuple[dict, dict]:
    print(f"\n── Coverage analysis  (k={KNN_K}) ──")

    lo  = np.minimum(real.min(0), synth.min(0))
    hi  = np.maximum(real.max(0), synth.max(0))
    rng = hi - lo
    rng[rng == 0] = 1.0
    real_n  = (real  - lo) / rng
    synth_n = (synth - lo) / rng

    nn_ss = NearestNeighbors(n_neighbors=KNN_K + 1, algorithm="ball_tree").fit(synth_n)
    dists_ss, _ = nn_ss.kneighbors(synth_n)
    synth_internal = dists_ss[:, 1:]  # exclude self
    threshold = np.percentile(synth_internal.mean(axis=1), COVERAGE_THRESHOLD_QUANTILE * 100)

    nn_rs = NearestNeighbors(n_neighbors=KNN_K, algorithm="ball_tree").fit(synth_n)
    dists_rs, _ = nn_rs.kneighbors(real_n)
    real_to_synth = dists_rs.mean(axis=1)

    uncovered = (real_to_synth > threshold).mean()
    print(f"  Coverage threshold (p{COVERAGE_THRESHOLD_QUANTILE*100:.0f} within-synth): {threshold:.4f}")
    print(f"  Fraction of real samples uncovered: {uncovered:.3f}")
    print(f"  Median real→synth distance: {np.median(real_to_synth):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _draw_coverage_dists(axes[0], real_to_synth, synth_internal, threshold, tag)
    _draw_coverage_curve(axes[1], real_to_synth, threshold, uncovered, tag)
    plt.tight_layout()
    _save_and_log(fig, "coverage", run, tag)

    result = {"threshold": threshold, "uncovered_fraction": uncovered,
              "median_real_to_synth_dist": float(np.median(real_to_synth))}
    if run is not None:
        run.log({f"coverage/{k}": v for k, v in result.items()})
    return result, {"real_to_synth": real_to_synth, "synth_internal": synth_internal,
                    "threshold": threshold, "uncovered": uncovered}


# ------------------------------------------------------------------
# Analysis 4: feature importance
# ------------------------------------------------------------------
def analyse_feature_importance(real: np.ndarray, synth: np.ndarray, run, tag: str = "",
                               feature_names: list[str] = FEATURE_NAMES
                               ) -> tuple[pd.DataFrame, dict]:
    print("\n── Feature importance (discriminator gain) ──")
    X_all = np.vstack([synth, real])
    y_all = np.array([0] * len(synth) + [1] * len(real))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=TEST_FRAC, random_state=RANDOM_STATE, stratify=y_all)

    clf = XGBClassifier(n_estimators=300, device=XGB_DEVICE,
                        eval_metric="logloss", verbosity=0,
                        random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    auc_val = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    print(f"  Discriminator AUC on held-out test set: {auc_val:.4f}")

    scores = clf.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame([
        {"feature": feature_names[int(k[1:])], "gain": v}
        for k, v in scores.items()
    ]).sort_values("gain", ascending=False)
    imp["gain_norm"] = imp["gain"] / imp["gain"].sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_feature_importance(ax, imp, auc_val, tag)
    plt.tight_layout()
    _save_and_log(fig, "feature_importance", run, tag)

    if run is not None:
        run.log({"feature_importance/discriminator_auc": auc_val,
                 "feature_importance/table": wandb.Table(dataframe=imp.round(5))})

    print(imp.head(10).to_string(index=False))
    return imp, {"imp": imp, "auc_val": auc_val}


# ------------------------------------------------------------------
# Analysis 5: PCA projection
# ------------------------------------------------------------------
def analyse_pca(real: np.ndarray, synth: np.ndarray, run,
                tag: str = "") -> tuple[None, dict]:
    print("\n── PCA projection ──")
    X_all  = np.vstack([real, synth])
    labels = np.array(["real"] * len(real) + ["synthetic"] * len(synth))

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_all)
    var  = pca.explained_variance_ratio_

    print(f"  PC1 variance explained: {var[0]:.3f}   PC2: {var[1]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_pca(ax, X_2d, labels, var, tag)
    plt.tight_layout()
    _save_and_log(fig, "pca", run, tag)

    if run is not None:
        run.log({"pca/pc1_var_explained": var[0], "pca/pc2_var_explained": var[1]})

    return None, {"X_2d": X_2d, "labels": labels, "var": var}


# ------------------------------------------------------------------
# Combined 3-row plots (one row per feature mode)
# ------------------------------------------------------------------
def plot_combined_all(rank: int, mode_data: dict, tag_base: str, cfg: dict) -> None:
    """Produce a 3-row combined figure for each measure (one row per feature mode)
    and log all five images in a single summary wandb run."""
    modes_ordered = list(FEATURE_MODES.keys())  # full, scalar_only, hist_only

    # Open one summary run for all combined figures for this config rank
    summary_run = None
    if _WANDB_AVAILABLE and WANDB_PROJECT:
        summary_run = wandb.init(
            project=WANDB_PROJECT,
            name=f"combined_{SOURCE_NAME}_{tag_base}",
            config={"source": SOURCE_NAME, "rank": rank + 1, "summary": True,
                    **{k: v for k, v in cfg.items() if k not in ("min_features", "max_features")}},
            reinit=True,
        )

    # ── per_feature_ks ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for ax, mode in zip(axes, modes_ordered):
        _draw_per_feature_ks(ax, mode_data[mode]["ks"]["df_ks"], mode)
    fig.suptitle(f"Per-feature KS — rank {rank+1} (all feature modes)", fontsize=13)
    plt.tight_layout()
    _save_and_log(fig, "combined_per_feature_ks", summary_run, tag_base)

    # ── bootstrap_auc ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(7, 10.5))
    for ax, mode in zip(axes, modes_ordered):
        d = mode_data[mode]["boot"]
        _draw_bootstrap_auc(ax, d["boot_aucs"], d["ci_lo"], d["ci_hi"], mode)
    fig.suptitle(f"Bootstrap AUC — rank {rank+1} (all feature modes)", fontsize=13)
    plt.tight_layout()
    _save_and_log(fig, "combined_bootstrap_auc", summary_run, tag_base)

    # ── coverage (3 rows × 2 cols) ──────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for row_i, mode in enumerate(modes_ordered):
        d = mode_data[mode]["cov"]
        _draw_coverage_dists(axes[row_i, 0], d["real_to_synth"], d["synth_internal"],
                             d["threshold"], mode)
        _draw_coverage_curve(axes[row_i, 1], d["real_to_synth"], d["threshold"],
                             d["uncovered"], mode)
    fig.suptitle(f"Coverage — rank {rank+1} (all feature modes)", fontsize=13)
    plt.tight_layout()
    _save_and_log(fig, "combined_coverage", summary_run, tag_base)

    # ── feature_importance ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for ax, mode in zip(axes, modes_ordered):
        d = mode_data[mode]["imp"]
        _draw_feature_importance(ax, d["imp"], d["auc_val"], mode)
    fig.suptitle(f"Feature importance — rank {rank+1} (all feature modes)", fontsize=13)
    plt.tight_layout()
    _save_and_log(fig, "combined_feature_importance", summary_run, tag_base)

    # ── pca ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(7, 18))
    for ax, mode in zip(axes, modes_ordered):
        d = mode_data[mode]["pca"]
        _draw_pca(ax, d["X_2d"], d["labels"], d["var"], mode)
    fig.suptitle(f"PCA — rank {rank+1} (all feature modes)", fontsize=13)
    plt.tight_layout()
    _save_and_log(fig, "combined_pca", summary_run, tag_base)

    if summary_run is not None:
        summary_run.finish()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _save_and_log(fig: plt.Figure, name: str, run, tag: str = "") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    path = OUTPUT_DIR / f"coverage_{name}{suffix}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Plot saved -> {path}")
    if run is not None:
        run.log({name: wandb.Image(str(path))})
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(results_csv: str, modes: list[str]) -> None:
    # ── Load top-k configs ────────────────────────────────────────
    df_results = pd.read_csv(results_csv)

    print(f"\nLoaded {len(df_results)} grid-search results from {results_csv}")
    top_configs = df_results.nsmallest(TOP_K, "auc").reset_index(drop=True)
    print(f"Top-{TOP_K} configs to analyse:")
    print(top_configs[["auc"] + [c for c in top_configs.columns if c != "auc"]].to_string())

    # ── Load cached real features (shared across all configs) ─────
    cache_path = CACHE_DIR / f"real_feats_{SOURCE_NAME}_{N_REAL_CACHE}_{MAX_ROWS}.npz"
    if not cache_path.exists():
        print(f"ERROR: cache not found at {cache_path}. Run basic-grid-search.py first.")
        sys.exit(1)
    all_real = np.load(cache_path)["feats"]
    rng = np.random.default_rng(RANDOM_STATE)
    real_idx = rng.choice(len(all_real), size=min(N_REAL_SAMPLE, len(all_real)), replace=False)
    real_feats = all_real[real_idx]
    print(f"\nReal features: {real_feats.shape}  (sampled from {len(all_real)} cached)")

    # Derive feature range from full real cache (mirrors basic-grid-search.py)
    col_counts = all_real[:, 0].astype(int)
    fixed = {"min_features": int(col_counts.min()), "max_features": int(col_counts.max())}

    all_modes_run = set(modes) == set(FEATURE_MODES.keys())

    # ── Loop over configs × feature modes ────────────────────────
    for rank, row in top_configs.iterrows():
        cfg = row.to_dict()
        cfg.update(fixed)

        print(f"\n{'#'*60}")
        print(f"  Config rank {rank + 1}/{TOP_K}  (AUC={cfg['auc']:.4f})")
        print(f"{'#'*60}")

        print(f"\nGenerating {N_SYNTH} synthetic tables ...")
        synth_feats_full = generate_synthetic_features(cfg, N_SYNTH)
        print(f"Synthetic features: {synth_feats_full.shape}")

        # Accumulate per-mode plot data for optional combined figures
        mode_data: dict[str, dict] = {}

        for mode in modes:
            tag = f"rank{rank + 1}_{mode}"
            real_m,  fnames = _select_features(real_feats,       mode)
            synth_m, _      = _select_features(synth_feats_full, mode)

            print(f"\n  ── mode={mode}  features={real_m.shape[1]}  [{tag}] ──")

            run = None
            if _WANDB_AVAILABLE and WANDB_PROJECT:
                run = wandb.init(
                    project=WANDB_PROJECT,
                    name=f"coverage_{SOURCE_NAME}_{tag}",
                    config={
                        "source":         SOURCE_NAME,
                        "rank":           rank + 1,
                        "feature_mode":   mode,
                        "n_features":     real_m.shape[1],
                        "n_synth":        N_SYNTH,
                        "n_real_sample":  N_REAL_SAMPLE,
                        "n_boot":         N_BOOT,
                        "boot_alpha":     BOOT_ALPHA,
                        "knn_k":          KNN_K,
                        "coverage_threshold_quantile": COVERAGE_THRESHOLD_QUANTILE,
                        **{k: v for k, v in cfg.items() if k not in ("min_features", "max_features")},
                    },
                    reinit=True,
                )

            _, ks_data   = analyse_per_feature(real_m, synth_m, run, tag, fnames)
            _, imp_data  = analyse_feature_importance(real_m, synth_m, run, tag, fnames)
            _, boot_data = analyse_bootstrap_auc(real_m, synth_m, run, tag)
            _, cov_data  = analyse_coverage(real_m, synth_m, run, tag)
            _, pca_data  = analyse_pca(real_m, synth_m, run, tag)

            mode_data[mode] = {
                "ks":   ks_data,
                "imp":  imp_data,
                "boot": boot_data,
                "cov":  cov_data,
                "pca":  pca_data,
            }

            if run is not None:
                run.finish()

        # ── Combined figures (only when all three modes were run) ──
        if all_modes_run:
            print(f"\n  ── Producing combined 3-mode figures for rank {rank+1} ──")
            plot_combined_all(rank, mode_data, f"rank{rank + 1}", cfg)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=RESULTS_CSV,
                        help="Path to grid results CSV (default: %(default)s)")
    parser.add_argument("--mode", nargs="+", choices=list(FEATURE_MODES.keys()),
                        default=list(FEATURE_MODES.keys()),
                        help="Feature mode(s) to run (default: all)")
    args = parser.parse_args()
    main(args.config, args.mode)
