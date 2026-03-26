"""Coverage analysis: tabICL prior vs real t4 data.

Generates synthetic feature vectors for the top-k grid-search configurations
and runs five analyses per config, each logged as a separate wandb run:

  1. Per-feature KS tests   — which of the 70 df_to_vec dimensions differ most
  2. Feature importance      — XGBoost discriminator gain per feature
  3. Bootstrap AUC CI       — exact CI by resampling + retraining (no Hanley-McNeil assumptions)
  4. Coverage (k-NN)        — fraction of real samples with no nearby synthetic neighbour
  5. PCA projection         — 2-D visualisation of the real / synthetic separation

When all three feature modes are run (the default), a combined 3-column figure is
also produced for each measure so the modes can be compared side-by-side.
Combined figures are sized for inclusion in a US letter-paper academic paper.

Usage
-----
    python coverage_analysis.py [--config grid-results-35275/combined_results_t4.csv]
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

# Display labels for the two sides in plots
_LABEL_REAL  = "Web-Scraped"   # t4 real data
_LABEL_SYNTH = "Synthetic"     # tabICL prior

# rcParams for combined figures intended for US letter-paper inclusion
# (full text width ≈ 6.5 in, 1-in margins)
_PAPER_RC = {
    "font.size":       7,
    "axes.titlesize":  8,
    "axes.labelsize":  7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.5,
    "figure.dpi":      300,
}

from tabdistances.discriminator.features import (
    select_features as _select_features,
    FEATURE_MODES,
    FEATURE_NAMES,
)


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

def _draw_per_feature_ks(ax, df_ks: pd.DataFrame, title_suffix: str = "",
                         top_n: int = 20) -> None:
    top = df_ks.head(top_n)
    colors = ["#e74c3c" if r > 0 else "#2ecc71" for r in top["rank_biserial_r"]]
    bars = ax.barh(top["feature"][::-1], top["ks_stat"][::-1],
                   color=colors[::-1], alpha=0.85)
    ax.set_xlabel("KS statistic  (higher = larger distributional gap)")
    title = f"Top-{top_n} KS: real vs synthetic (red = real > synth)"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title)
    for bar, row in zip(bars, top.iloc[::-1].itertuples()):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"r={row.rank_biserial_r:.2f}", va="center",
                fontsize=mpl.rcParams["legend.fontsize"])


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
                         threshold: float, title_suffix: str = "",
                         label_ref: str = "real", label_cand: str = "synthetic") -> None:
    ax.hist(synth_internal.mean(axis=1), bins=50, alpha=0.6,
            color="steelblue", label=f"within-{label_cand}", edgecolor="white")
    ax.hist(real_to_synth, bins=50, alpha=0.6,
            color="salmon", label=f"{label_ref}  $\\in$  nearest {label_cand}", edgecolor="white")
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
                         covered_fraction: float, title_suffix: str = "",
                         label_ref: str = "real") -> None:
    ax.plot(np.sort(real_to_synth), np.linspace(0, 1, len(real_to_synth)), color="salmon")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold: {threshold:.3f}")
    ax.axhline(covered_fraction, color="steelblue", linestyle=":",
               label=f"covered fraction: {covered_fraction:.3f}")
    ax.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
    ax.set_ylabel(f"Cumulative fraction of {label_ref} samples")
    title = "Coverage curve"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title)
    ax.legend(fontsize=8)


def _draw_feature_importance(ax, imp: pd.DataFrame, auc_val: float,
                             title_suffix: str = "", top_n: int = 30) -> None:
    top = imp.head(top_n)
    bars = ax.barh(top["feature"][::-1], top["gain_norm"][::-1],
                   color="steelblue", alpha=0.85)
    ax.set_xlabel("Normalised gain (fraction of total discriminator gain)")
    title = f"Feature importance — discriminator gain  (AUC={auc_val:.4f})"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title)
    for bar, val in zip(bars, top["gain_norm"][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                fontsize=mpl.rcParams["legend.fontsize"])


def _draw_pca(ax, X_2d: np.ndarray, labels: np.ndarray, var: np.ndarray,
              title_suffix: str = "",
              label_ref: str = "real", label_cand: str = "synthetic") -> None:
    for label, color, alpha in [(label_ref, "#e74c3c", 0.5), (label_cand, "#2980b9", 0.35)]:
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=label, alpha=alpha, s=8, linewidths=0)
    ax.set_xlabel(f"PC1  ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2  ({var[1]:.1%} variance)")
    # title = f"PCA: {label_ref} vs {label_cand}"
    # if title_suffix:
    #     title += f"  [{title_suffix}]"
    # ax.set_title(title)
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
                     tag: str = "",
                     label_ref: str = "real", label_cand: str = "synthetic",
                     log_prefix: str = "recall"
                     ) -> tuple[dict, dict]:
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

    covered_fraction = 1.0 - (real_to_synth > threshold).mean()
    print(f"  Coverage threshold (p{COVERAGE_THRESHOLD_QUANTILE*100:.0f} within-{label_cand}): {threshold:.4f}")
    print(f"  Fraction of {label_ref} samples covered: {covered_fraction:.3f}")
    print(f"  Median {label_ref} $\\in$ {label_cand} distance: {np.median(real_to_synth):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _draw_coverage_dists(axes[0], real_to_synth, synth_internal, threshold, tag,
                         label_ref=label_ref, label_cand=label_cand)
    _draw_coverage_curve(axes[1], real_to_synth, threshold, covered_fraction, tag,
                         label_ref=label_ref)
    plt.tight_layout()
    _save_and_log(fig, "coverage", run, tag)

    result = {"threshold": threshold, "covered_fraction": covered_fraction,
              "median_real_to_synth_dist": float(np.median(real_to_synth))}
    if run is not None:
        run.log({f"{log_prefix}/{k}": v for k, v in result.items()})
    return result, {"real_to_synth": real_to_synth, "synth_internal": synth_internal,
                    "threshold": threshold, "covered_fraction": covered_fraction,
                    "label_ref": label_ref, "label_cand": label_cand}


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
                tag: str = "",
                label_ref: str = "real", label_cand: str = "synthetic"
                ) -> tuple[None, dict]:
    print("\n── PCA projection ──")
    X_all  = np.vstack([real, synth])
    labels = np.array([label_ref] * len(real) + [label_cand] * len(synth))

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_all)
    var  = pca.explained_variance_ratio_

    print(f"  PC1 variance explained: {var[0]:.3f}   PC2: {var[1]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_pca(ax, X_2d, labels, var, tag, label_ref=label_ref, label_cand=label_cand)
    plt.tight_layout()
    _save_and_log(fig, "pca", run, tag)

    if run is not None:
        run.log({"pca/pc1_var_explained": var[0], "pca/pc2_var_explained": var[1]})

    return None, {"X_2d": X_2d, "labels": labels, "var": var,
                  "label_ref": label_ref, "label_cand": label_cand}


# ------------------------------------------------------------------
# Combined 3-row plots (one row per feature mode)
# ------------------------------------------------------------------
def plot_combined_all(rank: int, mode_data: dict, tag_base: str, cfg: dict) -> None:
    """Produce 1×3-column combined figures for each measure (one column per feature
    mode), sized for US letter-paper inclusion (full text width ≈ 6.5 in).
    All six figures are logged in a single summary wandb run."""
    modes_ordered = list(FEATURE_MODES.keys())  # full, scalar_only, hist_only
    _COLORS = {"full": "#2980b9", "scalar_only": "#e67e22", "hist_only": "#27ae60"}

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

    with mpl.rc_context(_PAPER_RC):
        # ── per_feature_ks  (1 row × 3 cols) ────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.5))
        for ax, mode in zip(axes, modes_ordered):
            _draw_per_feature_ks(ax, mode_data[mode]["ks"]["df_ks"], mode, top_n=10)
        fig.suptitle(f"Per-feature KS — rank {rank+1}", fontsize=9)
        plt.tight_layout()
        _save_and_log(fig, "combined_per_feature_ks", summary_run, tag_base)

        # ── bootstrap_auc  (1 row × 3 cols) ─────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
        for ax, mode in zip(axes, modes_ordered):
            d = mode_data[mode]["boot"]
            _draw_bootstrap_auc(ax, d["boot_aucs"], d["ci_lo"], d["ci_hi"], mode)
        fig.suptitle(f"Bootstrap AUC — rank {rank+1}", fontsize=9)
        plt.tight_layout()
        _save_and_log(fig, "combined_bootstrap_auc", summary_run, tag_base)

        # ── recall  (2 rows × 3 cols: row 0 = dists, row 1 = curve) ──
        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
        for col_i, mode in enumerate(modes_ordered):
            d = mode_data[mode]["recall"]
            _draw_coverage_dists(axes[0, col_i], d["real_to_synth"], d["synth_internal"],
                                 d["threshold"], mode,
                                 label_ref=d["label_ref"], label_cand=d["label_cand"])
            _draw_coverage_curve(axes[1, col_i], d["real_to_synth"], d["threshold"],
                                 d["covered_fraction"], mode, label_ref=d["label_ref"])
        fig.suptitle(f"Recall — rank {rank+1}", fontsize=9)
        plt.tight_layout()
        _save_and_log(fig, "combined_recall", summary_run, tag_base)

        # ── precision  (2 rows × 3 cols) ─────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
        for col_i, mode in enumerate(modes_ordered):
            d = mode_data[mode]["precision"]
            _draw_coverage_dists(axes[0, col_i], d["real_to_synth"], d["synth_internal"],
                                 d["threshold"], mode,
                                 label_ref=d["label_ref"], label_cand=d["label_cand"])
            _draw_coverage_curve(axes[1, col_i], d["real_to_synth"], d["threshold"],
                                 d["covered_fraction"], mode, label_ref=d["label_ref"])
        fig.suptitle(f"Precision — rank {rank+1}", fontsize=9)
        plt.tight_layout()
        _save_and_log(fig, "combined_precision", summary_run, tag_base)

        # ── overlaid recall/precision curves  (1×2: recall | precision) ──
        fig, (ax_fwd, ax_rev) = plt.subplots(1, 2, figsize=(6.5, 2.4))
        for mode in modes_ordered:
            d = mode_data[mode]["recall"]
            sorted_d = np.sort(d["real_to_synth"])
            ax_fwd.plot(sorted_d, np.linspace(0, 1, len(sorted_d)),
                        color=_COLORS[mode], label=mode, linewidth=0.9)
            ax_fwd.axvline(d["threshold"], color=_COLORS[mode], linestyle="--",
                           linewidth=0.6, alpha=0.7)
        ax_fwd.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
        ax_fwd.set_ylabel("CDF (Recall)")
        ax_fwd.set_title("Recall")
        ax_fwd.legend()

        for mode in modes_ordered:
            d = mode_data[mode]["precision"]
            sorted_d = np.sort(d["real_to_synth"])
            ax_rev.plot(sorted_d, np.linspace(0, 1, len(sorted_d)),
                        color=_COLORS[mode], label=mode, linewidth=0.9)
            ax_rev.axvline(d["threshold"], color=_COLORS[mode], linestyle="--",
                           linewidth=0.6, alpha=0.7)
        ax_rev.set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
        ax_rev.set_ylabel("CDF (Precision)")
        ax_rev.set_title("Precision")
        ax_rev.legend()

        fig.suptitle(f"Recall/Precision curves — rank {rank+1}", fontsize=9)
        plt.tight_layout()
        _save_and_log(fig, "combined_recall_precision_curves", summary_run, tag_base)

        # ── feature_importance  (1 row × 3 cols) ─────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.5))
        for ax, mode in zip(axes, modes_ordered):
            d = mode_data[mode]["imp"]
            _draw_feature_importance(ax, d["imp"], d["auc_val"], mode, top_n=10)
        fig.suptitle(f"Feature importance — rank {rank+1}", fontsize=9)
        plt.tight_layout()
        _save_and_log(fig, "combined_feature_importance", summary_run, tag_base)

        # ── pca  (1 row × 3 cols) ────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
        for ax, mode in zip(axes, modes_ordered):
            d = mode_data[mode]["pca"]
            _draw_pca(ax, d["X_2d"], d["labels"], d["var"], mode)
        fig.suptitle(f"PCA — rank {rank+1}", fontsize=9)
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
    fig.savefig(path, dpi=300, bbox_inches="tight")
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
    cache_path = CACHE_DIR / f"real_feats_{SOURCE_NAME}_{N_REAL_CACHE}_{MAX_ROWS}_v2.npz"
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

            _, ks_data      = analyse_per_feature(real_m, synth_m, run, tag, fnames)
            _, imp_data     = analyse_feature_importance(real_m, synth_m, run, tag, fnames)
            _, boot_data    = analyse_bootstrap_auc(real_m, synth_m, run, tag)
            _, recall_data    = analyse_coverage(real_m, synth_m, run, tag,
                                                label_ref=_LABEL_REAL,
                                                label_cand=_LABEL_SYNTH)
            _, precision_data = analyse_coverage(synth_m, real_m, run, tag + "_precision",
                                                 label_ref=_LABEL_SYNTH,
                                                 label_cand=_LABEL_REAL,
                                                 log_prefix="precision")
            _, pca_data     = analyse_pca(real_m, synth_m, run, tag,
                                          label_ref=_LABEL_REAL,
                                          label_cand=_LABEL_SYNTH)

            mode_data[mode] = {
                "ks":        ks_data,
                "imp":       imp_data,
                "boot":      boot_data,
                "recall":    recall_data,
                "precision": precision_data,
                "pca":       pca_data,
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
