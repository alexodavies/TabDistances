"""Coverage analysis: tabICL prior vs real t4 data.

Generates synthetic feature vectors for the top-k grid-search configurations
and runs five analyses per config, each logged as a separate wandb run:

  1. Per-feature KS tests   — which of the 70 df_to_vec dimensions differ most
  2. Feature importance      — XGBoost discriminator gain per feature
  3. Bootstrap AUC CI       — exact CI by resampling + retraining (no Hanley-McNeil assumptions)
  4. Coverage (k-NN)        — fraction of real samples with no nearby synthetic neighbour
  5. PCA projection         — 2-D visualisation of the real / synthetic separation

Usage
-----
    python coverage_analysis.py [--config grid_results_t4_gran_3_old.csv]
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
RESULTS_CSV   = "grid_results_t4_gran_3_old.csv"
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
        batch_size_per_gp=int(params["batch_size_per_gp"]),
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
            "tree_model":              params["tree_model"],
            "mix_probas":              (float(params["mix_prob_mlp"]),
                                        1.0 - float(params["mix_prob_mlp"])),
            "cat_prob":                float(params["cat_prob"]),
            "multiclass_ordered_prob": float(params["multiclass_ordered_prob"]),
            "balanced":                bool(params["balanced"]),
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
# Analysis 1: per-feature KS tests
# ------------------------------------------------------------------
def analyse_per_feature(real: np.ndarray, synth: np.ndarray, run, tag: str = "") -> pd.DataFrame:
    print("\n── Per-feature KS tests ──")
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        ks_stat, p_val = stats.ks_2samp(real[:, i], synth[:, i])
        # rank-biserial effect size via Mann-Whitney
        u_stat, _ = stats.mannwhitneyu(real[:, i], synth[:, i], alternative="two-sided")
        r = 1 - (2 * u_stat) / (len(real) * len(synth))
        rows.append({"feature": name, "ks_stat": ks_stat, "p_value": p_val,
                     "rank_biserial_r": r, "abs_r": abs(r)})

    df_ks = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)

    # ── plot: top-20 by KS statistic ──
    top20 = df_ks.head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if r > 0 else "#2ecc71" for r in top20["rank_biserial_r"]]
    bars = ax.barh(top20["feature"][::-1], top20["ks_stat"][::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("KS statistic  (higher = larger distributional gap)")
    ax.set_title("Top-20 features by KS statistic: real vs synthetic (red = real > synth)")
    for bar, row in zip(bars, top20.iloc[::-1].itertuples()):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"r={row.rank_biserial_r:.2f}", va="center", fontsize=7)
    plt.tight_layout()
    _save_and_log(fig, "per_feature_ks", run, tag)

    # ── scalar summary to wandb ──
    if run is not None:
        run.log({"per_feature/mean_ks": df_ks["ks_stat"].mean(),
                 "per_feature/max_ks":  df_ks["ks_stat"].max(),
                 "per_feature/n_significant_bonf": (df_ks["p_value"] * len(df_ks) < 0.005).sum(),
                 "per_feature/table": wandb.Table(dataframe=df_ks.round(4))})

    print(df_ks.head(10).to_string(index=False))
    return df_ks


# ------------------------------------------------------------------
# Analysis 2: bootstrap AUC CI
# ------------------------------------------------------------------
def analyse_bootstrap_auc(real: np.ndarray, synth: np.ndarray, run, tag: str = "") -> dict:
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
    ax.hist(boot_aucs, bins=30, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(ci_lo,  color="red",   linestyle="--",
               label=f"{100*(1-BOOT_ALPHA):.1f}% CI: ({ci_lo:.4f}, {ci_hi:.4f})")
    ax.axvline(ci_hi,  color="red",   linestyle="--")
    ax.axvline(boot_aucs.mean(), color="orange", linestyle="-", label=f"mean={boot_aucs.mean():.4f}")
    ax.axvline(0.5,    color="black", linestyle=":",  label="AUC=0.5 (chance)")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Bootstrap count")
    ax.set_title(f"Bootstrap AUC distribution  (n_boot={N_BOOT})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_and_log(fig, "bootstrap_auc", run, tag)

    result = {"ci_lo": ci_lo, "ci_hi": ci_hi,
              "mean": boot_aucs.mean(), "std": boot_aucs.std(),
              "lower_bound_above_0.5": bool(ci_lo > 0.5)}
    if run is not None:
        run.log({f"bootstrap/{k}": v for k, v in result.items()})
    return result


# ------------------------------------------------------------------
# Analysis 3: coverage (k-NN)
# ------------------------------------------------------------------
def analyse_coverage(real: np.ndarray, synth: np.ndarray, run, tag: str = "") -> dict:
    print(f"\n── Coverage analysis  (k={KNN_K}) ──")

    # Normalise features to [0,1] per dimension so distances are comparable
    lo  = np.minimum(real.min(0), synth.min(0))
    hi  = np.maximum(real.max(0), synth.max(0))
    rng = hi - lo
    rng[rng == 0] = 1.0
    real_n  = (real  - lo) / rng
    synth_n = (synth - lo) / rng

    # Within-synthetic k-NN distances → baseline for "typical" synthetic spacing
    nn_ss = NearestNeighbors(n_neighbors=KNN_K + 1, algorithm="ball_tree").fit(synth_n)
    dists_ss, _ = nn_ss.kneighbors(synth_n)
    synth_internal = dists_ss[:, 1:]  # exclude self
    threshold = np.percentile(synth_internal.mean(axis=1), COVERAGE_THRESHOLD_QUANTILE * 100)

    # Real-to-synthetic k-NN distances
    nn_rs = NearestNeighbors(n_neighbors=KNN_K, algorithm="ball_tree").fit(synth_n)
    dists_rs, _ = nn_rs.kneighbors(real_n)
    real_to_synth = dists_rs.mean(axis=1)

    uncovered = (real_to_synth > threshold).mean()
    print(f"  Coverage threshold (p{COVERAGE_THRESHOLD_QUANTILE*100:.0f} within-synth): {threshold:.4f}")
    print(f"  Fraction of real samples uncovered: {uncovered:.3f}")
    print(f"  Median real→synth distance: {np.median(real_to_synth):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(synth_internal.mean(axis=1), bins=50, alpha=0.6,
                 color="steelblue", label="within-synthetic", edgecolor="white")
    axes[0].hist(real_to_synth, bins=50, alpha=0.6,
                 color="salmon", label="real → nearest synthetic", edgecolor="white")
    axes[0].axvline(threshold, color="black", linestyle="--",
                    label=f"threshold (p{COVERAGE_THRESHOLD_QUANTILE*100:.0f}): {threshold:.3f}")
    axes[0].set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("k-NN distance distributions")
    axes[0].legend(fontsize=8)

    # Sorted real→synth distances (coverage curve)
    axes[1].plot(np.sort(real_to_synth), np.linspace(0, 1, len(real_to_synth)), color="salmon")
    axes[1].axvline(threshold, color="black", linestyle="--", label=f"threshold: {threshold:.3f}")
    axes[1].axhline(1 - uncovered, color="steelblue", linestyle=":",
                    label=f"covered fraction: {1-uncovered:.3f}")
    axes[1].set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
    axes[1].set_ylabel("Cumulative fraction of real samples")
    axes[1].set_title("Coverage curve")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save_and_log(fig, "coverage", run, tag)

    result = {"threshold": threshold, "uncovered_fraction": uncovered,
              "median_real_to_synth_dist": float(np.median(real_to_synth))}
    if run is not None:
        run.log({f"coverage/{k}": v for k, v in result.items()})
    return result


# ------------------------------------------------------------------
# Analysis 4: feature importance
# ------------------------------------------------------------------
def analyse_feature_importance(real: np.ndarray, synth: np.ndarray, run, tag: str = "") -> pd.DataFrame:
    print("\n── Feature importance (discriminator gain) ──")
    X_all = np.vstack([synth, real])
    y_all = np.array([0] * len(synth) + [1] * len(real))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=TEST_FRAC, random_state=RANDOM_STATE, stratify=y_all)

    clf = XGBClassifier(n_estimators=300, device=XGB_DEVICE,
                        eval_metric="logloss", verbosity=0,
                        random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    print(f"  Discriminator AUC on held-out test set: {auc:.4f}")

    # Gain importance — proportional contribution to loss reduction
    scores = clf.get_booster().get_score(importance_type="gain")
    # XGBoost names features f0..f69; map back to FEATURE_NAMES
    imp = pd.DataFrame([
        {"feature": FEATURE_NAMES[int(k[1:])], "gain": v}
        for k, v in scores.items()
    ]).sort_values("gain", ascending=False)
    imp["gain_norm"] = imp["gain"] / imp["gain"].sum()

    # ── plot: top-30 ──
    top30 = imp.head(30)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top30["feature"][::-1], top30["gain_norm"][::-1],
                   color="steelblue", alpha=0.85)
    ax.set_xlabel("Normalised gain (fraction of total discriminator gain)")
    ax.set_title(f"Feature importance — discriminator gain  (AUC={auc:.4f})")
    for bar, val in zip(bars, top30["gain_norm"][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)
    plt.tight_layout()
    _save_and_log(fig, "feature_importance", run, tag)

    if run is not None:
        run.log({"feature_importance/discriminator_auc": auc,
                 "feature_importance/table": wandb.Table(dataframe=imp.round(5))})

    print(imp.head(10).to_string(index=False))
    return imp


# ------------------------------------------------------------------
# Analysis 5: PCA projection
# ------------------------------------------------------------------
def analyse_pca(real: np.ndarray, synth: np.ndarray, run, tag: str = "") -> None:
    print("\n── PCA projection ──")
    X_all  = np.vstack([real, synth])
    labels = np.array(["real"] * len(real) + ["synthetic"] * len(synth))

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_all)
    var  = pca.explained_variance_ratio_

    print(f"  PC1 variance explained: {var[0]:.3f}   PC2: {var[1]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color, alpha in [("real", "#e74c3c", 0.5), ("synthetic", "#2980b9", 0.35)]:
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=label, alpha=alpha, s=8, linewidths=0)
    ax.set_xlabel(f"PC1  ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2  ({var[1]:.1%} variance)")
    ax.set_title("PCA: real vs synthetic feature vectors")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()
    _save_and_log(fig, "pca", run, tag)

    if run is not None:
        run.log({"pca/pc1_var_explained": var[0], "pca/pc2_var_explained": var[1]})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _save_and_log(fig: plt.Figure, name: str, run, tag: str = "") -> None:
    suffix = f"_{tag}" if tag else ""
    path = Path(f"coverage_{name}{suffix}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Plot saved -> {path}")
    if run is not None:
        run.log({name: wandb.Image(str(path))})
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(results_csv: str) -> None:
    # ── Load top-k configs ────────────────────────────────────────
    df_results = pd.read_csv(results_csv)
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

    # ── Loop over configs ─────────────────────────────────────────
    for rank, row in top_configs.iterrows():
        cfg = row.to_dict()
        cfg.update(fixed)
        tag = f"rank{rank + 1}"
        print(f"\n{'#'*60}")
        print(f"  Config rank {rank + 1}/{TOP_K}  (AUC={cfg['auc']:.4f})  [{tag}]")
        print(f"{'#'*60}")

        print(f"\nGenerating {N_SYNTH} synthetic tables ...")
        synth_feats = generate_synthetic_features(cfg, N_SYNTH)
        print(f"Synthetic features: {synth_feats.shape}")

        run = None
        if _WANDB_AVAILABLE and WANDB_PROJECT:
            run = wandb.init(
                project=WANDB_PROJECT,
                name=f"coverage_{SOURCE_NAME}_{tag}",
                config={
                    "source":         SOURCE_NAME,
                    "rank":           rank + 1,
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

        analyse_per_feature(real_feats, synth_feats, run, tag)
        analyse_feature_importance(real_feats, synth_feats, run, tag)
        analyse_bootstrap_auc(real_feats, synth_feats, run, tag)
        analyse_coverage(real_feats, synth_feats, run, tag)
        analyse_pca(real_feats, synth_feats, run, tag)

        if run is not None:
            run.finish()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=RESULTS_CSV,
                        help="Path to grid results CSV (default: %(default)s)")
    args = parser.parse_args()
    main(args.config)
