"""Coverage analysis for the highest-recall Pareto-front configurations.

For each source (tabfm, t4), loads the highest-recall point from the global
Pareto front, generates synthetic tables using those prior parameters, and
runs the same coverage analysis suite as analysis/coverage_analysis.py.

Usage
-----
    python pareto-best-analysis.py [--out-dir OUT_DIR]
"""

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
from tabdistances.discriminator.dataset import load_dataframes
from tabdistances.discriminator.features import df_to_vec, select_features, FEATURE_NAMES

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
REAL_SOURCES = {
    "tabfm": Path("tables-tabfm/parquet"),
    "t4":    Path("tables-t4/parquet"),
}

PARETO_DIR    = Path("pareto-results")
CACHE_DIR     = Path("feature-cache")
OUT_DIR       = Path("pareto-results/coverage-analysis")

N_REAL_CACHE  = 10000
N_SYNTH       = 1000
N_REAL_SAMPLE = 1000
MAX_ROWS      = 512
BATCH_SIZE    = 1
RANDOM_STATE  = 42

KNN_K                      = 5
COVERAGE_THRESHOLD_QUANTILE = 0.95

N_BOOT     = 200
BOOT_ALPHA = 0.005
TEST_FRAC  = 0.3

FEATURE_MODE  = "full"
WANDB_PROJECT = "tabicl-pareto-best-analysis"

_LABEL_SYNTH = "Synthetic"

SOURCE_DISPLAY = {"tabfm": "TabFM", "t4": "T4"}

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

# ------------------------------------------------------------------
# Real-data feature cache  (mirrors pareto-coverage.py)
# ------------------------------------------------------------------

def load_or_compute_real_feats(source_name: str, path: Path) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"real_feats_{source_name}_{N_REAL_CACHE}_{MAX_ROWS}_v2.npz"
    if cache_path.exists():
        print(f"  Loading cached features: {cache_path}")
        return np.load(cache_path)["feats"]
    print(f"  Loading up to {N_REAL_CACHE} real tables from {path} ...")
    dfs = [df for df, _ in load_dataframes(
        path, label=0, max_rows=MAX_ROWS, max_tables=N_REAL_CACHE, random_state=RANDOM_STATE,
    )]
    print(f"  Featurizing {len(dfs)} tables ...")
    feats = np.array([df_to_vec(df) for df in dfs])
    np.savez(cache_path, feats=feats)
    print(f"  Cached -> {cache_path}  shape={feats.shape}")
    return feats


# ------------------------------------------------------------------
# Synthetic generation  (mirrors pareto-coverage.py _make_prior)
# ------------------------------------------------------------------

def _make_prior(params: dict, n_cols: int) -> SCMPrior:
    return SCMPrior(
        batch_size=BATCH_SIZE,
        batch_size_per_gp=int(params["batch_size_per_gp"]),
        min_features=n_cols,
        max_features=n_cols,
        max_classes=int(params["max_classes"]),
        max_seq_len=MAX_ROWS,
        log_seq_len=True,
        min_train_size=0.1,
        max_train_size=0.9,
        prior_type=str(params["prior_type"]),
        replay_small=bool(params["replay_small"]),
        fixed_hp={
            "tree_model":              str(params["tree_model"]),
            "mix_probas":              (float(params["mix_prob_mlp"]),
                                        1.0 - float(params["mix_prob_mlp"])),
            "cat_prob":                float(params["cat_prob"]),
            "multiclass_ordered_prob": float(params["multiclass_ordered_prob"]),
            "balanced":                bool(params["balanced"]),
        },
        n_jobs=1,
        device="cpu",
    )


def generate_synthetic_features(params: dict, target_col_counts: list) -> np.ndarray:
    from collections import Counter
    count_map = Counter(target_col_counts)
    results_by_count: dict = {}
    for n_cols, n_needed in count_map.items():
        prior = _make_prior(params, n_cols)
        dfs: list = []
        with tqdm(total=n_needed, desc=f"  generating (n_cols={n_cols})", unit="table") as pbar:
            while len(dfs) < n_needed:
                X, y, d, seq_lens, _ = prior.get_batch()
                for i in range(X.shape[0]):
                    seq_len = int(seq_lens[i].item())
                    n_feat  = int(d[i].item())
                    if seq_len == 0:
                        continue
                    X_i = X[i, :seq_len, :n_feat].numpy().astype(float)
                    y_i = y[i, :seq_len].numpy().astype(int)
                    df  = pd.DataFrame(X_i, columns=[f"f{j}" for j in range(n_feat)])
                    df["target"] = y_i
                    dfs.append(df)
                    pbar.update(1)
                    if len(dfs) >= n_needed:
                        break
        results_by_count[n_cols] = dfs

    idx_by_count = {nc: 0 for nc in count_map}
    output_dfs = []
    for nc in target_col_counts:
        output_dfs.append(results_by_count[nc][idx_by_count[nc]])
        idx_by_count[nc] += 1
    return np.array([df_to_vec(df) for df in output_dfs])


# ------------------------------------------------------------------
# Analysis helpers
# ------------------------------------------------------------------

def _pareto_mask_2d(recall: np.ndarray, precision: np.ndarray) -> np.ndarray:
    n = len(recall)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        dominated[i] = np.any(
            (recall    >= recall[i])    &
            (precision >= precision[i]) &
            ((recall > recall[i]) | (precision > precision[i]))
        )
    return ~dominated


def _save(fig: plt.Figure, name: str, out_dir: Path, tag: str = "", run=None) -> None:
    suffix = f"_{tag}" if tag else ""
    path = out_dir / f"{name}{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    if run is not None:
        run.log({name: wandb.Image(str(path))})


# ------------------------------------------------------------------
# Analyses
# ------------------------------------------------------------------

def analyse_per_feature(real: np.ndarray, synth: np.ndarray,
                        tag: str, out_dir: Path, run=None,
                        feature_names: list[str] = FEATURE_NAMES) -> pd.DataFrame:
    print("  ── Per-feature KS tests ──")
    rows = []
    for i, name in enumerate(feature_names):
        ks_stat, p_val = stats.ks_2samp(real[:, i], synth[:, i])
        u_stat, _ = stats.mannwhitneyu(real[:, i], synth[:, i], alternative="two-sided")
        r = 1 - (2 * u_stat) / (len(real) * len(synth))
        rows.append({"feature": name, "ks_stat": ks_stat, "p_value": p_val,
                     "rank_biserial_r": r})
    df_ks = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)

    top = df_ks.head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if r > 0 else "#2ecc71" for r in top["rank_biserial_r"]]
    ax.barh(top["feature"][::-1], top["ks_stat"][::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("KS statistic  (higher = larger distributional gap)")
    ax.set_title(f"Top-20 KS: real vs synthetic  [{tag}]")
    plt.tight_layout()
    _save(fig, "per_feature_ks", out_dir, tag, run)
    if run is not None:
        run.log({"per_feature/mean_ks": df_ks["ks_stat"].mean(),
                 "per_feature/max_ks":  df_ks["ks_stat"].max(),
                 "per_feature/table":   wandb.Table(dataframe=df_ks.round(4))})
    return df_ks


def analyse_bootstrap_auc(real: np.ndarray, synth: np.ndarray,
                           tag: str, out_dir: Path, run=None) -> dict:
    print(f"  ── Bootstrap AUC CI  (n_boot={N_BOOT}) ──")
    rng = np.random.default_rng(RANDOM_STATE)
    boot_aucs = []
    for _ in tqdm(range(N_BOOT), desc="  bootstrap", leave=False):
        r_idx = rng.integers(0, len(real),  len(real))
        s_idx = rng.integers(0, len(synth), len(synth))
        X = np.vstack([synth[s_idx], real[r_idx]])
        y = np.array([0] * len(synth) + [1] * len(real))
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_FRAC, random_state=int(rng.integers(1e6)), stratify=y)
        clf = XGBClassifier(n_estimators=300, eval_metric="logloss",
                            verbosity=0, random_state=int(rng.integers(1e6)))
        clf.fit(X_tr, y_tr)
        boot_aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))

    boot_aucs = np.array(boot_aucs)
    pct_lo, pct_hi = 100 * BOOT_ALPHA / 2, 100 * (1 - BOOT_ALPHA / 2)
    ci_lo, ci_hi = np.percentile(boot_aucs, [pct_lo, pct_hi])
    print(f"  {100*(1-BOOT_ALPHA):.1f}% CI: ({ci_lo:.4f}, {ci_hi:.4f})  mean={boot_aucs.mean():.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(boot_aucs, bins=30, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(ci_lo, color="red", linestyle="--",
               label=f"{100*(1-BOOT_ALPHA):.1f}% CI: ({ci_lo:.4f}, {ci_hi:.4f})")
    ax.axvline(ci_hi, color="red", linestyle="--")
    ax.axvline(boot_aucs.mean(), color="orange", label=f"mean={boot_aucs.mean():.4f}")
    ax.axvline(0.5, color="black", linestyle=":", label="AUC=0.5 (chance)")
    ax.set_xlabel("AUC"); ax.set_ylabel("Bootstrap count")
    ax.set_title(f"Bootstrap AUC  [{tag}]"); ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "bootstrap_auc", out_dir, tag, run)
    result = {"ci_lo": ci_lo, "ci_hi": ci_hi, "mean": boot_aucs.mean()}
    if run is not None:
        run.log({f"bootstrap/{k}": v for k, v in result.items()})
    return result


def analyse_coverage(ref: np.ndarray, cand: np.ndarray,
                     label_ref: str, label_cand: str,
                     tag: str, out_dir: Path, run=None, log_prefix: str = "coverage") -> dict:
    print(f"  ── Coverage ({label_ref} ∈ {label_cand})  k={KNN_K} ──")
    lo  = np.minimum(ref.min(0), cand.min(0))
    hi  = np.maximum(ref.max(0), cand.max(0))
    rng = hi - lo;  rng[rng == 0] = 1.0
    ref_n  = (ref  - lo) / rng
    cand_n = (cand - lo) / rng

    nn_ss = NearestNeighbors(n_neighbors=KNN_K + 1, algorithm="ball_tree").fit(cand_n)
    d_ss, _ = nn_ss.kneighbors(cand_n)
    threshold = np.percentile(d_ss[:, 1:].mean(axis=1), COVERAGE_THRESHOLD_QUANTILE * 100)

    nn_rs = NearestNeighbors(n_neighbors=KNN_K, algorithm="ball_tree").fit(cand_n)
    d_rs, _ = nn_rs.kneighbors(ref_n)
    ref_to_cand = d_rs.mean(axis=1)

    covered = 1.0 - (ref_to_cand > threshold).mean()
    print(f"  Covered fraction: {covered:.3f}  threshold: {threshold:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(d_ss[:, 1:].mean(axis=1), bins=50, alpha=0.6, color="steelblue",
                 label=f"within-{label_cand}", edgecolor="white")
    axes[0].hist(ref_to_cand, bins=50, alpha=0.6, color="salmon",
                 label=f"{label_ref} ∈ nearest {label_cand}", edgecolor="white")
    axes[0].axvline(threshold, color="black", linestyle="--",
                    label=f"threshold (p{COVERAGE_THRESHOLD_QUANTILE*100:.0f}): {threshold:.3f}")
    axes[0].set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
    axes[0].set_ylabel("Count"); axes[0].set_title("k-NN distance distributions")
    axes[0].legend(fontsize=8)

    axes[1].plot(np.sort(ref_to_cand), np.linspace(0, 1, len(ref_to_cand)), color="salmon")
    axes[1].axvline(threshold, color="black", linestyle="--", label=f"threshold: {threshold:.3f}")
    axes[1].axhline(covered, color="steelblue", linestyle=":",
                    label=f"covered: {covered:.3f}")
    axes[1].set_xlabel(f"Mean {KNN_K}-NN distance (normalised)")
    axes[1].set_ylabel(f"Cumulative fraction of {label_ref}")
    axes[1].set_title("Coverage curve"); axes[1].legend(fontsize=8)

    fig.suptitle(f"{label_ref} ∈ {label_cand}  [{tag}]", fontsize=9)
    plt.tight_layout()
    _save(fig, f"coverage_{label_ref.lower()}_in_{label_cand.lower()}", out_dir, tag, run)
    result = {"covered_fraction": covered, "threshold": threshold}
    if run is not None:
        run.log({f"{log_prefix}/{k}": v for k, v in result.items()})
    return result


def analyse_pca(real: np.ndarray, synth: np.ndarray, label_real: str,
                tag: str, out_dir: Path, run=None) -> None:
    print("  ── PCA projection ──")
    X_all  = np.vstack([real, synth])
    labels = np.array([label_real] * len(real) + [_LABEL_SYNTH] * len(synth))
    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d   = pca.fit_transform(X_all)
    var    = pca.explained_variance_ratio_
    print(f"  PC1={var[0]:.3f}  PC2={var[1]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for lbl, color, alpha in [(label_real, "#e74c3c", 0.5), (_LABEL_SYNTH, "#2980b9", 0.35)]:
        mask = labels == lbl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=lbl,
                   alpha=alpha, s=8, linewidths=0)
    ax.set_xlabel(f"PC1  ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2  ({var[1]:.1%} variance)")
    ax.set_title(f"PCA: {label_real} vs {_LABEL_SYNTH}  [{tag}]")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()
    _save(fig, "pca", out_dir, tag, run)
    if run is not None:
        run.log({"pca/pc1_var_explained": var[0], "pca/pc2_var_explained": var[1]})


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR), type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for source_name, real_path in REAL_SOURCES.items():
        label = SOURCE_DISPLAY[source_name]
        print(f"\n{'#'*60}")
        print(f"  {label}")
        print(f"{'#'*60}")

        csv_path = PARETO_DIR / f"pareto_results_{source_name}.csv"
        if not csv_path.exists():
            print(f"  Pareto CSV not found: {csv_path} — skipping.")
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={"real_by_synth": "recall", "synth_by_real": "precision"}, inplace=True)

        pareto_2d = _pareto_mask_2d(df["recall"].values, df["precision"].values)
        front = df[pareto_2d]
        best  = front.loc[front["recall"].idxmax()]
        print(f"  Highest-recall Pareto point: recall={best['recall']:.4f}  "
              f"precision={best['precision']:.4f}")
        print(f"  Parameters: {best[['prior_type','tree_model','max_classes','cat_prob']].to_dict()}")

        # Real features
        if not real_path.exists():
            print(f"  Real data path not found: {real_path} — skipping.")
            continue
        all_real = load_or_compute_real_feats(source_name, real_path)
        rng      = np.random.default_rng(RANDOM_STATE)
        idx      = rng.choice(len(all_real), size=min(N_REAL_SAMPLE, len(all_real)), replace=False)
        real_feats = all_real[idx]

        # Synthetic features
        col_counts = all_real[:, 0].astype(int)
        target_cols = rng.choice(col_counts, size=N_SYNTH, replace=True).tolist()
        print(f"  Generating {N_SYNTH} synthetic tables ...")
        synth_feats_full = generate_synthetic_features(best.to_dict(), target_cols)

        real_m,  fnames = select_features(real_feats,       FEATURE_MODE)
        synth_m, _      = select_features(synth_feats_full, FEATURE_MODE)
        print(f"  Feature matrix: real={real_m.shape}  synth={synth_m.shape}")

        tag = f"pareto_best_{source_name}"

        run = None
        if _WANDB_AVAILABLE:
            run = wandb.init(
                project=WANDB_PROJECT,
                name=tag,
                config={
                    "source":       source_name,
                    "recall":       float(best["recall"]),
                    "precision":    float(best["precision"]),
                    "feature_mode": FEATURE_MODE,
                    "n_synth":      N_SYNTH,
                    "n_real":       N_REAL_SAMPLE,
                    **{k: best[k] for k in ["prior_type", "tree_model", "max_classes",
                                             "cat_prob", "mix_prob_mlp",
                                             "multiclass_ordered_prob", "balanced",
                                             "replay_small"]},
                },
                reinit=True,
            )

        analyse_per_feature(real_m, synth_m, tag, args.out_dir, run, fnames)
        analyse_bootstrap_auc(real_m, synth_m, tag, args.out_dir, run)
        analyse_coverage(real_m,  synth_m, label,        _LABEL_SYNTH, tag,
                         args.out_dir, run, log_prefix="recall")
        analyse_coverage(synth_m, real_m,  _LABEL_SYNTH, label,        tag + "_precision",
                         args.out_dir, run, log_prefix="precision")
        analyse_pca(real_m, synth_m, label, tag, args.out_dir, run)

        if run is not None:
            run.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
