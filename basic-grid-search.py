"""Grid search over TabICL prior parameters.

Uses aggregate table metrics (70-D vectors from df_to_vec) fed into an
XGBoost discriminator with automatic GPU detection.  Real-data features
are pre-computed once and cached to disk, so only synthetic data needs
featurizing on each evaluation.

Usage
-----
    python basic-grid-search.py

Configuration
-------------
All tunable constants are in the CONFIG section below.
"""

import sys
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tqdm import tqdm

from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.discriminator.dataset import load_dataframes
from tabdistances.discriminator.features import df_to_vec

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
REAL_SOURCES = {
    "t4":    Path("tables-t4/parquet"),
    "tabfm": Path("tables-tabfm/parquet"),
}

N_EVAL       = 50     # synthetic (and real) tables per discriminator evaluation
N_REAL       = 5000   # real tables to pre-featurize and cache
MAX_ROWS     = 512    # max rows sampled per table
BATCH_SIZE   = 8      # SCMPrior batch size
TEST_FRAC    = 0.3
RANDOM_STATE = 42
GRANULARITY  = 3      # grid points for int/float parameters (cat params use all values)
N_WORKERS    = 12     # parallel grid evaluations (set to 1 to disable)
CACHE_DIR    = Path("cache")
WANDB_PROJECT = "tabicl-grid-search"  # set to None to disable wandb

# SCMPrior internal parallelism — scale down so workers don't over-subscribe CPUs
_SCMPRIOR_JOBS = max(1, 8 // N_WORKERS)

PARAM_SPACE = {
    "max_classes":             ("int",   2,    20),
    "batch_size_per_gp":       ("int",   1,    BATCH_SIZE),
    "prior_type":              ("cat",   ["mlp_scm", "tree_scm", "mix_scm"]),
    "tree_model":              ("cat",   ["decision_tree", "extra_trees",
                                          "random_forest", "xgboost"]),
    "mix_prob_mlp":            ("float", 0.0,  1.0),
    "cat_prob":                ("float", 0.0,  0.6),
    "multiclass_ordered_prob": ("float", 0.0,  1.0),
    "balanced":                ("cat",   [False, True]),
    "replay_small":            ("cat",   [False, True]),
}


# ------------------------------------------------------------------
# GPU detection
# ------------------------------------------------------------------

def _xgb_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

XGB_DEVICE = _xgb_device()


# ------------------------------------------------------------------
# Grid construction
# ------------------------------------------------------------------

def build_grid(granularity: int) -> list[dict]:
    axes = {}
    for k, spec in PARAM_SPACE.items():
        if spec[0] == "cat":
            axes[k] = list(spec[1])
        elif spec[0] == "int":
            vals = np.linspace(spec[1], spec[2], granularity).round().astype(int)
            axes[k] = list(dict.fromkeys(vals.tolist()))  # dedup, preserve order
        else:  # float
            axes[k] = np.linspace(spec[1], spec[2], granularity).tolist()

    keys = list(axes.keys())
    combos = list(itertools.product(*[axes[k] for k in keys]))
    return [dict(zip(keys, c)) for c in combos]


# ------------------------------------------------------------------
# Real-data feature caching
# ------------------------------------------------------------------

def load_or_compute_real_feats(source_name: str, path: Path) -> np.ndarray:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"real_feats_{source_name}_{N_REAL}_{MAX_ROWS}.npz"

    if cache_path.exists():
        print(f"  Loading cached features: {cache_path}")
        return np.load(cache_path)["feats"]

    print(f"  Loading up to {N_REAL} real tables from {path} ...")
    dfs = [df for df, _ in load_dataframes(
        path, label=0, max_rows=MAX_ROWS, max_tables=N_REAL, random_state=RANDOM_STATE,
    )]
    print(f"  Featurizing {len(dfs)} tables ...")
    feats = np.array([df_to_vec(df) for df in tqdm(dfs, desc="  featurizing", unit="table")])
    np.savez(cache_path, feats=feats)
    print(f"  Cached -> {cache_path}  shape={feats.shape}")
    return feats


# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------

def generate_synthetic_dataframes(params: dict, n: int) -> list:
    prior = SCMPrior(
        batch_size=BATCH_SIZE,
        batch_size_per_gp=params["batch_size_per_gp"],
        min_features=params["min_features"],
        max_features=params["max_features"],
        max_classes=params["max_classes"],
        max_seq_len=MAX_ROWS,
        log_seq_len=True,
        min_train_size=0.1,
        max_train_size=0.9,
        prior_type=params["prior_type"],
        replay_small=params["replay_small"],
        fixed_hp={
            "tree_model":              params["tree_model"],
            "mix_probas":              (params["mix_prob_mlp"],
                                        1.0 - params["mix_prob_mlp"]),
            "cat_prob":                params["cat_prob"],
            "multiclass_ordered_prob": params["multiclass_ordered_prob"],
            "balanced":                params["balanced"],
        },
        n_jobs=_SCMPRIOR_JOBS,
        device="cpu",
    )

    dataframes = []
    with tqdm(total=n, desc="  generating tables", unit="table", leave=False) as pbar:
        while len(dataframes) < n:
            X, y, d, seq_lens, _ = prior.get_batch()
            for i in range(X.shape[0]):
                n_feat  = int(d[i].item())
                seq_len = int(seq_lens[i].item())
                if n_feat < params["min_features"] or seq_len == 0:
                    continue
                X_i = X[i, :seq_len, :n_feat].numpy().astype(float)
                y_i = y[i, :seq_len].numpy().astype(int)
                df  = pd.DataFrame(X_i, columns=[f"f{j}" for j in range(n_feat)])
                df["target"] = y_i
                dataframes.append(df)
                pbar.update(1)
                if len(dataframes) >= n:
                    break

    return dataframes


# ------------------------------------------------------------------
# Fitness
# ------------------------------------------------------------------

def evaluate(params: dict, real_feats: np.ndarray, seed: int) -> float:
    """ROC-AUC for tabicl vs one real source.  Lower AUC = more realistic."""
    try:
        synth_dfs = generate_synthetic_dataframes(params, N_EVAL)
    except Exception as exc:
        print(f"    [generate error] {exc}")
        return 1.0

    synth_feats = np.array([df_to_vec(df) for df in synth_dfs])

    # Subsample N_EVAL real features to match synthetic count (no re-featurization)
    n_real_sample = min(N_EVAL, len(real_feats))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(real_feats), size=n_real_sample, replace=False)
    real_sample = real_feats[idxs]

    X_all = np.vstack([synth_feats, real_sample])
    y_all = np.array([0] * len(synth_feats) + [1] * n_real_sample)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=TEST_FRAC,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    clf = XGBClassifier(
        n_estimators=300,
        device=XGB_DEVICE,
        eval_metric="logloss",
        verbosity=0,
        random_state=RANDOM_STATE,
    )
    print(f"    Fitting discriminator ({len(X_tr)} train / {len(X_te)} test) ...", flush=True)
    clf.fit(X_tr, y_tr)
    print(f"    Discriminator fit done.", flush=True)
    proba = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


# ------------------------------------------------------------------
# Wandb summary plots
# ------------------------------------------------------------------

_CAT_PARAMS   = [k for k, s in PARAM_SPACE.items() if s[0] == "cat"]
_FLOAT_PARAMS = [k for k, s in PARAM_SPACE.items() if s[0] in ("float", "int")]


def _fig_auc_histogram(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["auc"], bins=30, edgecolor="white", color="#4C72B0")
    ax.axvline(df["auc"].min(), color="red",    linestyle="--", label=f"best={df['auc'].min():.4f}")
    ax.axvline(df["auc"].mean(), color="orange", linestyle="--", label=f"mean={df['auc'].mean():.4f}")
    ax.set_xlabel("AUC  (lower = more realistic)")
    ax.set_ylabel("Count")
    ax.set_title("AUC distribution across grid")
    ax.legend()
    fig.tight_layout()
    return fig


def _fig_cat_param_means(df: pd.DataFrame) -> plt.Figure:
    n = len(_CAT_PARAMS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, param in zip(axes, _CAT_PARAMS):
        means = df.groupby(param)["auc"].mean().sort_values()
        bars  = ax.bar([str(v) for v in means.index], means.values, color="#DD8452")
        ax.set_title(param)
        ax.set_xlabel("value")
        if ax is axes[0]:
            ax.set_ylabel("mean AUC")
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Mean AUC by categorical parameter value", y=1.02)
    fig.tight_layout()
    return fig



def _fig_float_param_scatter(df: pd.DataFrame) -> plt.Figure:
    n = len(_FLOAT_PARAMS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, param in zip(axes, _FLOAT_PARAMS):
        ax.scatter(df[param], df["auc"], alpha=0.3, s=10, color="#55A868")
        # trend line
        z = np.polyfit(df[param].astype(float), df["auc"], 1)
        xs = np.linspace(df[param].min(), df[param].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color="red", linewidth=1.5)
        ax.set_xlabel(param)
        if ax is axes[0]:
            ax.set_ylabel("AUC")
        ax.set_title(param)
    fig.suptitle("AUC vs continuous/integer parameters", y=1.02)
    fig.tight_layout()
    return fig


def _fig_parallel_coords(df: pd.DataFrame, top_n: int = 100) -> plt.Figure:
    """Parallel coordinates plot for the top_n trials (lowest AUC)."""
    top = df.nsmallest(top_n, "auc").copy()

    # Encode categoricals to numeric ranks for the plot
    plot_df = pd.DataFrame()
    for col in list(PARAM_SPACE.keys()) + ["auc"]:
        if col not in top.columns:
            continue
        vals = top[col]
        if vals.dtype == object or vals.dtype == bool:
            uniq = sorted(vals.unique(), key=str)
            plot_df[col] = vals.map({v: i for i, v in enumerate(uniq)})
        else:
            plot_df[col] = vals.astype(float)

    # Normalise each axis to [0, 1]
    normed = plot_df.copy()
    for col in normed.columns:
        rng = normed[col].max() - normed[col].min()
        if rng > 0:
            normed[col] = (normed[col] - normed[col].min()) / rng

    # Colour by AUC (lower = darker blue)
    cmap   = plt.cm.viridis_r
    colors = [cmap(v) for v in normed["auc"]]

    fig, ax = plt.subplots(figsize=(max(12, len(normed.columns) * 1.4), 5))
    cols = normed.columns.tolist()
    for row, color in zip(normed.itertuples(index=False), colors):
        ax.plot(range(len(cols)), list(row), color=color, alpha=0.4, linewidth=0.8)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Normalised value")
    ax.set_title(f"Parallel coordinates – top {top_n} trials (colour = AUC, darker = better)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(top["auc"].min(), top["auc"].max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="AUC", pad=0.01)
    fig.tight_layout()
    return fig


def log_summary_plots(df_results: pd.DataFrame, source_name: str, run) -> None:
    """Create summary figures and log them to wandb + disk."""
    plots = {
        "auc_histogram":      _fig_auc_histogram(df_results),
        "cat_param_means":    _fig_cat_param_means(df_results),
        "float_param_scatter": _fig_float_param_scatter(df_results),
        "parallel_coords":    _fig_parallel_coords(df_results),
    }
    for name, fig in plots.items():
        path = Path(f"grid_plot_{source_name}_{name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  Plot saved -> {path}")
        if run is not None:
            run.log({f"{source_name}/{name}": wandb.Image(str(path))})
        plt.close(fig)


# ------------------------------------------------------------------
# Grid search (single source)
# ------------------------------------------------------------------

def run_grid_search(source_name: str, real_feats: np.ndarray) -> pd.DataFrame:
    print(f"\n{'#'*60}")
    print(f"  Grid search: tabicl  vs  {source_name}")
    print(f"{'#'*60}")

    # Derive feature range from cached features (feature index 0 = n_cols)
    col_counts   = real_feats[:, 0].astype(int)
    fixed_params = {
        "min_features": int(col_counts.min()),
        "max_features": int(col_counts.max()),
    }
    print(f"  Derived: min_features={fixed_params['min_features']}  "
          f"max_features={fixed_params['max_features']}")

    grid    = build_grid(GRANULARITY)
    n_total = len(grid)
    print(f"  Grid size: {n_total} combinations  (GRANULARITY={GRANULARITY})")
    print(f"  XGB device: {XGB_DEVICE}\n")

    run = None
    if _WANDB_AVAILABLE and WANDB_PROJECT:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"grid_{source_name}",
            config={
                "source":       source_name,
                "n_eval":       N_EVAL,
                "n_real":       len(real_feats),
                "max_rows":     MAX_ROWS,
                "granularity":  GRANULARITY,
                "grid_size":    n_total,
                "xgb_device":   XGB_DEVICE,
                **fixed_params,
            },
            reinit=True,
        )

    # Submit all trials; each worker gets its own seed for reproducible but
    # independent real-feature subsampling.
    records  = [None] * n_total
    best_auc = float("inf")
    done     = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(evaluate, {**params, **fixed_params}, real_feats, RANDOM_STATE + i): (i, params)
            for i, params in enumerate(grid, 1)
        }
        for fut in as_completed(futures):
            i, params = futures[fut]
            auc = fut.result()
            records[i - 1] = {**params, "auc": auc}
            done += 1
            if auc < best_auc:
                best_auc = auc
            print(f"  [{done:5d}/{n_total}] (trial {i:5d}) AUC={auc:.4f}  best={best_auc:.4f}  {params}")
            if run is not None:
                run.log({"trial": i, "auc": auc, "best_auc": best_auc, **params})

    df_results = pd.DataFrame(records)
    out_path   = Path(f"grid_results_{source_name}.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")

    # Summary table + plots
    if run is not None:
        table = wandb.Table(dataframe=df_results)
        run.log({f"{source_name}/results_table": table})

    print(f"  Generating summary plots ...")
    log_summary_plots(df_results, source_name, run)

    if run is not None:
        run.finish()

    return df_results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    missing = [n for n, p in REAL_SOURCES.items() if not any(p.glob("*.parquet"))]
    if missing:
        print("Missing real-data tables for:", ", ".join(missing))
        print("Run cleantab-export.py and tabfm-export.py first.")
        sys.exit(1)

    print(f"Pre-computing / loading real-data features  (N_REAL={N_REAL}, MAX_ROWS={MAX_ROWS}) ...")
    real_feats_by_source = {}
    for name, path in REAL_SOURCES.items():
        real_feats_by_source[name] = load_or_compute_real_feats(name, path)
        print(f"  {name}: {real_feats_by_source[name].shape[0]} tables cached")

    all_results = {}
    for source_name, real_feats in real_feats_by_source.items():
        df_res = run_grid_search(source_name, real_feats)
        all_results[source_name] = df_res

    print(f"\n{'#'*60}")
    print("  TOP-10 RESULTS PER SOURCE  (lower AUC = more realistic)")
    print(f"{'#'*60}")
    for source_name, df_res in all_results.items():
        top10 = df_res.nsmallest(10, "auc")
        print(f"\n  [{source_name}]")
        print(top10.to_string(index=False))
