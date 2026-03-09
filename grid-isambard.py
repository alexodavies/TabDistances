"""Grid search over TabICL prior parameters — Isambard HPC edition.

Uses aggregate table metrics (70-D vectors from df_to_vec) fed into an
XGBoost discriminator (CPU-only).  Real-data features are pre-computed
once and cached to disk, so only synthetic data needs featurizing on
each evaluation.

Usage
-----
    python grid-isambard.py          # auto-detects CPUs from SLURM_CPUS_PER_TASK
    sbatch grid-isambard.sh          # submit via Slurm

Configuration
-------------
All tunable constants are in the CONFIG section below.
Environment variables override path defaults at runtime:
  TAB_DATA_ROOT  – root containing tables-t4/ and tables-tabfm/ (default ".")
  TAB_OUT_DIR    – where CSVs, plots and cache land (default grid-results-<JOB_ID>)

Multi-node array jobs
---------------------
Add  #SBATCH --array=0-N  to grid-isambard.sh to shard the grid across N+1
nodes.  The sharding logic inside run_grid_search() reads SLURM_ARRAY_TASK_ID
and SLURM_ARRAY_TASK_COUNT automatically — no other changes needed.
"""

import sys
import os
import warnings

# Suppress noisy sklearn warning produced by TabICL's internal joblib usage
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
    module="sklearn",
)

# ------------------------------------------------------------------
# Thread-explosion guard — must come before numpy / torch imports so
# that OpenMP, OpenBLAS and MKL see these values at initialisation.
# Each ProcessPoolExecutor worker + each loky subprocess that SCMPrior
# spawns would otherwise try to create its own thread pool, leading to
# hundreds of threads competing over the available cores.
# ------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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

# --- Slurm / CPU auto-detection -----------------------------------
JOB_ID      = os.environ.get("SLURM_JOB_ID", "local")
_TOTAL_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

# Outer parallelism: one worker per grid point evaluated concurrently.
# C // 8 workers leaves each worker ~8 CPUs for SCMPrior's loky pool.
# Capped at 32 to keep wall-clock predictable on very large allocations.
N_WORKERS      = max(1, min(32, _TOTAL_CPUS // 8))

# Inner parallelism: SCMPrior uses loky with this many subprocesses to
# parallelise generate_dataset() calls within a single get_batch() call.
_SCMPRIOR_JOBS = max(1, _TOTAL_CPUS // N_WORKERS)

# --- Paths ---------------------------------------------------------
DATA_ROOT = Path(os.environ.get("TAB_DATA_ROOT", "."))
OUT_DIR   = Path(os.environ.get("TAB_OUT_DIR", f"grid-results-{JOB_ID}"))
CACHE_DIR = OUT_DIR / "cache"

REAL_SOURCES = {
    "t4":    DATA_ROOT / "tables-t4/parquet",
    "tabfm": DATA_ROOT / "tables-tabfm/parquet",
}

# --- Evaluation settings ------------------------------------------
N_EVAL       = int(os.environ.get("TAB_N_EVAL",    200))   # synthetic (and real) tables per discriminator evaluation
N_REAL       = int(os.environ.get("TAB_N_REAL",   5000))  # real tables to pre-featurize and cache
MAX_ROWS     = int(os.environ.get("TAB_MAX_ROWS",  512))   # max rows sampled per table
BATCH_SIZE   = 8      # SCMPrior batch size
TEST_FRAC    = 0.3
RANDOM_STATE = 42
GRANULARITY  = int(os.environ.get("TAB_GRANULARITY", 5))   # grid points for int/float parameters (cat params use all values)
WANDB_PROJECT = "tabicl-grid-search"  # set to None to disable wandb

# --- XGBoost device -----------------------------------------------
# Forced to CPU: each worker would otherwise contend for the single
# H100 GPU on a GH200 node.  The discriminator fits are tiny (≤280
# rows, 300 trees) so CPU is fast enough.
XGB_DEVICE = "cpu"

# --- Parameter space ----------------------------------------------
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"real_feats_{source_name}_{N_REAL}_{MAX_ROWS}.npz"

    if cache_path.exists():
        print(f"  Loading cached features: {cache_path}", flush=True)
        return np.load(cache_path)["feats"]

    print(f"  Loading up to {N_REAL} real tables from {path} ...", flush=True)
    dfs = [df for df, _ in load_dataframes(
        path, label=0, max_rows=MAX_ROWS, max_tables=N_REAL, random_state=RANDOM_STATE,
    )]
    print(f"  Featurizing {len(dfs)} tables ...", flush=True)
    feats = np.array([df_to_vec(df) for df in tqdm(dfs, desc="  featurizing", unit="table")])
    np.savez(cache_path, feats=feats)
    print(f"  Cached -> {cache_path}  shape={feats.shape}", flush=True)
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
        print(f"    [generate error] {exc}", flush=True)
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
        path = OUT_DIR / f"grid_plot_{source_name}_{name}_{JOB_ID}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  Plot saved -> {path}", flush=True)
        if run is not None:
            run.log({f"{source_name}/{name}": wandb.Image(str(path))})
        plt.close(fig)


# ------------------------------------------------------------------
# Grid search (single source)
# ------------------------------------------------------------------

def run_grid_search(source_name: str, real_feats: np.ndarray) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}", flush=True)
    print(f"  Grid search: tabicl  vs  {source_name}", flush=True)
    print(f"{'#'*60}", flush=True)

    # Derive feature range from cached features (feature index 0 = n_cols)
    col_counts   = real_feats[:, 0].astype(int)
    fixed_params = {
        "min_features": int(col_counts.min()),
        "max_features": int(col_counts.max()),
    }
    print(
        f"  Derived: min_features={fixed_params['min_features']}  "
        f"max_features={fixed_params['max_features']}",
        flush=True,
    )

    grid    = build_grid(GRANULARITY)

    # ------------------------------------------------------------------
    # Slurm array sharding: if this is one task in a job array, slice
    # the grid so each node covers an independent subset.
    # ------------------------------------------------------------------
    _n_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    _shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID",    0))
    if _n_shards > 1:
        grid = grid[_shard_id::_n_shards]
        print(f"  Array shard {_shard_id}/{_n_shards}: {len(grid)} of {len(build_grid(GRANULARITY))} combos", flush=True)

    n_total = len(grid)
    print(f"  Grid size: {n_total} combinations  (GRANULARITY={GRANULARITY})", flush=True)
    print(f"  XGB device: {XGB_DEVICE}", flush=True)
    print(f"  N_WORKERS={N_WORKERS}  _SCMPRIOR_JOBS={_SCMPRIOR_JOBS}", flush=True)

    run = None
    if _WANDB_AVAILABLE and WANDB_PROJECT:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"grid_{source_name}_{JOB_ID}",
            config={
                "source":          source_name,
                "n_eval":          N_EVAL,
                "n_real":          len(real_feats),
                "max_rows":        MAX_ROWS,
                "granularity":     GRANULARITY,
                "grid_size":       n_total,
                "xgb_device":      XGB_DEVICE,
                "n_workers":       N_WORKERS,
                "scmprior_jobs":   _SCMPRIOR_JOBS,
                "total_cpus":      _TOTAL_CPUS,
                "job_id":          JOB_ID,
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
            print(f"  [{done:5d}/{n_total}] (trial {i:5d}) AUC={auc:.4f}  best={best_auc:.4f}  {params}", flush=True)
            if run is not None:
                run.log({"trial": i, "auc": auc, "best_auc": best_auc, **params})

    df_results = pd.DataFrame(records)
    out_path   = OUT_DIR / f"grid_results_{source_name}_{JOB_ID}.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}", flush=True)

    # Summary table + plots
    if run is not None:
        table = wandb.Table(dataframe=df_results)
        run.log({f"{source_name}/results_table": table})

    print(f"  Generating summary plots ...", flush=True)
    log_summary_plots(df_results, source_name, run)

    if run is not None:
        run.finish()

    return df_results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print(f"SLURM_JOB_ID        = {JOB_ID}", flush=True)
    print(f"SLURM_CPUS_PER_TASK = {_TOTAL_CPUS}", flush=True)
    print(f"N_WORKERS           = {N_WORKERS}", flush=True)
    print(f"_SCMPRIOR_JOBS      = {_SCMPRIOR_JOBS}", flush=True)
    print(f"OUT_DIR             = {OUT_DIR}", flush=True)
    print(f"DATA_ROOT           = {DATA_ROOT}", flush=True)
    print(f"XGB_DEVICE          = {XGB_DEVICE}", flush=True)

    missing = [n for n, p in REAL_SOURCES.items() if not any(p.glob("*.parquet"))]
    if missing:
        print("Missing real-data tables for:", ", ".join(missing), flush=True)
        print("Run cleantab-export.py and tabfm-export.py first.", flush=True)
        sys.exit(1)

    print(f"Pre-computing / loading real-data features  (N_REAL={N_REAL}, MAX_ROWS={MAX_ROWS}) ...", flush=True)
    real_feats_by_source = {}
    for name, path in REAL_SOURCES.items():
        real_feats_by_source[name] = load_or_compute_real_feats(name, path)
        print(f"  {name}: {real_feats_by_source[name].shape[0]} tables cached", flush=True)

    all_results = {}
    for source_name, real_feats in real_feats_by_source.items():
        df_res = run_grid_search(source_name, real_feats)
        all_results[source_name] = df_res

    print(f"\n{'#'*60}", flush=True)
    print("  TOP-10 RESULTS PER SOURCE  (lower AUC = more realistic)", flush=True)
    print(f"{'#'*60}", flush=True)
    for source_name, df_res in all_results.items():
        top10 = df_res.nsmallest(10, "auc")
        print(f"\n  [{source_name}]", flush=True)
        print(top10.to_string(index=False), flush=True)
