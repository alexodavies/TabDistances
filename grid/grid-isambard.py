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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pebble import ProcessPool
from concurrent.futures import TimeoutError as FutTimeoutError
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tqdm import tqdm

from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.discriminator.dataset import load_dataframes
from tabdistances.discriminator.features import df_to_vec, select_features, FEATURE_MODES
from tabdistances.discriminator.loss import combined_score

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

# --- Slurm / CPU auto-detection -----------------------------------
JOB_ID        = os.environ.get("SLURM_JOB_ID", "local")
ARRAY_JOB_ID  = os.environ.get("SLURM_ARRAY_JOB_ID", JOB_ID)   # shared across all tasks in an array
SHARD_ID      = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
_TOTAL_CPUS   = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

# Outer parallelism: one worker per grid point evaluated concurrently.
# Capped conservatively: each outer worker loads a full SCMPrior + PyTorch
# model, so total RSS scales with N_WORKERS.
N_WORKERS      = max(1, min(16, _TOTAL_CPUS // 16))

# Inner parallelism: loky subprocesses within each SCMPrior get_batch call.
# Capped at 4 so total processes = N_WORKERS * (1 + _SCMPRIOR_JOBS) ≤ ~40
# rather than the 128 that caused OOM.
_SCMPRIOR_JOBS = max(1, min(4, _TOTAL_CPUS // N_WORKERS // 4))

# --- Paths ---------------------------------------------------------
DATA_ROOT = Path(os.environ.get("TAB_DATA_ROOT", "."))
OUT_DIR   = Path(os.environ.get("TAB_OUT_DIR", f"grid-results-{JOB_ID}"))
CACHE_DIR = OUT_DIR / "cache"

REAL_SOURCES = {
    "tabfm": DATA_ROOT / "tables-tabfm/parquet",
    "t4":    DATA_ROOT / "tables-t4/parquet",
}

# --- Evaluation settings ------------------------------------------
N_EVAL       = int(os.environ.get("TAB_N_EVAL",    500))   # synthetic (and real) tables per discriminator evaluation
N_REAL       = int(os.environ.get("TAB_N_REAL",   5000))  # real tables to pre-featurize and cache
MAX_ROWS     = int(os.environ.get("TAB_MAX_ROWS",  512))   # max rows sampled per table
BATCH_SIZE   = 32     # SCMPrior batch size — larger batches amortize loky startup overhead
RANDOM_STATE = 42
GRANULARITY  = int(os.environ.get("TAB_GRANULARITY", 4))   # grid points for int/float parameters (cat params use all values)
# Shard runs land in tabicl-grid-shards (grouped by array job).
# The combine step logs the final merged run to tabicl-grid-search.
WANDB_PROJECT = "tabicl-coverage-shards"  # set to None to disable wandb
FEATURE_MODE  = "full"   # feature selection mode: "full" | "scalar_only" | "hist_only"

# --- Per-trial timeout --------------------------------------------
# If a worker deadlocks (e.g. inside SCMPrior's loky pool), pebble will
# kill the subprocess after this many seconds and record AUC=1.0.
TRIAL_TIMEOUT = 300   # seconds

# --- Parameter space ----------------------------------------------
PARAM_SPACE = {
    # Shared: active for all prior types
    "max_classes":             ("int",   2,    20),
    "cat_prob":                ("float", 0.0,  0.6),
    "multiclass_ordered_prob": ("float", 0.0,  1.0),
    "balanced":                ("cat",   [False, True]),
    "replay_small":            ("cat",   [False, True]),
    # Conditional: prior_type always varies
    "prior_type":              ("cat",   ["mlp_scm", "tree_scm", "mix_scm"]),
    # Conditional: tree_model active for tree_scm and mix_scm only
    "tree_model":              ("cat",   ["decision_tree", "extra_trees",
                                          "random_forest", "xgboost"]),
    # Conditional: mix_prob_mlp active for mix_scm only
    "mix_prob_mlp":            ("float", 0.0,  1.0),
    # Conditional: tree_depth_lambda and tree_n_estimators_lambda active for tree_scm and mix_scm only
    "tree_depth_lambda":       ("float", 0.3,  1.0),
    "tree_n_estimators_lambda":("float", 0.3,  1.0),
}


# ------------------------------------------------------------------
# Grid construction
# ------------------------------------------------------------------

def build_grid(granularity: int) -> list[dict]:
    """Build the parameter grid, respecting conditional relevance:

    - tree_model   : irrelevant for mlp_scm  → fixed to "xgboost"
    - mix_prob_mlp : irrelevant for mlp_scm and tree_scm → fixed to 0.5

    This avoids ~60% redundant evaluations vs a full Cartesian product.

    Grid sizes at G=6:
      mlp_scm  : G^3 x 2^2          =   864
      tree_scm : G^3 x 4 x 2^2      = 3,456
      mix_scm  : G^4 x 4 x 2^2      = 20,736
      total                          = 25,056  (vs 62,208 full grid)
    """
    def _int_vals(lo, hi):
        v = np.linspace(lo, hi, granularity).round().astype(int)
        return list(dict.fromkeys(v.tolist()))

    def _float_vals(lo, hi):
        return np.linspace(lo, hi, granularity).tolist()

    # Params active for every prior type
    shared_axes = {
        "max_classes":             _int_vals(2, 20),
        "cat_prob":                _float_vals(0.0, 0.6),
        "multiclass_ordered_prob": _float_vals(0.0, 1.0),
        "balanced":                [False, True],
        "replay_small":            [False, True],
    }
    tree_models       = ["decision_tree", "extra_trees", "random_forest", "xgboost"]
    mix_prob_vals     = _float_vals(0.0, 1.0)
    tree_depth_vals   = _float_vals(0.3, 1.0)
    tree_n_est_vals   = _float_vals(0.3, 1.0)

    shared_combos = [
        dict(zip(shared_axes, vals))
        for vals in itertools.product(*shared_axes.values())
    ]

    grid = []

    # mlp_scm — tree_model, mix_prob_mlp, tree_depth_lambda, tree_n_estimators_lambda
    #           are all irrelevant; fix to neutral defaults
    for base in shared_combos:
        grid.append({**base, "prior_type": "mlp_scm",
                     "tree_model": "xgboost", "mix_prob_mlp": 0.5,
                     "tree_depth_lambda": 0.5, "tree_n_estimators_lambda": 0.5})

    # tree_scm — mix_prob_mlp irrelevant; vary tree_model + tree params
    for base in shared_combos:
        for tm in tree_models:
            for tdl in tree_depth_vals:
                for tnel in tree_n_est_vals:
                    grid.append({**base, "prior_type": "tree_scm",
                                 "tree_model": tm, "mix_prob_mlp": 0.5,
                                 "tree_depth_lambda": tdl,
                                 "tree_n_estimators_lambda": tnel})

    # mix_scm — all conditional params active
    for base in shared_combos:
        for tm in tree_models:
            for mp in mix_prob_vals:
                for tdl in tree_depth_vals:
                    for tnel in tree_n_est_vals:
                        grid.append({**base, "prior_type": "mix_scm",
                                     "tree_model": tm, "mix_prob_mlp": mp,
                                     "tree_depth_lambda": tdl,
                                     "tree_n_estimators_lambda": tnel})

    return grid


# ------------------------------------------------------------------
# Real-data feature caching
# ------------------------------------------------------------------

def load_or_compute_real_feats(source_name: str, path: Path) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"real_feats_{source_name}_{N_REAL}_{MAX_ROWS}_v2.npz"

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

def _make_prior(params: dict, n_cols: int) -> SCMPrior:
    """Construct a SCMPrior fixed to exactly *n_cols* columns."""
    return SCMPrior(
        batch_size=BATCH_SIZE,
        batch_size_per_gp=1,
        min_features=n_cols,
        max_features=n_cols,
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
            "max_depth_lambda":        params["tree_depth_lambda"],
            "n_estimators_lambda":     params["tree_n_estimators_lambda"],
        },
        n_jobs=_SCMPRIOR_JOBS,
        device="cpu",
    )


def generate_synthetic_dataframes(params: dict, target_col_counts: list[int]) -> list:
    """Generate synthetic tables whose column counts match *target_col_counts*.

    Groups the requested counts by unique value so that only one SCMPrior
    per distinct column count is constructed (avoids repeated model init).
    Returns tables in the same order as *target_col_counts*.
    """
    from collections import Counter
    count_map = Counter(target_col_counts)
    results_by_count: dict[int, list] = {}

    for n_cols, n_needed in count_map.items():
        prior = _make_prior(params, n_cols)
        dfs: list = []
        with tqdm(total=n_needed, desc=f"  gen cols={n_cols}", unit="table", leave=False) as pbar:
            while len(dfs) < n_needed:
                X, y, d, seq_lens, _ = prior.get_batch()
                for i in range(X.shape[0]):
                    n_feat  = int(d[i].item())
                    seq_len = int(seq_lens[i].item())
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

    # Return in the original requested order
    output = []
    idx_by_count: dict[int, int] = {nc: 0 for nc in count_map}
    for nc in target_col_counts:
        output.append(results_by_count[nc][idx_by_count[nc]])
        idx_by_count[nc] += 1
    return output


# ------------------------------------------------------------------
# Coverage + discriminator helpers  (no matplotlib side-effects)
# ------------------------------------------------------------------

_KNN_K      = 5
_COVERAGE_Q = 0.95


def _knn_coverage(ref_m: np.ndarray, cand_m: np.ndarray) -> float:
    """Fraction of ref covered by cand (1 - uncovered_fraction)."""
    lo = np.minimum(ref_m.min(0), cand_m.min(0))
    hi = np.maximum(ref_m.max(0), cand_m.max(0))
    rng = hi - lo
    rng[rng == 0] = 1.0
    ref_n  = (ref_m  - lo) / rng
    cand_n = (cand_m - lo) / rng

    nn_ss = NearestNeighbors(n_neighbors=_KNN_K + 1, algorithm="ball_tree").fit(cand_n)
    d_ss, _ = nn_ss.kneighbors(cand_n)
    threshold = np.percentile(d_ss[:, 1:].mean(axis=1), _COVERAGE_Q * 100)

    nn_rs = NearestNeighbors(n_neighbors=_KNN_K, algorithm="ball_tree").fit(cand_n)
    d_rs, _ = nn_rs.kneighbors(ref_n)
    return float(1.0 - (d_rs.mean(axis=1) > threshold).mean())


def _discriminator_auc(real_m: np.ndarray, synth_m: np.ndarray, seed: int) -> float:
    """XGBoost (100 trees) ROC-AUC for real vs synth. 0.5=indistinguishable."""
    X = np.vstack([real_m, synth_m])
    y = np.concatenate([np.zeros(len(real_m)), np.ones(len(synth_m))])
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        clf = XGBClassifier(n_estimators=100, random_state=seed,
                            n_jobs=1, verbosity=0, eval_metric="logloss")
        clf.fit(X_tr, y_tr)
        return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
    except Exception as exc:
        print(f"    [auc error] {exc}", flush=True)
        return 0.5


# ------------------------------------------------------------------
# Fitness
# ------------------------------------------------------------------

def evaluate(
    params: dict,
    real_feats: np.ndarray,
    seed: int,
    running_mean: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return (score, recall, precision, discriminator_auc).

    score = combined_score(recall, precision, auc) in [0, 1]; MAXIMISE.
    On generation failure or timeout, returns (running_mean, 0, 0, 0.5).
    """
    rng = np.random.default_rng(seed)

    # Sample column counts i.i.d. from the real distribution
    all_col_counts = real_feats[:, 0].astype(int)
    target_col_counts = rng.choice(all_col_counts, size=N_EVAL, replace=True).tolist()

    try:
        synth_dfs = generate_synthetic_dataframes(params, target_col_counts)
    except Exception as exc:
        print(f"    [generate error] {exc}", flush=True)
        return running_mean, 0.0, 0.0, 0.5

    with ThreadPoolExecutor(max_workers=4) as tp:
        synth_feats = np.array(list(tp.map(df_to_vec, synth_dfs)))

    # Subsample real features (independent draw from cache)
    n_real_sample = min(N_EVAL, len(real_feats))
    idxs = rng.choice(len(real_feats), size=n_real_sample, replace=False)
    real_sample = real_feats[idxs]

    real_m,  _ = select_features(real_sample, FEATURE_MODE)
    synth_m, _ = select_features(synth_feats, FEATURE_MODE)

    recall    = _knn_coverage(real_m,  synth_m)
    precision = _knn_coverage(synth_m, real_m)
    auc       = _discriminator_auc(real_m, synth_m, seed)
    score     = combined_score(recall, precision, auc)
    return score, recall, precision, auc


# ------------------------------------------------------------------
# Wandb summary plots
# ------------------------------------------------------------------

_CAT_PARAMS   = [k for k, s in PARAM_SPACE.items() if s[0] == "cat"]
_FLOAT_PARAMS = [k for k, s in PARAM_SPACE.items() if s[0] in ("float", "int")]


def _fig_coverage_histogram(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["score"], bins=30, edgecolor="white", color="#4C72B0")
    ax.axvline(df["score"].max(),  color="red",    linestyle="--", label=f"best={df['score'].max():.4f}")
    ax.axvline(df["score"].mean(), color="orange", linestyle="--", label=f"mean={df['score'].mean():.4f}")
    ax.set_xlabel("Score  (Euclidean quality, normalised to [0, 1])")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution across grid")
    ax.legend()
    fig.tight_layout()
    return fig


def _fig_cat_param_means(df: pd.DataFrame) -> plt.Figure:
    n = len(_CAT_PARAMS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, param in zip(axes, _CAT_PARAMS):
        means = df.groupby(param)["score"].mean().sort_values(ascending=False)
        bars  = ax.bar([str(v) for v in means.index], means.values, color="#DD8452")
        ax.set_title(param)
        ax.set_xlabel("value")
        if ax is axes[0]:
            ax.set_ylabel("mean score")
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Mean score by categorical parameter value", y=1.02)
    fig.tight_layout()
    return fig


def _fig_float_param_scatter(df: pd.DataFrame) -> plt.Figure:
    n = len(_FLOAT_PARAMS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, param in zip(axes, _FLOAT_PARAMS):
        ax.scatter(df[param], df["score"], alpha=0.3, s=10, color="#55A868")
        # trend line
        z = np.polyfit(df[param].astype(float), df["score"], 1)
        xs = np.linspace(df[param].min(), df[param].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color="red", linewidth=1.5)
        ax.set_xlabel(param)
        if ax is axes[0]:
            ax.set_ylabel("Score")
        ax.set_title(param)
    fig.suptitle("Score vs continuous/integer parameters", y=1.02)
    fig.tight_layout()
    return fig


def _fig_parallel_coords(df: pd.DataFrame, top_n: int = 100) -> plt.Figure:
    """Parallel coordinates plot for the top_n trials (highest coverage)."""
    top = df.nlargest(top_n, "score").copy()

    # Encode categoricals to numeric ranks for the plot
    plot_df = pd.DataFrame()
    for col in list(PARAM_SPACE.keys()) + ["score"]:
        if col not in top.columns:
            continue
        vals = top[col]
        if not pd.api.types.is_numeric_dtype(vals):
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

    # Colour by combined score (higher = darker blue)
    cmap   = plt.cm.viridis
    colors = [cmap(v) for v in normed["score"]]

    fig, ax = plt.subplots(figsize=(max(12, len(normed.columns) * 1.4), 5))
    cols = normed.columns.tolist()
    for row, color in zip(normed.itertuples(index=False), colors):
        ax.plot(range(len(cols)), list(row), color=color, alpha=0.4, linewidth=0.8)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Normalised value")
    ax.set_title(f"Parallel coordinates – top {top_n} trials (colour = score, darker = better)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(top["score"].min(), top["score"].max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Score", pad=0.01)
    fig.tight_layout()
    return fig


def log_summary_plots(df_results: pd.DataFrame, source_name: str, run) -> None:
    """Create summary figures and log them to wandb + disk."""
    plots = {
        "coverage_histogram":  _fig_coverage_histogram(df_results),
        "cat_param_means":     _fig_cat_param_means(df_results),
        "float_param_scatter": _fig_float_param_scatter(df_results),
        "parallel_coords":     _fig_parallel_coords(df_results),
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

    # Column counts are sampled from the real distribution inside evaluate();
    # no global min/max fixed_params needed.
    col_counts = real_feats[:, 0].astype(int)
    print(
        f"  Real col counts: min={col_counts.min()}  max={col_counts.max()}  "
        f"median={int(np.median(col_counts))}",
        flush=True,
    )

    grid       = build_grid(GRANULARITY)
    full_size  = len(grid)

    # ------------------------------------------------------------------
    # Slurm array sharding: if this is one task in a job array, slice
    # the grid so each node covers an independent subset.
    # ------------------------------------------------------------------
    _n_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    _shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID",    0))
    if _n_shards > 1:
        grid = grid[_shard_id::_n_shards]
        print(f"  Array shard {_shard_id}/{_n_shards}: {len(grid)} of {full_size} combos", flush=True)

    n_total = len(grid)
    print(f"  Grid size: {n_total} combinations  (GRANULARITY={GRANULARITY})", flush=True)
    print(f"  Feature mode: {FEATURE_MODE}", flush=True)
    print(f"  N_WORKERS={N_WORKERS}  _SCMPRIOR_JOBS={_SCMPRIOR_JOBS}", flush=True)

    run = None
    if _WANDB_AVAILABLE and WANDB_PROJECT:
        run = wandb.init(
            project=WANDB_PROJECT,
            group=ARRAY_JOB_ID,
            name=f"shard{SHARD_ID}_{source_name}",
            config={
                "source":          source_name,
                "n_eval":          N_EVAL,
                "n_real":          len(real_feats),
                "max_rows":        MAX_ROWS,
                "granularity":     GRANULARITY,
                "grid_size":       n_total,
                "feature_mode":    FEATURE_MODE,
                "n_workers":       N_WORKERS,
                "scmprior_jobs":   _SCMPRIOR_JOBS,
                "total_cpus":      _TOTAL_CPUS,
                "job_id":          JOB_ID,
            },
            reinit=True,
        )

    # Submit all trials; each worker gets its own seed for reproducible but
    # independent column-count sampling and real-feature subsampling.
    records       = [None] * n_total
    best_score    = 0.0
    done          = 0
    running_sum   = 0.0
    _live_points: list[list[float]] = []
    with ProcessPool(max_workers=N_WORKERS) as pool:
        futures = {
            pool.schedule(
                evaluate,
                args=[params, real_feats, RANDOM_STATE + i, running_sum / done if done else 0.0],
                timeout=TRIAL_TIMEOUT,
            ): (i, params)
            for i, params in enumerate(grid, 1)
        }
        for fut in as_completed(futures):
            i, params = futures[fut]
            running_mean = running_sum / done if done else 0.0
            try:
                score, recall, precision, auc = fut.result()
            except FutTimeoutError:
                print(f"  (trial {i:5d}) TIMEOUT after {TRIAL_TIMEOUT}s — using running mean", flush=True)
                score, recall, precision, auc = running_mean, 0.0, 0.0, 0.5
            except Exception as exc:
                print(f"  (trial {i:5d}) ERROR: {exc} — using running mean", flush=True)
                score, recall, precision, auc = running_mean, 0.0, 0.0, 0.5
            records[i - 1] = {"A": source_name, "B": "tabicl", **params, "score": score,
                               "recall": recall, "precision": precision,
                               "discriminator_auc": auc}
            _live_points.append([recall, precision, auc])
            done += 1
            running_sum += score
            if score > best_score:
                best_score = score
            print(f"  [{done:5d}/{n_total}] (trial {i:5d}) "
                  f"score={score:.4f}  Recall={recall:.4f}  "
                  f"Precision={precision:.4f}  auc={auc:.4f}  best={best_score:.4f}", flush=True)
            if run is not None:
                scatter_table = wandb.Table(
                    data=_live_points,
                    columns=["Recall", "Precision", "AUC"],
                )
                run.log({
                    "trial":                         i,
                    "A":                             source_name,
                    "B":                             "tabicl",
                    f"{source_name}/score":          score,
                    f"{source_name}/best_score":     best_score,
                    f"{source_name}/recall":         recall,
                    f"{source_name}/precision":      precision,
                    f"{source_name}/discriminator_auc": auc,
                    f"{source_name}/coverage_scatter": wandb.plot.scatter(
                        scatter_table, "Recall", "Precision",
                        title=f"Recall/Precision scatter — {source_name}",
                    ),
                    **{f"{source_name}/{k}": v for k, v in params.items()},
                })

    df_results = pd.DataFrame(records)
    out_path   = OUT_DIR / f"grid_results_{source_name}_{JOB_ID}.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}", flush=True)

    # Summary table + plots
    if run is not None:
        run.log({f"{source_name}/results_table": wandb.Table(dataframe=df_results.rename(columns={
            "recall":    "Recall",
            "precision": "Precision",
        }))})

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
    print(f"FEATURE_MODE        = {FEATURE_MODE}", flush=True)

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
    print("  TOP-10 RESULTS PER SOURCE  (higher score = more realistic)", flush=True)
    print(f"{'#'*60}", flush=True)
    for source_name, df_res in all_results.items():
        top10 = df_res.nlargest(10, "score")[
            ["score", "recall", "precision", "discriminator_auc"]
            + list(PARAM_SPACE.keys())
        ]
        print(f"\n  [{source_name}]", flush=True)
        print(top10.to_string(index=False), flush=True)
