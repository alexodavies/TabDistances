"""Bayesian optimisation of TabICL prior parameters — multi-restart TPE.

Runs N_RESTARTS independent Optuna TPE studies per real-data source.
Each restart starts from a fresh random position in parameter space so the
optimiser can escape local optima.  Results are persisted to disk after
every restart, so the script can be interrupted safely.

Objective: maximise combined_score(recall, precision, AUC)
  = Euclidean distance from origin toward the ideal (1, 1, 1) point,
    normalised to [0, 1].

Usage
-----
    python bayesian.py

Configuration
-------------
All tunable constants are in the CONFIG section below.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import optuna
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.discriminator.dataset import load_dataframes
from tabdistances.discriminator.features import df_to_vec, select_features
from tabdistances.discriminator.loss import combined_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
REAL_SOURCES = {
    "tabfm": Path("tables-tabfm/parquet"),
    "t4":    Path("tables-t4/parquet"),
}

N_EVAL        = 500    # synthetic (and real) tables per fitness evaluation
N_REAL_CACHE  = 10000  # real tables to pre-featurize and cache
MAX_ROWS      = 512    # max rows sampled per table
BATCH_SIZE    = 1      # SCMPrior batch size
RANDOM_STATE  = 42
N_TRIALS      = 100    # Optuna trials per restart
N_RESTARTS    = 100    # independent TPE restarts (each with a fresh random seed)
FEATURE_MODE  = "full"
CACHE_DIR     = Path("feature-cache")
OUT_DIR       = Path("bayesian-results")
WANDB_PROJECT = "tabicl-coverage-bayesian"

# Coverage k-NN settings (must match grid-isambard.py)
_KNN_K      = 5
_COVERAGE_Q = 0.95


# ------------------------------------------------------------------
# Real-data feature caching
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
# Parameter proposal
# ------------------------------------------------------------------

def make_params(trial: optuna.Trial) -> dict:
    params = {
        "max_classes":             trial.suggest_int("max_classes", 2, 20),
        "batch_size_per_gp":       trial.suggest_int("batch_size_per_gp", 1, BATCH_SIZE),
        "prior_type":              trial.suggest_categorical(
                                       "prior_type", ["mlp_scm", "tree_scm", "mix_scm"]),
        "tree_model":              trial.suggest_categorical(
                                       "tree_model", ["decision_tree", "extra_trees",
                                                      "random_forest", "xgboost"]),
        "mix_prob_mlp":            trial.suggest_float("mix_prob_mlp", 0.0, 1.0),
        "cat_prob":                trial.suggest_float("cat_prob", 0.0, 1.0),
        "multiclass_ordered_prob": trial.suggest_float("multiclass_ordered_prob", 0.0, 1.0),
        "balanced":                trial.suggest_categorical("balanced", [False, True]),
        "replay_small":            trial.suggest_categorical("replay_small", [False, True]),
    }
    trial.set_user_attr("params", params)
    return params


# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------

def _make_prior(params: dict, n_cols: int) -> SCMPrior:
    return SCMPrior(
        batch_size=BATCH_SIZE,
        batch_size_per_gp=params["batch_size_per_gp"],
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
        },
        n_jobs=1,
        device="cpu",
    )


def generate_synthetic_dataframes(params: dict, target_col_counts: list) -> list:
    """Generate synthetic tables whose column counts match *target_col_counts*.

    Groups requests by unique column count; one SCMPrior per unique count.
    Returns tables in the same order as *target_col_counts*.
    """
    count_map = Counter(target_col_counts)
    results_by_count: dict = {}

    for n_cols, n_needed in count_map.items():
        prior = _make_prior(params, n_cols)
        dfs: list = []
        while len(dfs) < n_needed:
            X, y, d, seq_lens, _ = prior.get_batch()
            for i in range(X.shape[0]):
                seq_len = int(seq_lens[i].item())
                n_feat  = int(d[i].item())
                if seq_len == 0:
                    continue
                X_i = X[i, :seq_len, :n_feat].numpy().astype(float)
                y_i = y[i, :seq_len].numpy().astype(int)
                df = pd.DataFrame(X_i, columns=[f"f{j}" for j in range(n_feat)])
                df["target"] = y_i
                dfs.append(df)
                if len(dfs) >= n_needed:
                    break
        results_by_count[n_cols] = dfs

    # Return in original order
    idx_by_count = {nc: 0 for nc in count_map}
    output = []
    for nc in target_col_counts:
        output.append(results_by_count[nc][idx_by_count[nc]])
        idx_by_count[nc] += 1
    return output


# ------------------------------------------------------------------
# Coverage + discriminator helpers
# ------------------------------------------------------------------

def _knn_coverage(ref_m: np.ndarray, cand_m: np.ndarray) -> float:
    """Fraction of ref points covered by cand."""
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
        print(f"    [auc error] {exc}")
        return 0.5


# ------------------------------------------------------------------
# Fitness
# ------------------------------------------------------------------

def evaluate(
    params: dict,
    real_feats: np.ndarray,
    seed: int,
    running_mean: float = 0.0,
) -> tuple:
    """Return (score, recall, precision, discriminator_auc).

    score = combined_score(...) in [0, 1]; MAXIMISE.
    On generation failure, returns (running_mean, 0, 0, 0.5).
    """
    rng = np.random.default_rng(seed)

    # Sample column counts i.i.d. from the real distribution
    all_col_counts = real_feats[:, 0].astype(int)
    target_col_counts = rng.choice(all_col_counts, size=N_EVAL, replace=True).tolist()

    try:
        synth_dfs = generate_synthetic_dataframes(params, target_col_counts)
    except Exception as exc:
        print(f"    [generate error] {exc}")
        return running_mean, 0.0, 0.0, 0.5

    synth_feats = np.array([df_to_vec(df) for df in synth_dfs])

    # Independent real sample from cache
    idxs = rng.choice(len(real_feats), size=min(N_EVAL, len(real_feats)), replace=False)
    real_sample = real_feats[idxs]

    real_m,  _ = select_features(real_sample, FEATURE_MODE)
    synth_m, _ = select_features(synth_feats, FEATURE_MODE)

    recall    = _knn_coverage(real_m,  synth_m)
    precision = _knn_coverage(synth_m, real_m)
    auc       = _discriminator_auc(real_m, synth_m, seed)
    score     = combined_score(recall, precision, auc)
    return score, recall, precision, auc


# ------------------------------------------------------------------
# Multi-restart Bayesian study (single source)
# ------------------------------------------------------------------

def run_single_restart(
    source_name: str,
    real_feats: np.ndarray,
    restart_idx: int,
    out_csv: Path,
    run=None,
) -> tuple:
    """One independent TPE study.

    Appends all trial rows to *out_csv* immediately (interrupt-safe).
    Returns (best_params, best_score) for this restart.
    """
    seed = RANDOM_STATE + restart_idx * 1000   # well-separated seeds per restart

    running_sum   = [0.0]
    running_count = [0]
    _live_points: list = []

    def objective(trial: optuna.Trial) -> float:
        rm = running_sum[0] / running_count[0] if running_count[0] > 0 else 0.0
        params = make_params(trial)
        trial_seed = seed + trial.number
        score, recall, precision, auc = evaluate(params, real_feats, trial_seed, rm)

        running_sum[0]   += score
        running_count[0] += 1
        _live_points.append([recall, precision, auc])

        print(f"  [r{restart_idx+1:03d}|t{trial.number:03d}] "
              f"score={score:.4f}  Recall={recall:.4f}  Precision={precision:.4f}  auc={auc:.4f}")

        # Live append to CSV
        row = {
            "restart":          restart_idx,
            "trial":            trial.number,
            "score":            score,
            "recall":           recall,
            "precision":        precision,
            "discriminator_auc": auc,
            **{k: (1 if v is True else 0 if v is False else v)
               for k, v in params.items()},
        }
        pd.DataFrame([row]).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False
        )

        if run is not None:
            log_params = {k: (1 if v is True else 0 if v is False else v)
                          for k, v in params.items()}
            scatter_table = wandb.Table(
                data=_live_points,
                columns=["Recall", "Precision", "discriminator_auc"],
            )
            run.log({
                f"{source_name}/r{restart_idx}/score":            score,
                f"{source_name}/r{restart_idx}/recall":           recall,
                f"{source_name}/r{restart_idx}/precision":        precision,
                f"{source_name}/r{restart_idx}/discriminator_auc": auc,
                f"{source_name}/r{restart_idx}/trial":            trial.number,
                f"{source_name}/r{restart_idx}/coverage_scatter": wandb.plot.scatter(
                    scatter_table, "Recall", "Precision",
                    title=f"Recall/Precision scatter r{restart_idx} — {source_name}",
                ),
                **{f"{source_name}/r{restart_idx}/param/{k}": v for k, v in log_params.items()},
            })

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    best_trial  = study.best_trial
    best_params = best_trial.user_attrs["params"]
    best_score  = best_trial.value
    print(f"  Restart {restart_idx+1} done: best_score={best_score:.4f}  trial={best_trial.number}")
    return best_params, best_score


def run_study(source_name: str, real_feats: np.ndarray, run=None) -> tuple:
    """Run N_RESTARTS independent TPE restarts; return global best."""
    print(f"\n{'#'*60}")
    print(f"  Multi-restart Bayesian: tabicl vs {source_name}")
    print(f"  {N_RESTARTS} restarts × {N_TRIALS} trials = {N_RESTARTS * N_TRIALS} total evaluations")
    print(f"{'#'*60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"bayesian_results_{source_name}.csv"
    # Start fresh CSV (remove any previous run's file)
    if out_csv.exists():
        out_csv.unlink()
    print(f"  Live results -> {out_csv}")

    all_results: list[tuple[dict, float]] = []
    for r in range(N_RESTARTS):
        print(f"\n  ===== Restart {r+1}/{N_RESTARTS} =====")
        best_params, best_score = run_single_restart(
            source_name, real_feats, r, out_csv, run=run
        )
        all_results.append((best_params, best_score))

        if run is not None:
            run.log({
                f"{source_name}/restart_best": best_score,
                f"{source_name}/restart":      r,
            })

    # Global best across all restarts
    best_r, (best_params, best_score) = max(
        enumerate(all_results), key=lambda x: x[1][1]
    )
    print(f"\n  Global best: restart={best_r+1}  score={best_score:.4f}")

    if run is not None:
        run.summary[f"{source_name}/best_score"]   = best_score
        run.summary[f"{source_name}/best_restart"] = best_r + 1

    return best_params, best_score


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    missing = [n for n, p in REAL_SOURCES.items() if not any(p.glob("*.parquet"))]
    if missing:
        print("Missing real-data tables for:", ", ".join(missing))
        print("Run export scripts first.")
        sys.exit(1)

    print("Loading real feature vectors...")
    real_feats_by_source = {}
    for name, path in REAL_SOURCES.items():
        real_feats_by_source[name] = load_or_compute_real_feats(name, path)
        print(f"  {name}: {real_feats_by_source[name].shape[0]} tables")

    results = {}
    for source_name, real_feats in real_feats_by_source.items():
        run = None
        if _WANDB_AVAILABLE:
            run = wandb.init(
                project=WANDB_PROJECT,
                name=f"bayesian_{source_name}",
                config={
                    "source":        source_name,
                    "n_eval":        N_EVAL,
                    "n_real":        len(real_feats),
                    "n_trials":      N_TRIALS,
                    "n_restarts":    N_RESTARTS,
                    "total_trials":  N_RESTARTS * N_TRIALS,
                    "feature_mode":  FEATURE_MODE,
                    "score_formula": "sqrt(fwd^2 + rev^2 + ((1-auc)*2)^2) / sqrt(3)",
                },
                reinit=True,
            )

        best_params, best_score = run_study(source_name, real_feats, run=run)
        results[source_name] = (best_params, best_score)

        if run is not None:
            run.finish()

    print(f"\n{'#'*60}")
    print("  RESULTS")
    print(f"{'#'*60}")
    for source_name, (params, score) in results.items():
        print(f"\n  [{source_name}]  best score = {score:.4f}  (higher = more realistic)")
        for k, v in params.items():
            print(f"    {k}: {v}")
