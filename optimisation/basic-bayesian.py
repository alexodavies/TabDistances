"""Bayesian optimisation of TabICL prior parameters — fast RF variant.

Uses aggregate table features fed into a Random Forest discriminator (much
faster than the deep-learning discriminator in bayesian.py).  Objective:
minimise ROC-AUC (lower = harder for the RF to separate real from synthetic
= more realistic synthetic data).

Runs N_RESTARTS independent Optuna TPE studies per source.  Results are
persisted to disk after every trial so the script can be interrupted safely.

Usage
-----
    python basic-bayesian.py

Configuration
-------------
All tunable constants are in the CONFIG section below.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.discriminator.dataset import load_dataframes
from tabdistances.discriminator.models.rf import RFDiscriminator

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
REAL_SOURCES = {
    "t4":    Path("tables-t4/parquet"),
    "tabfm": Path("tables-tabfm/parquet"),
}

N_EVAL       = 500   # synthetic tables per fitness evaluation
N_REAL       = 500   # real tables loaded per source
MAX_ROWS     = 1024  # max rows sampled per table
BATCH_SIZE   = 8     # SCMPrior batch size
TEST_FRAC    = 0.3
RANDOM_STATE = 42
N_TRIALS     = 100   # Optuna trials per restart
N_RESTARTS   = 100   # independent TPE restarts per source

OUT_DIR = Path("basic-bayesian-results")


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
    """Generate synthetic tables whose column counts match *target_col_counts*."""
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
    idx_by_count = {nc: 0 for nc in count_map}
    output = []
    for nc in target_col_counts:
        output.append(results_by_count[nc][idx_by_count[nc]])
        idx_by_count[nc] += 1
    return output


# ------------------------------------------------------------------
# Fitness
# ------------------------------------------------------------------

def evaluate(
    params: dict,
    real_dfs: list,
    seed: int,
    running_mean: float = 1.0,
) -> float:
    """ROC-AUC for tabicl vs one real source.  Lower = more realistic.

    Column counts for synthetic tables are sampled i.i.d. from the real
    distribution.  On generation failure returns *running_mean* (the
    current average AUC across completed trials).
    """
    rng = np.random.default_rng(seed)
    col_counts = [len(df.columns) for df in real_dfs]
    target_col_counts = rng.choice(col_counts, size=N_EVAL, replace=True).tolist()

    try:
        synth = generate_synthetic_dataframes(params, target_col_counts)
    except Exception as exc:
        print(f"    [generate error] {exc}")
        return running_mean

    # Sample N_EVAL real tables (with replacement to allow N_EVAL > N_REAL)
    real_idxs = rng.choice(len(real_dfs), size=N_EVAL, replace=True).tolist()
    real_sample = [real_dfs[i] for i in real_idxs]

    X_all = synth + real_sample
    y_all = [0] * len(synth) + [1] * len(real_sample)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=TEST_FRAC,
        random_state=seed,
        stratify=y_all,
    )
    clf = RFDiscriminator(random_state=seed)
    clf.fit(X_tr, np.array(y_tr))
    proba = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


# ------------------------------------------------------------------
# Multi-restart Bayesian study (single source)
# ------------------------------------------------------------------

def run_single_restart(
    source_name: str,
    real_dfs: list,
    restart_idx: int,
    out_csv: Path,
) -> tuple:
    """One independent TPE study (direction=minimize AUC).

    Appends all trial rows to *out_csv* immediately (interrupt-safe).
    Returns (best_params, best_auc).
    """
    seed = RANDOM_STATE + restart_idx * 1000

    running_sum   = [0.0]
    running_count = [0]

    def objective(trial: optuna.Trial) -> float:
        rm = running_sum[0] / running_count[0] if running_count[0] > 0 else 1.0
        params = make_params(trial)
        trial_seed = seed + trial.number
        auc = evaluate(params, real_dfs, trial_seed, rm)

        running_sum[0]   += auc
        running_count[0] += 1

        print(f"  [r{restart_idx+1:03d}|t{trial.number:03d}] AUC={auc:.4f}")

        row = {
            "restart": restart_idx,
            "trial":   trial.number,
            "auc":     auc,
            **{k: (1 if v is True else 0 if v is False else v)
               for k, v in params.items()},
        }
        pd.DataFrame([row]).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False
        )
        return auc

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    best_trial  = study.best_trial
    best_params = best_trial.user_attrs["params"]
    best_auc    = best_trial.value
    print(f"  Restart {restart_idx+1} done: best_auc={best_auc:.4f}  trial={best_trial.number}")
    return best_params, best_auc


def run_study(source_name: str, real_dfs: list) -> tuple:
    """Run N_RESTARTS independent TPE restarts; return global best."""
    print(f"\n{'#'*60}")
    print(f"  Multi-restart Bayesian (basic): tabicl vs {source_name}")
    print(f"  {N_RESTARTS} restarts × {N_TRIALS} trials = {N_RESTARTS * N_TRIALS} total evaluations")
    print(f"{'#'*60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"basic_bayesian_results_{source_name}.csv"
    if out_csv.exists():
        out_csv.unlink()
    print(f"  Live results -> {out_csv}")

    all_results: list = []
    for r in range(N_RESTARTS):
        print(f"\n  ===== Restart {r+1}/{N_RESTARTS} =====")
        best_params, best_auc = run_single_restart(source_name, real_dfs, r, out_csv)
        all_results.append((best_params, best_auc))

    # Global best: lowest AUC across all restarts
    best_r, (best_params, best_auc) = min(
        enumerate(all_results), key=lambda x: x[1][1]
    )
    print(f"\n  Global best: restart={best_r+1}  AUC={best_auc:.4f}")
    return best_params, best_auc


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    missing = [n for n, p in REAL_SOURCES.items() if not any(p.glob("*.parquet"))]
    if missing:
        print("Missing real-data tables for:", ", ".join(missing))
        print("Run export scripts first.")
        sys.exit(1)

    print("Loading real tables...")
    real_dfs_by_source = {}
    for name, path in REAL_SOURCES.items():
        dfs = [df for df, _ in load_dataframes(
            path, label=0, max_rows=MAX_ROWS, max_tables=N_REAL, random_state=RANDOM_STATE,
        )]
        real_dfs_by_source[name] = dfs
        print(f"  {name}: {len(dfs)} tables")

    results = {}
    for source_name, real_dfs in real_dfs_by_source.items():
        best_params, best_auc = run_study(source_name, real_dfs)
        results[source_name] = (best_params, best_auc)

    print(f"\n{'#'*60}")
    print("  RESULTS")
    print(f"{'#'*60}")
    for source_name, (params, auc) in results.items():
        print(f"\n  [{source_name}]  best AUC = {auc:.4f}  (lower = more realistic)")
        for k, v in params.items():
            print(f"    {k}: {v}")
