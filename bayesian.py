"""Bayesian optimisation of TabICL prior parameters using Optuna (TPE).

Runs one independent study per real-data source (cleantablib, tabfm).
Each study maximises k-NN coverage: the fraction of real tables whose nearest
synthetic neighbours fall within the synthetic set's internal distance
threshold.  Higher coverage means the prior's output spans more of the
real distribution.

Usage
-----
Populate the real-data table directories first, then run::

    python cleantab-export.py
    python tabfm-export.py
    python bayesian.py

Configuration
-------------
All tunable constants are in the CONFIG section below.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd
import optuna

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.discriminator.dataset import load_dataframes
from tabdistances.discriminator.features import df_to_vec
from coverage_analysis import analyse_coverage, _select_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
REAL_SOURCES = {
    "tabfm":       Path("tables-tabfm/parquet"),
    "t4":          Path("tables-t4/parquet"),
    # "cleantablib": Path("tables/parquet"),
}

N_EVAL       = 200    # synthetic tables generated per fitness evaluation
N_REAL_CACHE = 10000   # real tables to pre-featurize and cache
N_REAL_EVAL  = 200    # real tables sampled per trial (from cache)
MAX_ROWS     = 512    # max rows sampled per table (discriminator caps at 512 anyway)
BATCH_SIZE   = 1      # SCMPrior batch size (fixed; not optimised)
RANDOM_STATE = 42
N_TRIALS     = 100    # Optuna trials per source
FEATURE_MODE = "full"   # feature selection mode for coverage: "full" | "scalar_only" | "hist_only"
CACHE_DIR    = Path("feature-cache")
WANDB_PROJECT = "tabicl-coverage-bayesian"


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
    """Suggest a parameter set for one Optuna trial.

    min_features / max_features are fitted from real data and injected
    externally; they are not suggested here.
    """
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
# Synthetic data generation  (identical to genetic.py)
# ------------------------------------------------------------------

def generate_synthetic_dataframes(params: dict, n: int) -> list:
    prior = SCMPrior(
        batch_size=BATCH_SIZE,
        batch_size_per_gp=params["batch_size_per_gp"],
        min_features=params["min_features"],
        max_features=params["max_features"],
        max_classes=params["max_classes"],
        max_seq_len=1024,
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

    dataframes = []
    while len(dataframes) < n:
        X, y, d, seq_lens, _ = prior.get_batch()
        for i in range(X.shape[0]):
            n_feat  = int(d[i].item())
            seq_len = int(seq_lens[i].item())
            if n_feat < params["min_features"] or seq_len == 0:
                continue
            X_i = X[i, :seq_len, :n_feat].numpy().astype(float)
            y_i = y[i, :seq_len].numpy().astype(int)
            df = pd.DataFrame(X_i, columns=[f"f{j}" for j in range(n_feat)])
            df["target"] = y_i
            dataframes.append(df)
            if len(dataframes) >= n:
                break

    return dataframes


# ------------------------------------------------------------------
# Fitness (single source)  (identical to genetic.py)
# ------------------------------------------------------------------

def evaluate(params: dict, real_feats: np.ndarray) -> tuple[float, float, float]:
    """Return (score, coverage_real_by_synth, coverage_synth_by_real).

    score = coverage_real_by_synth - (1 - coverage_synth_by_real), range [-1, 1].
    """
    try:
        synth_dfs = generate_synthetic_dataframes(params, N_EVAL)
    except Exception as exc:
        print(f"    [generate error] {exc}")
        return 0.0, 0.0, 0.0

    # Sample a fresh subset of real tables for diversity across trials
    idxs = np.random.choice(len(real_feats), size=min(N_REAL_EVAL, len(real_feats)), replace=False)
    real_sample = real_feats[idxs]

    synth_feats = np.array([df_to_vec(df) for df in synth_dfs])
    real_m,  _  = _select_features(real_sample,  FEATURE_MODE)
    synth_m, _  = _select_features(synth_feats, FEATURE_MODE)

    fwd, _ = analyse_coverage(real_m, synth_m, None, "", label_ref="real",  label_cand="synth")
    rev, _ = analyse_coverage(synth_m, real_m, None, "", label_ref="synth", label_cand="real")
    coverage_real_by_synth = float(1.0 - fwd["uncovered_fraction"])
    coverage_synth_by_real = float(1.0 - rev["uncovered_fraction"])
    score = coverage_real_by_synth - (1.0 - coverage_synth_by_real)
    return score, coverage_real_by_synth, coverage_synth_by_real


# ------------------------------------------------------------------
# Bayesian study (single source)
# ------------------------------------------------------------------

def run_study(source_name: str, real_feats: np.ndarray, run=None) -> tuple[dict, float]:
    """Run an Optuna TPE study against one real-data source.

    real_feats: pre-featurized array shape (N, 70); index 0 = n_cols.
    Returns the best (params, coverage) found.
    """
    print(f"\n{'#'*60}")
    print(f"  Bayesian: tabicl  vs  {source_name}  ({N_TRIALS} trials)")
    print(f"{'#'*60}")

    # Derive column range from cached feature vectors (index 0 = n_cols)
    col_counts   = real_feats[:, 0].astype(int)
    fixed_params = {
        "min_features": int(col_counts.min()),
        "max_features": int(col_counts.max()),
    }
    print(f"  Fitted from data: min_features={fixed_params['min_features']}  "
          f"max_features={fixed_params['max_features']}")
    print(f"  Using {len(real_feats)} cached real feature vectors")

    _running_best = [0.0]

    def objective(trial: optuna.Trial) -> float:
        params                       = {**make_params(trial), **fixed_params}
        score, fwd_cov, rev_cov      = evaluate(params, real_feats)
        print(f"  Trial {trial.number:3d}: score={score:.4f}  "
              f"real_by_synth={fwd_cov:.4f}  synth_by_real={rev_cov:.4f}  {params}")

        if score > _running_best[0]:
            _running_best[0] = score

        if run is not None:
            log_params = {k: (1 if v is True else 0 if v is False else v)
                          for k, v in trial.user_attrs["params"].items()}
            run.log({
                f"{source_name}/score":         score,
                f"{source_name}/best_score":    _running_best[0],
                f"{source_name}/real_by_synth": fwd_cov,
                f"{source_name}/synth_by_real": rev_cov,
                f"{source_name}/trial":         trial.number,
                **{f"{source_name}/param/{k}": v for k, v in log_params.items()},
            })

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    best_trial     = study.best_trial
    best_params    = best_trial.user_attrs["params"]
    best_coverage  = best_trial.value

    print(f"\n  Best trial: {best_trial.number}  score={best_coverage:.4f}")
    return best_params, best_coverage


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    missing = [n for n, p in REAL_SOURCES.items() if not any(p.glob("*.parquet"))]
    if missing:
        print("Missing real-data tables for:", ", ".join(missing))
        print("Run cleantab-export.py and tabfm-export.py first.")
        sys.exit(1)

    # Load / compute real feature vectors once per source
    print("Loading real feature vectors...")
    real_feats_by_source = {}
    for name, path in REAL_SOURCES.items():
        real_feats_by_source[name] = load_or_compute_real_feats(name, path)
        print(f"  {name}: {real_feats_by_source[name].shape[0]} tables")

    # Run one study per source
    results = {}
    for source_name, real_feats in real_feats_by_source.items():
        run = None
        if _WANDB_AVAILABLE:
            run = wandb.init(
                project=WANDB_PROJECT,
                name=f"bayesian_{source_name}",
                config={
                    "source":       source_name,
                    "n_eval":       N_EVAL,
                    "n_real":       len(real_feats),
                    "n_trials":     N_TRIALS,
                    "feature_mode": FEATURE_MODE,
                },
                reinit=True,
            )

        best_params, best_coverage = run_study(source_name, real_feats, run=run)
        results[source_name] = (best_params, best_coverage)

        if run is not None:
            run.summary[f"{source_name}/best_score"] = best_coverage
            run.finish()

    # Final summary
    print(f"\n{'#'*60}")
    print("  RESULTS")
    print(f"{'#'*60}")
    for source_name, (params, coverage) in results.items():
        print(f"\n  [{source_name}]  best coverage = {coverage:.4f}  (higher = more realistic)")
        for k, v in params.items():
            print(f"    {k}: {v}")
