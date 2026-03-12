"""Genetic optimisation of TabICL prior parameters.

Runs one independent GA per real-data source (cleantablib, tabfm).
Each GA maximises k-NN coverage: the fraction of real tables whose nearest
synthetic neighbours fall within the synthetic set's internal distance
threshold.  Higher coverage means the prior's output spans more of the
real distribution.

Usage
-----
Populate the real-data table directories first, then run::

    python cleantab-export.py
    python tabfm-export.py
    python genetic.py

Configuration
-------------
All tunable constants are in the CONFIG section below.
"""

import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.discriminator.dataset import load_dataframes
from tabdistances.discriminator.features import df_to_vec
from coverage_analysis import analyse_coverage, _select_features

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
REAL_SOURCES = {
    "tabfm": Path("tables-tabfm/parquet"),
    "t4":    Path("tables-t4/parquet"),
}

N_EVAL        = 50    # synthetic tables generated per fitness evaluation
N_REAL        = 100   # real tables sampled per source (cap for speed)
MAX_ROWS      = 1024  # max rows sampled per table (discriminator caps at 512 anyway)
BATCH_SIZE    = 8     # SCMPrior batch size (fixed; not optimised)
RANDOM_STATE  = 42
FEATURE_MODE  = "full"   # feature selection mode for coverage: "full" | "scalar_only" | "hist_only"
CACHE_DIR     = Path("feature-cache")
WANDB_PROJECT = "tabicl-coverage-genetic"

# Genetic algorithm
POP_SIZE      = 8
N_GENERATIONS = 6
N_ELITE       = 2     # individuals carried over unchanged each generation
TOURNAMENT_K  = 3     # contestants per tournament selection
MUTATION_PROB = 0.25  # per-parameter mutation probability

# ------------------------------------------------------------------
# Parameter space
# ("int",   lo, hi)           — uniform integer in [lo, hi]
# ("float", lo, hi)           — uniform float in [lo, hi]
# ("cat",   [v1, v2, ...])    — one of the listed values
# ------------------------------------------------------------------
PARAM_SPACE = {
    "max_classes":             ("int",   2,    20),
    "batch_size_per_gp":       ("int",   1,    BATCH_SIZE),
    "prior_type":              ("cat",   ["mlp_scm", "tree_scm", "mix_scm"]),
    "tree_model":              ("cat",   ["decision_tree", "extra_trees",
                                          "random_forest", "xgboost"]),
    "mix_prob_mlp":            ("float", 0.0, 1.0),
    "cat_prob":                ("float", 0.0, 0.6),
    "multiclass_ordered_prob": ("float", 0.0, 1.0),
    "balanced":                ("cat",   [False, True]),
    "replay_small":            ("cat",   [False, True]),
}


# ------------------------------------------------------------------
# Real-data feature caching
# ------------------------------------------------------------------

def load_or_compute_real_feats(source_name: str, path: Path) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"real_feats_{source_name}_{N_REAL}_{MAX_ROWS}_v2.npz"

    if cache_path.exists():
        print(f"  Loading cached features: {cache_path}")
        return np.load(cache_path)["feats"]

    print(f"  Loading up to {N_REAL} real tables from {path} ...")
    dfs = [df for df, _ in load_dataframes(
        path, label=0, max_rows=MAX_ROWS, max_tables=N_REAL, random_state=RANDOM_STATE,
    )]
    print(f"  Featurizing {len(dfs)} tables ...")
    feats = np.array([df_to_vec(df) for df in dfs])
    np.savez(cache_path, feats=feats)
    print(f"  Cached -> {cache_path}  shape={feats.shape}")
    return feats


# ------------------------------------------------------------------
# Individual helpers
# ------------------------------------------------------------------

def _sample_param(spec):
    if spec[0] == "int":
        return random.randint(spec[1], spec[2])
    if spec[0] == "float":
        return random.uniform(spec[1], spec[2])
    return random.choice(spec[1])


def _clip_param(spec, value):
    if spec[0] == "int":
        return int(np.clip(value, spec[1], spec[2]))
    if spec[0] == "float":
        return float(np.clip(value, spec[1], spec[2]))
    return value if value in spec[1] else random.choice(spec[1])


def sample_individual() -> dict:
    return {k: _sample_param(v) for k, v in PARAM_SPACE.items()}


def crossover(a: dict, b: dict) -> dict:
    return {k: (a[k] if random.random() < 0.5 else b[k]) for k in PARAM_SPACE}


def mutate(ind: dict) -> dict:
    ind = dict(ind)
    for k, spec in PARAM_SPACE.items():
        if random.random() >= MUTATION_PROB:
            continue
        if spec[0] == "int":
            step = max(1, (spec[2] - spec[1]) // 6)
            ind[k] = _clip_param(spec, ind[k] + random.randint(-step, step))
        elif spec[0] == "float":
            step = (spec[2] - spec[1]) * 0.1
            ind[k] = _clip_param(spec, ind[k] + random.uniform(-step, step))
        else:
            ind[k] = random.choice(spec[1])
    return ind


def tournament_select(population: list, fitnesses: list) -> dict:
    """Higher fitness = better (maximise coverage)."""
    idxs = random.sample(range(len(population)), TOURNAMENT_K)
    winner = max(idxs, key=lambda i: fitnesses[i])
    return population[winner]


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
# Fitness (single source)
# ------------------------------------------------------------------

def evaluate(params: dict, real_feats: np.ndarray) -> float:
    """k-NN coverage of real by synthetic.  Higher = more realistic."""
    try:
        synth_dfs = generate_synthetic_dataframes(params, N_EVAL)
    except Exception as exc:
        print(f"    [generate error] {exc}")
        return 0.0

    synth_feats = np.array([df_to_vec(df) for df in synth_dfs])
    real_m,  _  = _select_features(real_feats,  FEATURE_MODE)
    synth_m, _  = _select_features(synth_feats, FEATURE_MODE)

    result, _ = analyse_coverage(real_m, synth_m, None, "",
                                  label_ref="real", label_cand="synth")
    return float(1.0 - result["uncovered_fraction"])


# ------------------------------------------------------------------
# GA (single source)
# ------------------------------------------------------------------

def run_ga(source_name: str, real_feats: np.ndarray, run=None) -> tuple[dict, float]:
    """Run the genetic algorithm against one real-data source.

    real_feats: pre-featurized array shape (N, 70); index 0 = n_cols.
    Returns the best (params, coverage) found.
    """
    print(f"\n{'#'*60}")
    print(f"  GA: tabicl  vs  {source_name}")
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

    population = [sample_individual() for _ in range(POP_SIZE)]
    fitnesses  = [None] * POP_SIZE
    best_params, best_coverage = None, 0.0
    eval_count = 0

    for gen in range(N_GENERATIONS):
        print(f"\n{'='*60}")
        print(f"  Generation {gen + 1} / {N_GENERATIONS}  [{source_name}]")
        print(f"{'='*60}")

        for idx, ind in enumerate(population):
            if fitnesses[idx] is not None:
                continue
            print(f"\n  Individual {idx + 1}/{POP_SIZE}: {ind}")
            fitnesses[idx] = evaluate({**ind, **fixed_params}, real_feats)
            eval_count += 1
            print(f"  → coverage = {fitnesses[idx]:.4f}")

            if fitnesses[idx] > best_coverage:
                best_coverage = fitnesses[idx]
                best_params   = dict(ind)

            if run is not None:
                run.log({
                    f"{source_name}/coverage":      fitnesses[idx],
                    f"{source_name}/best_coverage": best_coverage,
                    f"{source_name}/eval":          eval_count,
                    f"{source_name}/generation":    gen + 1,
                    **{f"{source_name}/param/{k}": (1 if v is True else 0 if v is False else v)
                       for k, v in ind.items()},
                })

        ranked = sorted(range(POP_SIZE), key=lambda i: fitnesses[i], reverse=True)
        print(f"\n  Generation {gen + 1} summary  [{source_name}]:")
        for rank, idx in enumerate(ranked):
            print(f"    [{rank+1}] coverage={fitnesses[idx]:.4f}  {population[idx]}")

        if gen == N_GENERATIONS - 1:
            break

        elite     = [dict(population[i]) for i in ranked[:N_ELITE]]
        elite_fit = [fitnesses[i] for i in ranked[:N_ELITE]]

        next_pop = elite[:]
        next_fit = elite_fit[:]
        while len(next_pop) < POP_SIZE:
            parent_a = tournament_select(population, fitnesses)
            parent_b = tournament_select(population, fitnesses)
            child    = mutate(crossover(parent_a, parent_b))
            next_pop.append(child)
            next_fit.append(None)

        population, fitnesses = next_pop, next_fit

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

    # Run one GA per source
    random.seed(RANDOM_STATE)
    results = {}
    for source_name, real_feats in real_feats_by_source.items():
        run = None
        if _WANDB_AVAILABLE:
            run = wandb.init(
                project=WANDB_PROJECT,
                name=f"ga_{source_name}",
                config={
                    "source":        source_name,
                    "n_eval":        N_EVAL,
                    "n_real":        len(real_feats),
                    "pop_size":      POP_SIZE,
                    "n_generations": N_GENERATIONS,
                    "feature_mode":  FEATURE_MODE,
                },
                reinit=True,
            )

        best_params, best_coverage = run_ga(source_name, real_feats, run=run)
        results[source_name] = (best_params, best_coverage)

        if run is not None:
            run.summary[f"{source_name}/best_coverage"] = best_coverage
            run.finish()

    # Final summary
    print(f"\n{'#'*60}")
    print("  RESULTS")
    print(f"{'#'*60}")
    for source_name, (params, coverage) in results.items():
        print(f"\n  [{source_name}]  best coverage = {coverage:.4f}  (higher = more realistic)")
        for k, v in params.items():
            print(f"    {k}: {v}")
