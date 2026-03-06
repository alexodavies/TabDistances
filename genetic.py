"""Genetic optimisation of TabICL prior parameters.

Runs one independent GA per real-data source (cleantablib, tabfm).
Each GA minimises the ROC-AUC of a binary discriminator that tries to
separate TabICL-generated tables from that source's tables.
A lower ROC-AUC means the prior's output is less distinguishable from the
target real tables — i.e. the prior is more realistic for that source.

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.discriminator.dataset import load_dataframes
from tabdistances.discriminator.models.tabicl_disc import SmallTabICLDiscriminator

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
REAL_SOURCES = {
    "tabfm": Path("tables-tabfm/parquet"),
    "t4":    Path("tables-t4/parquet"),
}

N_EVAL        = 50    # synthetic tables generated per fitness evaluation
N_REAL        = 200   # real tables sampled per source (cap for speed)
MAX_ROWS      = 4096  # max rows sampled per table (large tables trimmed)
BATCH_SIZE    = 8     # SCMPrior batch size (fixed; not optimised)
TEST_FRAC     = 0.3   # fraction held out for AUC estimation
RANDOM_STATE  = 42

# Genetic algorithm
POP_SIZE      = 12
N_GENERATIONS = 10
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
    """Lower fitness = better (minimise ROC-AUC)."""
    idxs = random.sample(range(len(population)), TOURNAMENT_K)
    winner = min(idxs, key=lambda i: fitnesses[i])
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

def evaluate(params: dict, real_dfs: list) -> float:
    """ROC-AUC for tabicl vs one real source.  Lower is better."""
    try:
        synth = generate_synthetic_dataframes(params, len(real_dfs))
    except Exception as exc:
        print(f"    [generate error] {exc}")
        return 1.0

    X_all = synth + real_dfs
    y_all = [0] * len(synth) + [1] * len(real_dfs)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=TEST_FRAC,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    clf = SmallTabICLDiscriminator(num_classes=2)
    clf.fit(X_tr, np.array(y_tr))
    proba = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


# ------------------------------------------------------------------
# GA (single source)
# ------------------------------------------------------------------

def run_ga(source_name: str, real_dfs: list) -> tuple[dict, float]:
    """Run the genetic algorithm against one real-data source.

    Returns the best (params, auc) found.
    """
    print(f"\n{'#'*60}")
    print(f"  GA: tabicl  vs  {source_name}")
    print(f"{'#'*60}")

    # Compute column range directly from the real DataFrames
    col_counts   = [len(df.columns) for df in real_dfs]
    fixed_params = {
        "min_features": min(col_counts),
        "max_features": max(col_counts),
    }
    print(f"  Fitted from data: min_features={fixed_params['min_features']}  "
          f"max_features={fixed_params['max_features']}")

    population = [sample_individual() for _ in range(POP_SIZE)]
    fitnesses  = [None] * POP_SIZE
    best_params, best_fit = None, float("inf")

    for gen in range(N_GENERATIONS):
        print(f"\n{'='*60}")
        print(f"  Generation {gen + 1} / {N_GENERATIONS}  [{source_name}]")
        print(f"{'='*60}")

        for idx, ind in enumerate(population):
            if fitnesses[idx] is not None:
                continue
            print(f"\n  Individual {idx + 1}/{POP_SIZE}: {ind}")
            fitnesses[idx] = evaluate({**ind, **fixed_params}, real_dfs)
            print(f"  → AUC = {fitnesses[idx]:.4f}")

            if fitnesses[idx] < best_fit:
                best_fit    = fitnesses[idx]
                best_params = dict(ind)

        ranked = sorted(range(POP_SIZE), key=lambda i: fitnesses[i])
        print(f"\n  Generation {gen + 1} summary  [{source_name}]:")
        for rank, idx in enumerate(ranked):
            print(f"    [{rank+1}] AUC={fitnesses[idx]:.4f}  {population[idx]}")

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

    return best_params, best_fit


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    missing = [n for n, p in REAL_SOURCES.items() if not any(p.glob("*.parquet"))]
    if missing:
        print("Missing real-data tables for:", ", ".join(missing))
        print("Run cleantab-export.py and tabfm-export.py first.")
        sys.exit(1)

    # Load real DataFrames once per source (subsample for speed)
    print("Loading real tables...")
    rng = np.random.default_rng(RANDOM_STATE)
    real_dfs_by_source = {}
    for name, path in REAL_SOURCES.items():
        dfs = [df for df, _ in load_dataframes(path, label=0)]
        if len(dfs) > N_REAL:
            idxs = rng.choice(len(dfs), N_REAL, replace=False)
            dfs = [dfs[i] for i in idxs]
        dfs = [df.sample(n=MAX_ROWS, random_state=RANDOM_STATE) if len(df) > MAX_ROWS else df
               for df in dfs]
        real_dfs_by_source[name] = dfs
        print(f"  {name}: {len(dfs)} tables")

    # Run one GA per source
    random.seed(RANDOM_STATE)
    results = {}
    for source_name, real_dfs in real_dfs_by_source.items():
        best_params, best_fit = run_ga(source_name, real_dfs)
        results[source_name] = (best_params, best_fit)

    # Final summary
    print(f"\n{'#'*60}")
    print("  RESULTS")
    print(f"{'#'*60}")
    for source_name, (params, auc) in results.items():
        print(f"\n  [{source_name}]  best AUC = {auc:.4f}  (lower = more realistic)")
        for k, v in params.items():
            print(f"    {k}: {v}")
