"""Bayesian optimisation of TabICL prior parameters using Optuna (TPE).

Runs one independent study per real-data source (cleantablib, tabfm).
Each study minimises the ROC-AUC of a binary discriminator that tries to
separate TabICL-generated graphs from that source's graphs.

Usage
-----
Populate the real-data graph directories first, then run::

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
import wandb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.graph.builder import build_graph
from tabdistances.discriminator.dataset import load_graphs
from tabdistances.discriminator.models.rf import RFDiscriminator

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
REAL_SOURCES = {
    "cleantablib": Path("graphs/json"),
    "tabfm":       Path("graphs-tabfm/json"),
}

N_EVAL       = 50    # synthetic graphs generated per fitness evaluation
N_REAL       = 200   # real graphs sampled per source (cap for speed)
BATCH_SIZE   = 8     # SCMPrior batch size (fixed; not optimised)
TEST_FRAC    = 0.3   # fraction held out for AUC estimation
RANDOM_STATE = 42
N_TRIALS     = 50    # Optuna trials per source


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
        "cat_prob":                trial.suggest_float("cat_prob", 0.0, 0.6),
        "multiclass_ordered_prob": trial.suggest_float("multiclass_ordered_prob", 0.0, 1.0),
        "balanced":                trial.suggest_categorical("balanced", [False, True]),
        "replay_small":            trial.suggest_categorical("replay_small", [False, True]),
    }
    trial.set_user_attr("params", params)
    return params


# ------------------------------------------------------------------
# Synthetic data generation  (identical to genetic.py)
# ------------------------------------------------------------------

def generate_synthetic_graphs(params: dict, n: int) -> list:
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

    graphs = []
    while len(graphs) < n:
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
            try:
                G = build_graph(df, threshold=0.7, cardinality_threshold=0.2)
                graphs.append(G)
            except Exception:
                pass
            if len(graphs) >= n:
                break

    return graphs


# ------------------------------------------------------------------
# Fitness (single source)  (identical to genetic.py)
# ------------------------------------------------------------------

def evaluate(params: dict, real_graphs: list) -> float:
    """ROC-AUC for tabicl vs one real source.  Lower is better."""
    try:
        synth = generate_synthetic_graphs(params, N_EVAL)
    except Exception as exc:
        print(f"    [generate error] {exc}")
        return 1.0

    X_all = synth + real_graphs
    y_all = [0] * len(synth) + [1] * len(real_graphs)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=TEST_FRAC,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    clf = RFDiscriminator(n_estimators=100)
    clf.fit(X_tr, np.array(y_tr))
    proba = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


# ------------------------------------------------------------------
# Bayesian study (single source)
# ------------------------------------------------------------------

def run_study(source_name: str, real_graphs: list) -> tuple[dict, float]:
    """Run an Optuna TPE study against one real-data source.

    Returns the best (params, auc) found.
    """
    print(f"\n{'#'*60}")
    print(f"  Bayesian: tabicl  vs  {source_name}  ({N_TRIALS} trials)")
    print(f"{'#'*60}")

    # Compute column range directly from the real graphs
    node_counts  = [G.number_of_nodes() for G in real_graphs]
    fixed_params = {
        "min_features": min(node_counts),
        "max_features": max(node_counts),
    }
    print(f"  Fitted from data: min_features={fixed_params['min_features']}  "
          f"max_features={fixed_params['max_features']}")

    wandb.init(
        project="tabdistances",
        name=f"bayesian-{source_name}",
        config={
            "optimizer": "bayesian",
            "source":    source_name,
            "n_eval":    N_EVAL,
            "n_real":    N_REAL,
            "n_trials":  N_TRIALS,
            **fixed_params,
        },
    )

    def objective(trial: optuna.Trial) -> float:
        params = {**make_params(trial), **fixed_params}
        auc = evaluate(params, real_graphs)
        print(f"  Trial {trial.number:3d}: AUC={auc:.4f}  {params}")
        wandb.log({"trial/auc": auc,
                   **{f"param/{k}": v for k, v in trial.params.items()}},
                  step=trial.number)
        return auc

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    best_trial  = study.best_trial
    best_params = best_trial.user_attrs["params"]
    best_auc    = best_trial.value

    print(f"\n  Best trial: {best_trial.number}  AUC={best_auc:.4f}")

    wandb.run.summary["best_auc"] = best_auc
    wandb.run.summary.update({f"best/{k}": v for k, v in best_params.items()})
    wandb.finish()

    return best_params, best_auc


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    missing = [n for n, p in REAL_SOURCES.items() if not any(p.glob("*.json"))]
    if missing:
        print("Missing real-data graphs for:", ", ".join(missing))
        print("Run cleantab-export.py and tabfm-export.py first.")
        sys.exit(1)

    # Load real graphs once per source (subsample for speed)
    print("Loading real graphs...")
    rng = np.random.default_rng(RANDOM_STATE)
    real_graphs_by_source = {}
    for name, path in REAL_SOURCES.items():
        graphs = [g for g, _ in load_graphs(path, label=0)]
        if len(graphs) > N_REAL:
            idxs = rng.choice(len(graphs), N_REAL, replace=False)
            graphs = [graphs[i] for i in idxs]
        real_graphs_by_source[name] = graphs
        print(f"  {name}: {len(graphs)} graphs")

    # Run one study per source
    results = {}
    for source_name, real_graphs in real_graphs_by_source.items():
        best_params, best_auc = run_study(source_name, real_graphs)
        results[source_name] = (best_params, best_auc)

    # Final summary
    print(f"\n{'#'*60}")
    print("  RESULTS")
    print(f"{'#'*60}")
    for source_name, (params, auc) in results.items():
        print(f"\n  [{source_name}]  best AUC = {auc:.4f}  (lower = more realistic)")
        for k, v in params.items():
            print(f"    {k}: {v}")
