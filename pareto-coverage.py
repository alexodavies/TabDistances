"""Multi-objective Pareto-front analysis of TabICL prior parameters.

Runs an Optuna NSGA-II study that optimises BOTH coverage directions
independently, then visualises the Pareto front to reveal whether a genuine
tradeoff exists between:

  coverage_real_by_synth  (recall)  — how much of real is spanned by synthetic
  coverage_synth_by_real  (precision) — how realistic/focused the synthetic is

Usage
-----
    python pareto-coverage.py

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
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
}

N_EVAL        = 200    # synthetic tables generated per evaluation
N_REAL_CACHE  = 10000  # real tables to pre-featurize and cache (reuses bayesian.py cache)
N_REAL_EVAL   = 200    # real tables sampled per evaluation (from cache)
MAX_ROWS      = 512    # must match bayesian.py to share the cache
BATCH_SIZE    = 1
RANDOM_STATE  = 42
N_TRIALS      = 500    # more trials than bayesian.py to map the front densely
FEATURE_MODE  = "full"
CACHE_DIR     = Path("feature-cache")
OUT_DIR       = Path("pareto-results")
WANDB_PROJECT = "tabicl-coverage-pareto"


# ------------------------------------------------------------------
# Real-data feature caching  (same key as bayesian.py — shares cache)
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
# Evaluation
# ------------------------------------------------------------------

def evaluate(params: dict, real_feats: np.ndarray) -> tuple[float, float]:
    """Return (coverage_real_by_synth, coverage_synth_by_real).

    Both in [0, 1]; higher = better for each objective independently.
    """
    try:
        synth_dfs = generate_synthetic_dataframes(params, N_EVAL)
    except Exception as exc:
        print(f"    [generate error] {exc}")
        return 0.0, 0.0

    idxs = np.random.choice(len(real_feats), size=min(N_REAL_EVAL, len(real_feats)), replace=False)
    real_sample = real_feats[idxs]

    synth_feats = np.array([df_to_vec(df) for df in synth_dfs])
    real_m,  _  = _select_features(real_sample,  FEATURE_MODE)
    synth_m, _  = _select_features(synth_feats, FEATURE_MODE)

    fwd, _ = analyse_coverage(real_m, synth_m, None, "", label_ref="real",  label_cand="synth")
    rev, _ = analyse_coverage(synth_m, real_m, None, "", label_ref="synth", label_cand="real")
    return float(1.0 - fwd["uncovered_fraction"]), float(1.0 - rev["uncovered_fraction"])


# ------------------------------------------------------------------
# Pareto helpers
# ------------------------------------------------------------------

def _pareto_mask(fwd: np.ndarray, rev: np.ndarray) -> np.ndarray:
    """Boolean mask; True where point (fwd[i], rev[i]) is Pareto-optimal (maximise both)."""
    n = len(fwd)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        # point i is dominated if any other point j is >= on both axes and > on at least one
        dominated[i] = np.any(
            (fwd >= fwd[i]) & (rev >= rev[i]) & ((fwd > fwd[i]) | (rev > rev[i]))
        )
    return ~dominated


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def _fig_pareto_scatter(df: pd.DataFrame, source_name: str) -> plt.Figure:
    """Scatter all trials coloured by trial order; overlay Pareto front in red."""
    fig, ax = plt.subplots(figsize=(6, 6))

    fwd = df["real_by_synth"].values
    rev = df["synth_by_real"].values
    trial_idx = df["trial"].values
    pareto = df["is_pareto"].values

    sc = ax.scatter(fwd[~pareto], rev[~pareto],
                    c=trial_idx[~pareto], cmap="Blues", s=15, alpha=0.5,
                    vmin=0, vmax=len(df), label="all trials")
    fig.colorbar(sc, ax=ax, label="trial number")

    ax.scatter(fwd[pareto], rev[pareto],
               color="red", s=50, zorder=5, label=f"Pareto front ({pareto.sum()} pts)")

    # Connect Pareto front with a step line (sorted by fwd_cov)
    pf = df[pareto].sort_values("real_by_synth")
    ax.step(pf["real_by_synth"].values, pf["synth_by_real"].values,
            where="post", color="red", linewidth=1.5, zorder=4)

    # Equal-coverage diagonal
    lims = [min(fwd.min(), rev.min()) - 0.02, max(fwd.max(), rev.max()) + 0.02]
    ax.plot(lims, lims, "--", color="grey", linewidth=1, alpha=0.6, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)

    ax.set_xlabel("coverage_real_by_synth  (recall)")
    ax.set_ylabel("coverage_synth_by_real  (precision)")
    ax.set_title(f"Pareto front — {source_name}  ({N_TRIALS} trials)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def _fig_pareto_correlation(df: pd.DataFrame, source_name: str) -> plt.Figure:
    """Scatter of fwd vs rev with regression line and Pearson r to show tradeoff."""
    fig, ax = plt.subplots(figsize=(5, 5))

    fwd = df["real_by_synth"].values
    rev = df["synth_by_real"].values

    ax.scatter(fwd, rev, alpha=0.3, s=10, color="#4C72B0")

    slope, intercept, r, p, _ = stats.linregress(fwd, rev)
    xs = np.linspace(fwd.min(), fwd.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="red", linewidth=1.5,
            label=f"r = {r:.3f}  (p={p:.2e})")

    ax.set_xlabel("coverage_real_by_synth  (recall)")
    ax.set_ylabel("coverage_synth_by_real  (precision)")
    ax.set_title(f"Recall–precision correlation — {source_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def log_plots(df: pd.DataFrame, source_name: str, run) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots = {
        "pareto_scatter":     _fig_pareto_scatter(df, source_name),
        "pareto_correlation": _fig_pareto_correlation(df, source_name),
    }
    for name, fig in plots.items():
        path = OUT_DIR / f"{name}_{source_name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  Plot saved -> {path}")
        if run is not None:
            run.log({f"{source_name}/{name}": wandb.Image(str(path))})
        plt.close(fig)


# ------------------------------------------------------------------
# Pareto study (single source)
# ------------------------------------------------------------------

def run_pareto_study(source_name: str, real_feats: np.ndarray, run=None) -> pd.DataFrame:
    """Run NSGA-II multi-objective study.  Returns a DataFrame of all trials."""
    print(f"\n{'#'*60}")
    print(f"  Pareto study: tabicl  vs  {source_name}  ({N_TRIALS} trials)")
    print(f"{'#'*60}")

    col_counts   = real_feats[:, 0].astype(int)
    fixed_params = {
        "min_features": int(col_counts.min()),
        "max_features": int(col_counts.max()),
    }
    print(f"  Fitted from data: min_features={fixed_params['min_features']}  "
          f"max_features={fixed_params['max_features']}")
    print(f"  Using {len(real_feats)} cached real feature vectors")

    _live_points: list[list[float]] = []  # accumulates [fwd_cov, rev_cov] per trial

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params              = {**make_params(trial), **fixed_params}
        fwd_cov, rev_cov    = evaluate(params, real_feats)
        _live_points.append([fwd_cov, rev_cov])
        print(f"  Trial {trial.number:3d}: real_by_synth={fwd_cov:.4f}  "
              f"synth_by_real={rev_cov:.4f}")

        if run is not None:
            log_p = {k: (1 if v is True else 0 if v is False else v)
                     for k, v in trial.user_attrs["params"].items()}
            scatter_table = wandb.Table(
                data=_live_points,
                columns=["real_by_synth", "synth_by_real"],
            )
            run.log({
                f"{source_name}/real_by_synth": fwd_cov,
                f"{source_name}/synth_by_real": rev_cov,
                f"{source_name}/trial":         trial.number,
                f"{source_name}/coverage_scatter": wandb.plot.scatter(
                    scatter_table,
                    "real_by_synth",
                    "synth_by_real",
                    title=f"Coverage scatter — {source_name}",
                ),
                **{f"{source_name}/param/{k}": v for k, v in log_p.items()},
            })

        return fwd_cov, rev_cov

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    # Collect all trial results into a DataFrame
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        row = {"trial": t.number, "real_by_synth": t.values[0], "synth_by_real": t.values[1]}
        row.update(t.user_attrs.get("params", {}))
        rows.append(row)
    df = pd.DataFrame(rows)

    # Annotate Pareto-optimal points
    df["is_pareto"] = _pareto_mask(df["real_by_synth"].values, df["synth_by_real"].values)

    # Print Pareto front
    front = df[df["is_pareto"]].sort_values("real_by_synth", ascending=False)
    print(f"\n  Pareto front ({len(front)} trials)  [{source_name}]:")
    print(front[["trial", "real_by_synth", "synth_by_real"]].to_string(index=False))

    return df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    missing = [n for n, p in REAL_SOURCES.items() if not any(p.glob("*.parquet"))]
    if missing:
        print("Missing real-data tables for:", ", ".join(missing))
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading real feature vectors...")
    real_feats_by_source = {}
    for name, path in REAL_SOURCES.items():
        real_feats_by_source[name] = load_or_compute_real_feats(name, path)
        print(f"  {name}: {real_feats_by_source[name].shape[0]} tables")

    for source_name, real_feats in real_feats_by_source.items():
        run = None
        if _WANDB_AVAILABLE:
            run = wandb.init(
                project=WANDB_PROJECT,
                name=f"pareto_{source_name}",
                config={
                    "source":       source_name,
                    "n_eval":       N_EVAL,
                    "n_real":       len(real_feats),
                    "n_trials":     N_TRIALS,
                    "feature_mode": FEATURE_MODE,
                    "sampler":      "NSGA-II",
                },
                reinit=True,
            )

        df = run_pareto_study(source_name, real_feats, run=run)

        # Save results CSV
        csv_path = OUT_DIR / f"pareto_results_{source_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Results CSV -> {csv_path}")

        if run is not None:
            run.log({f"{source_name}/results_table": wandb.Table(dataframe=df)})

        log_plots(df, source_name, run)

        if run is not None:
            run.summary[f"{source_name}/n_pareto"] = int(df["is_pareto"].sum())
            run.finish()

    print("\nDone.")
