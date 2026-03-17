"""Multi-objective Pareto-front analysis of TabICL prior parameters.

Runs N_RESTARTS_PARETO independent NSGA-II studies (merged afterward) and
N_RESTARTS_SINGLE independent single-objective TPE studies.  Results from
every restart are appended to disk immediately so the script can be
interrupted safely.

Objectives:
  coverage_real_by_synth  (recall)    — how much of real is spanned by synth
  coverage_synth_by_real  (precision) — how realistic/focused the synth is

Single-objective also tracks:
  combined_score = Euclidean quality score in [0, 1] (maximise)

Usage
-----
    python pareto-coverage.py

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
from scipy import stats
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

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

N_EVAL          = 500    # synthetic tables generated per evaluation
N_REAL_CACHE    = 10000  # real tables to pre-featurize and cache
MAX_ROWS        = 512
BATCH_SIZE      = 1
RANDOM_STATE    = 42
N_TRIALS        = 1000   # trials per restart (NSGA-II and TPE)
N_RESTARTS_PARETO  = 100   # independent NSGA-II restarts (merged for global Pareto front)
N_RESTARTS_SINGLE  = 100   # independent single-objective TPE restarts
FEATURE_MODE    = "full"
CACHE_DIR       = Path("feature-cache")
OUT_DIR         = Path("pareto-results")
WANDB_PROJECT   = "tabicl-coverage-pareto"

# Coverage k-NN settings
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
# Coverage + discriminator helpers
# ------------------------------------------------------------------

def _knn_coverage(ref_m: np.ndarray, cand_m: np.ndarray) -> float:
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


def compute_discriminator_auc(real_m: np.ndarray, synth_m: np.ndarray, seed: int) -> float:
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


def evaluate(
    params: dict,
    real_feats: np.ndarray,
    seed: int,
) -> tuple:
    """Return (fwd_cov, rev_cov, auc).  On generation failure returns (0, 0, 0.5)."""
    rng = np.random.default_rng(seed)
    all_col_counts = real_feats[:, 0].astype(int)
    target_col_counts = rng.choice(all_col_counts, size=N_EVAL, replace=True).tolist()

    try:
        synth_dfs = generate_synthetic_dataframes(params, target_col_counts)
    except Exception as exc:
        print(f"    [generate error] {exc}")
        return 0.0, 0.0, 0.5

    synth_feats = np.array([df_to_vec(df) for df in synth_dfs])
    idxs = rng.choice(len(real_feats), size=min(N_EVAL, len(real_feats)), replace=False)
    real_sample = real_feats[idxs]

    real_m,  _ = select_features(real_sample, FEATURE_MODE)
    synth_m, _ = select_features(synth_feats, FEATURE_MODE)

    fwd_cov = _knn_coverage(real_m,  synth_m)
    rev_cov = _knn_coverage(synth_m, real_m)
    auc     = compute_discriminator_auc(real_m, synth_m, seed)
    return fwd_cov, rev_cov, auc


# ------------------------------------------------------------------
# Pareto helper
# ------------------------------------------------------------------

def _pareto_mask(fwd: np.ndarray, rev: np.ndarray) -> np.ndarray:
    """Boolean mask; True where (fwd[i], rev[i]) is Pareto-optimal (maximise both)."""
    n = len(fwd)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        dominated[i] = np.any(
            (fwd >= fwd[i]) & (rev >= rev[i]) & ((fwd > fwd[i]) | (rev > rev[i]))
        )
    return ~dominated


# ------------------------------------------------------------------
# Single-objective study (combined_score, multi-restart)
# ------------------------------------------------------------------

def run_single_restart(
    source_name: str,
    real_feats: np.ndarray,
    restart_idx: int,
    out_csv: Path,
    run=None,
) -> tuple:
    """One TPE restart maximising combined_score."""
    seed = RANDOM_STATE + restart_idx * 1000
    running_sum   = [0.0]
    running_count = [0]
    _live_points: list = []

    def objective(trial: optuna.Trial) -> float:
        rm = running_sum[0] / running_count[0] if running_count[0] > 0 else 0.0
        params = make_params(trial)
        trial_seed = seed + trial.number
        fwd_cov, rev_cov, auc = evaluate(params, real_feats, trial_seed)
        score = combined_score(fwd_cov, rev_cov, auc)

        running_sum[0]   += score
        running_count[0] += 1
        _live_points.append([fwd_cov, rev_cov, auc])

        cov_col_ab = f"Coverage {source_name}->tabicl"
        cov_col_ba = f"Coverage tabicl->{source_name}"
        print(f"  [single r{restart_idx+1:03d}|t{trial.number:03d}] "
              f"score={score:.4f}  {cov_col_ab}={fwd_cov:.4f}  {cov_col_ba}={rev_cov:.4f}  auc={auc:.4f}")

        row = {
            "restart": restart_idx, "trial": trial.number,
            "A": source_name, "B": "tabicl",
            "cov_A_to_B": fwd_cov, "cov_B_to_A": rev_cov,
            "discriminator_auc": auc, "score": score,
            **{k: (1 if v is True else 0 if v is False else v) for k, v in params.items()},
        }
        pd.DataFrame([row]).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False
        )

        if run is not None:
            scatter_table = wandb.Table(
                data=_live_points, columns=[cov_col_ab, cov_col_ba, "AUC"]
            )
            run.log({
                "restart":                          restart_idx,
                "A":                                source_name,
                "B":                                "tabicl",
                f"{source_name}_single/score":      score,
                f"{source_name}_single/discriminator_auc": auc,
                cov_col_ab:                         fwd_cov,
                cov_col_ba:                         rev_cov,
                f"{source_name}_single/coverage_scatter": wandb.plot.scatter(
                    scatter_table, cov_col_ab, cov_col_ba,
                    title=f"Single-obj — {source_name}",
                ),
            })
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=N_TRIALS)
    best_trial  = study.best_trial
    best_params = best_trial.user_attrs["params"]
    print(f"  Single restart {restart_idx+1} done: best_score={best_trial.value:.4f}")
    return best_params, best_trial.value


def run_single_objective_study(
    source_name: str,
    real_feats: np.ndarray,
    run=None,
) -> pd.DataFrame:
    """N_RESTARTS_SINGLE independent TPE studies.  Returns full trial DataFrame."""
    print(f"\n{'#'*60}")
    print(f"  Single-objective (combined_score) study: tabicl vs {source_name}")
    print(f"  {N_RESTARTS_SINGLE} restarts × {N_TRIALS} trials")
    print(f"{'#'*60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"single_results_{source_name}.csv"
    if out_csv.exists():
        out_csv.unlink()
    print(f"  Live results -> {out_csv}")

    all_results = []
    for r in range(N_RESTARTS_SINGLE):
        print(f"\n  -- Single restart {r+1}/{N_RESTARTS_SINGLE} --")
        best_params, best_score = run_single_restart(
            source_name, real_feats, r, out_csv, run=run
        )
        all_results.append((best_params, best_score))
        if run is not None:
            run.log({f"{source_name}_single/restart_best": best_score,
                     f"{source_name}_single/restart": r})

    best_r, (_, best_score) = max(enumerate(all_results), key=lambda x: x[1][1])
    print(f"\n  Single-obj global best: restart={best_r+1}  score={best_score:.4f}")
    if run is not None:
        run.summary[f"{source_name}_single/best_score"]   = best_score
        run.summary[f"{source_name}_single/best_restart"] = best_r + 1

    return pd.read_csv(out_csv)


# ------------------------------------------------------------------
# Multi-objective Pareto study (multi-restart NSGA-II)
# ------------------------------------------------------------------

def run_pareto_restart(
    source_name: str,
    real_feats: np.ndarray,
    restart_idx: int,
    out_csv: Path,
    run=None,
) -> pd.DataFrame:
    """One NSGA-II restart.  Appends all trials to *out_csv* immediately."""
    seed = RANDOM_STATE + restart_idx * 1000
    _live_points: list = []

    def objective(trial: optuna.Trial) -> tuple:
        params = make_params(trial)
        trial_seed = seed + trial.number
        fwd_cov, rev_cov, auc = evaluate(params, real_feats, trial_seed)
        score = combined_score(fwd_cov, rev_cov, auc)
        _live_points.append([fwd_cov, rev_cov, auc])

        cov_col_ab = f"Coverage {source_name}->tabicl"
        cov_col_ba = f"Coverage tabicl->{source_name}"
        print(f"  [pareto r{restart_idx+1:03d}|t{trial.number:03d}] "
              f"{cov_col_ab}={fwd_cov:.4f}  {cov_col_ba}={rev_cov:.4f}  auc={auc:.4f}  score={score:.4f}")

        row = {
            "restart": restart_idx, "trial": trial.number,
            "A": source_name, "B": "tabicl",
            "cov_A_to_B": fwd_cov, "cov_B_to_A": rev_cov,
            "discriminator_auc": auc, "score": score,
            **{k: (1 if v is True else 0 if v is False else v) for k, v in params.items()},
        }
        pd.DataFrame([row]).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False
        )

        if run is not None:
            scatter_table = wandb.Table(
                data=_live_points, columns=[cov_col_ab, cov_col_ba, "AUC"]
            )
            run.log({
                "restart":                      restart_idx,
                "A":                            source_name,
                "B":                            "tabicl",
                f"{source_name}/score":         score,
                f"{source_name}/discriminator_auc": auc,
                cov_col_ab:                     fwd_cov,
                cov_col_ba:                     rev_cov,
                f"{source_name}/coverage_scatter": wandb.plot.scatter(
                    scatter_table, cov_col_ab, cov_col_ba,
                    title=f"Pareto — {source_name}",
                ),
            })
        return fwd_cov, rev_cov

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
    )
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"  Pareto restart {restart_idx+1} done.")
    # Return DataFrame for this restart (read back from CSV for consistency)
    return pd.read_csv(out_csv).query(f"restart == {restart_idx}")


def run_pareto_study(
    source_name: str,
    real_feats: np.ndarray,
    run=None,
) -> pd.DataFrame:
    """N_RESTARTS_PARETO independent NSGA-II studies; merge and compute global Pareto front."""
    print(f"\n{'#'*60}")
    print(f"  Pareto study: tabicl vs {source_name}")
    print(f"  {N_RESTARTS_PARETO} restarts × {N_TRIALS} trials")
    print(f"{'#'*60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"pareto_results_{source_name}.csv"
    if out_csv.exists():
        out_csv.unlink()
    print(f"  Live results -> {out_csv}")

    for r in range(N_RESTARTS_PARETO):
        print(f"\n  ===== Pareto restart {r+1}/{N_RESTARTS_PARETO} =====")
        run_pareto_restart(source_name, real_feats, r, out_csv, run=run)

    # Load full results and compute global Pareto front
    df = pd.read_csv(out_csv)
    df["is_pareto"] = _pareto_mask(df["cov_A_to_B"].values, df["cov_B_to_A"].values)

    front = df[df["is_pareto"]].sort_values("cov_A_to_B", ascending=False)
    print(f"\n  Global Pareto front ({len(front)} points) across {N_RESTARTS_PARETO} restarts:")
    print(front[["restart", "trial", "A", "B", "cov_A_to_B", "cov_B_to_A",
                 "discriminator_auc", "score"]].to_string(index=False))
    return df


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def _fig_pareto_scatter(df: pd.DataFrame, source_name: str, df_single: pd.DataFrame | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    fwd    = df["cov_A_to_B"].values
    rev    = df["cov_B_to_A"].values
    pareto = df["is_pareto"].values
    auc    = df["discriminator_auc"].values if "discriminator_auc" in df.columns else np.full(len(df), 0.75)
    xlabel = f"Coverage {source_name}->tabicl  (recall)"
    ylabel = f"Coverage tabicl->{source_name}  (precision)"

    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    cmap = cm.RdYlGn_r

    ax.scatter(fwd[~pareto], rev[~pareto], c=auc[~pareto], cmap=cmap, norm=norm,
               s=10, alpha=0.3, zorder=2, label=f"NSGA-II ({(~pareto).sum()} trials)")
    ax.scatter(fwd[pareto], rev[pareto], c=auc[pareto], cmap=cmap, norm=norm,
               s=80, zorder=5, marker="*", edgecolors="k", linewidths=0.4,
               label=f"Global Pareto front ({pareto.sum()} pts)")
    pf = df[pareto].sort_values("cov_A_to_B")
    ax.step(pf["cov_A_to_B"].values, pf["cov_B_to_A"].values,
            where="post", color="k", linewidth=1.4, zorder=4, alpha=0.5)

    if df_single is not None and len(df_single):
        auc_s = (df_single["discriminator_auc"].values
                 if "discriminator_auc" in df_single.columns
                 else np.full(len(df_single), 0.75))
        ax.scatter(df_single["cov_A_to_B"].values, df_single["cov_B_to_A"].values,
                   c=auc_s, cmap=cmap, norm=norm, s=15, alpha=0.4, zorder=3, marker="^",
                   edgecolors="k", linewidths=0.2,
                   label=f"Single-obj TPE ({len(df_single)} trials)")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="discriminator AUC  (0.5 = indistinguishable)", shrink=0.85)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, alpha=0.5, label="y = x")
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.1)
    ax.set_xticks(np.arange(0, 1.11, 0.1)); ax.set_yticks(np.arange(0, 1.11, 0.1))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"Global Pareto front — {source_name}  ({N_RESTARTS_PARETO} NSGA-II restarts)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def _fig_pareto_correlation(df: pd.DataFrame, source_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    fwd = df["cov_A_to_B"].values
    rev = df["cov_B_to_A"].values
    ax.scatter(fwd, rev, alpha=0.2, s=8, color="#4C72B0")
    slope, intercept, r, p, _ = stats.linregress(fwd, rev)
    xs = np.linspace(fwd.min(), fwd.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="red", linewidth=1.5,
            label=f"r = {r:.3f}  (p={p:.2e})")
    ax.set_xlabel(f"Coverage {source_name}->tabicl  (recall)")
    ax.set_ylabel(f"Coverage tabicl->{source_name}  (precision)")
    ax.set_title(f"Coverage correlation — {source_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def log_plots(df: pd.DataFrame, source_name: str, run, df_single: pd.DataFrame | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots = {
        "pareto_scatter":     _fig_pareto_scatter(df, source_name, df_single=df_single),
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
                    "source":              source_name,
                    "n_eval":              N_EVAL,
                    "n_real":              len(real_feats),
                    "n_trials":            N_TRIALS,
                    "n_restarts_pareto":   N_RESTARTS_PARETO,
                    "n_restarts_single":   N_RESTARTS_SINGLE,
                    "feature_mode":        FEATURE_MODE,
                    "sampler_multi":       "NSGA-II",
                    "sampler_single":      "TPE",
                    "score_formula":       "sqrt(fwd^2 + rev^2 + ((1-auc)*2)^2) / sqrt(3)",
                },
                reinit=True,
            )

        df_single = run_single_objective_study(source_name, real_feats, run=run)
        df        = run_pareto_study(source_name, real_feats, run=run)

        if run is not None:
            rename = {"cov_A_to_B": f"Coverage {source_name}->tabicl",
                      "cov_B_to_A": f"Coverage tabicl->{source_name}"}
            run.log({
                f"{source_name}/results_table":        wandb.Table(dataframe=df.rename(columns=rename)),
                f"{source_name}_single/results_table": wandb.Table(dataframe=df_single.rename(columns=rename)),
            })

        log_plots(df, source_name, run, df_single=df_single)

        if run is not None:
            run.summary[f"{source_name}/n_pareto"] = int(df["is_pareto"].sum())
            run.finish()

    print("\nDone.")
