"""Merge shard results from a grid-isambard.py array job and log combined figures.

Usage
-----
    python grid-combine.py <results-dir>
    python grid-combine.py grid-results-35200/

Globs <results-dir>/shard_*/grid_results_<source>_*.csv, concatenates all
shards per source, logs combined plots to wandb, and prints the top-10 results.
"""

import sys
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

WANDB_PROJECT = "tabicl-grid-search"

# Must match PARAM_SPACE in grid-isambard.py
PARAM_SPACE = {
    "max_classes":             ("int",   2,    20),
    "batch_size_per_gp":       ("int",   1,    32),
    "prior_type":              ("cat",   ["mlp_scm", "tree_scm", "mix_scm"]),
    "tree_model":              ("cat",   ["decision_tree", "extra_trees",
                                          "random_forest", "xgboost"]),
    "mix_prob_mlp":            ("float", 0.0,  1.0),
    "cat_prob":                ("float", 0.0,  0.6),
    "multiclass_ordered_prob": ("float", 0.0,  1.0),
    "balanced":                ("cat",   [False, True]),
    "replay_small":            ("cat",   [False, True]),
}

_CAT_PARAMS   = [k for k, s in PARAM_SPACE.items() if s[0] == "cat"]
_FLOAT_PARAMS = [k for k, s in PARAM_SPACE.items() if s[0] in ("float", "int")]


# ------------------------------------------------------------------
# Collect
# ------------------------------------------------------------------

def collect_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Find all shard CSVs, group by source name, concatenate."""
    csvs = sorted(results_dir.glob("shard_*/grid_results_*.csv"))
    if not csvs:
        print(f"No CSV files found under {results_dir}/shard_*/", flush=True)
        sys.exit(1)

    by_source: dict[str, list[pd.DataFrame]] = {}
    for csv in csvs:
        # filename pattern: grid_results_<source>_<job_id>.csv
        m = re.match(r"grid_results_(.+?)_\d+\.csv", csv.name)
        source = m.group(1) if m else csv.stem
        by_source.setdefault(source, []).append(pd.read_csv(csv))

    combined = {}
    for source, dfs in by_source.items():
        df = pd.concat(dfs, ignore_index=True)
        combined[source] = df
        print(f"  {source}: {len(df)} rows from {len(dfs)} shards", flush=True)

    return combined


# ------------------------------------------------------------------
# Plots (same logic as grid-isambard.py)
# ------------------------------------------------------------------

def _fig_auc_histogram(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["auc"], bins=30, edgecolor="white", color="#4C72B0")
    ax.axvline(df["auc"].min(),  color="red",    linestyle="--", label=f"best={df['auc'].min():.4f}")
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
        if param not in df.columns:
            continue
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
        if param not in df.columns:
            continue
        ax.scatter(df[param], df["auc"], alpha=0.3, s=10, color="#55A868")
        z  = np.polyfit(df[param].astype(float), df["auc"], 1)
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
    top = df.nsmallest(top_n, "auc").copy()

    plot_df = pd.DataFrame()
    for col in list(PARAM_SPACE.keys()) + ["auc"]:
        if col not in top.columns:
            continue
        vals = top[col]
        if not pd.api.types.is_numeric_dtype(vals):
            uniq = sorted(vals.unique(), key=str)
            plot_df[col] = vals.map({v: i for i, v in enumerate(uniq)})
        else:
            plot_df[col] = vals.astype(float)

    normed = plot_df.copy()
    for col in normed.columns:
        rng = normed[col].max() - normed[col].min()
        if rng > 0:
            normed[col] = (normed[col] - normed[col].min()) / rng

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


def log_summary_plots(df: pd.DataFrame, source: str, out_dir: Path, run) -> None:
    plots = {
        "auc_histogram":       _fig_auc_histogram(df),
        "cat_param_means":     _fig_cat_param_means(df),
        "float_param_scatter": _fig_float_param_scatter(df),
        "parallel_coords":     _fig_parallel_coords(df),
    }
    for name, fig in plots.items():
        path = out_dir / f"combined_plot_{source}_{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  Plot saved -> {path}", flush=True)
        if run is not None:
            run.log({f"{source}/{name}": wandb.Image(str(path))})
        plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grid-combine.py <results-dir>", flush=True)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}", flush=True)
        sys.exit(1)

    job_label = results_dir.name   # e.g. "grid-results-35200"

    print(f"Results dir : {results_dir}", flush=True)
    print(f"Job label   : {job_label}", flush=True)

    combined = collect_results(results_dir)

    for source, df in combined.items():
        print(f"\n{'#'*60}", flush=True)
        print(f"  Combined results: tabicl vs {source}  ({len(df)} trials)", flush=True)
        print(f"{'#'*60}", flush=True)

        # Save merged CSV
        out_csv = results_dir / f"combined_results_{source}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Merged CSV -> {out_csv}", flush=True)

        run = None
        if _WANDB_AVAILABLE and WANDB_PROJECT:
            run = wandb.init(
                project=WANDB_PROJECT,
                name=f"combined_{source}_{job_label}",
                config={"source": source, "n_trials": len(df), "job_label": job_label},
                reinit=True,
            )
            table = wandb.Table(dataframe=df)
            run.log({f"{source}/results_table": table})

        log_summary_plots(df, source, results_dir, run)

        if run is not None:
            run.finish()

        top10 = df.nsmallest(10, "auc")
        print(f"\n  TOP-10  [{source}]  (lower AUC = more realistic)", flush=True)
        print(top10.to_string(index=False), flush=True)

    print(f"\nDone.", flush=True)
