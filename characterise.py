#!/usr/bin/env python3
"""
characterise.py — Interpretable characterisation metrics across three table sources.

Computes structural and statistical metrics for CleanTabLib, TabICL, and TabFM tables,
then compares distributions across sources using summary statistics and Wasserstein-1 distances.

Structural
  - Row / column count distributions
  - Missingness rate (fraction of missing cells)
  - Missingness structure (mean pairwise correlation of per-column missingness indicators;
    high value → structured MNAR/MAR-like missingness, low → MCAR)
  - Numeric-column fraction
  - Categorical-column cardinality distribution

Statistical
  - Marginal distribution shape: mean absolute skewness and excess kurtosis per table
    (column-wise, numeric features only)
  - Mean absolute pairwise Pearson correlation across feature columns
    (measures inter-feature dependence strength)
  - Class balance: normalised entropy of target column (1 = uniform, 0 = single class)
  - Target leakage: max |Pearson r| between any numeric feature and target column
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from scipy.stats import gaussian_kde, wasserstein_distance
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

SOURCES = {
    # "CleanTabLib": Path("tables/parquet"),
    "TabICL":      Path("tables-tabicl/parquet"),
    "TabFM":       Path("tables-tabfm/parquet"),
    "T4":          Path("tables-t4/parquet"),
}
COLORS = {
    "CleanTabLib": "#4C72B0",
    "TabICL":      "#DD8452",
    "TabFM":       "#55A868",
    "T4":          "#C44E52",
}

MAX_TABLES = 4000   # cap per source for speed; set to None to use all
MAX_ROWS   = 1000  # max rows sampled per table

# Common target-column names (case-sensitive first match)
TARGET_NAMES = {"target", "label", "class", "y", "Target", "Label", "Class"}

# ── Per-table metrics ──────────────────────────────────────────────────────────

def _find_target(df: pd.DataFrame) -> str | None:
    for name in TARGET_NAMES:
        if name in df.columns:
            return name
    return None



def _normalised_entropy(series: pd.Series) -> float:
    """Entropy of value-count distribution normalised to [0, 1]."""
    counts = series.dropna().value_counts(normalize=True)
    if len(counts) < 2:
        return 0.0
    entropy    = float(-np.sum(counts * np.log(counts + 1e-12)))
    max_entropy = np.log(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_table_metrics(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    target_col     = _find_target(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [
        c for c in df.columns
        if (df[c].nunique() / max(len(df), 1)) < 0.2
    ]
    feat_cols    = [c for c in numeric_cols if c != target_col and c not in cat_cols]

    # ── Structural ──────────────────────────────────────────────────────────
    cat_fraction = len(cat_cols) / n_cols if n_cols else 0.0

    if cat_cols:
        cardinalities        = [df[c].nunique() for c in cat_cols]
        mean_cat_cardinality = float(np.mean(cardinalities))
        max_cat_cardinality  = float(np.max(cardinalities))
    else:
        mean_cat_cardinality = 0.0
        max_cat_cardinality  = 0.0

    # ── Statistical ─────────────────────────────────────────────────────────
    # Marginal distribution shape (numeric features, excluding target)
    if feat_cols:
        feat_data = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        skewnesses = feat_data.skew(skipna=True).abs()
        kurtoses   = feat_data.kurt(skipna=True).abs()
        mean_skewness = float(skewnesses.mean(skipna=True))
        mean_kurtosis = float(kurtoses.mean(skipna=True))
    else:
        mean_skewness = float("nan")
        mean_kurtosis = float("nan")

    # Mean absolute pairwise Pearson correlation (inter-feature dependence)
    if len(feat_cols) >= 2:
        corr_matrix = df[feat_cols].corr(method="pearson").abs().to_numpy().copy()
        n = corr_matrix.shape[0]
        corr_matrix[range(n), range(n)] = 1
        # np.fill_diagonal(corr_matrix, 0)
        mean_pairwise_corr = float(np.nanmean(corr_matrix))
    else:
        mean_pairwise_corr = float("nan")

    # Class balance: normalised entropy over target class distribution (1 = uniform, 0 = single class)
    class_balance = _normalised_entropy(df[target_col]) if target_col else float("nan")

    return {
        "n_rows":               n_rows,
        "n_cols":               n_cols,
        "cat_fraction":         cat_fraction,
        "mean_cat_cardinality": mean_cat_cardinality,
        "max_cat_cardinality":  max_cat_cardinality,
        "mean_skewness":        mean_skewness,
        "mean_kurtosis":        mean_kurtosis,
        "mean_pairwise_corr":   mean_pairwise_corr,
        "class_balance":        class_balance,
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def load_source_metrics(
    parquet_dir: Path,
    source_name: str,
    max_tables: int | None,
    max_rows: int,
) -> pd.DataFrame:
    files = sorted(parquet_dir.glob("*.parquet"))
    if max_tables and len(files) > max_tables:
        rng   = np.random.default_rng(42)
        files = rng.choice(files, size=max_tables, replace=False).tolist()

    records = []
    for f in tqdm(files, desc=f"  {source_name}", unit="tbl"):
        # try:
        df = pd.read_parquet(f)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
        m = compute_table_metrics(df)
        m["source"] = source_name
        records.append(m)
        # except Exception as e:
            # print(e)
            # pass

    return pd.DataFrame(records)


# ── Metric metadata (display label, axis scale) ────────────────────────────────

METRIC_META: dict[str, tuple[str, str]] = {
    "n_rows":               ("Row count",                      "log"),
    "n_cols":               ("Column count",                   "linear"),
    "cat_fraction":         ("Categorical column fraction",    "linear"),
    "mean_cat_cardinality": ("Mean categorical cardinality",   "log"),
    "max_cat_cardinality":  ("Max categorical cardinality",    "log"),
    "mean_skewness":        ("Mean |skewness|",                "log"),
    "mean_kurtosis":        ("Mean |excess kurtosis|",         "log"),
    "mean_pairwise_corr":   ("Mean |pairwise Pearson corr|",  "linear"),
    "class_balance":        ("Class balance (norm. entropy)",  "linear"),
}


# ── Reporting ──────────────────────────────────────────────────────────────────

def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return wasserstein_distance(a, b)


def print_summary(all_metrics: pd.DataFrame) -> None:
    sources      = list(SOURCES.keys())
    metric_cols  = [m for m in METRIC_META if m in all_metrics.columns]
    pairs        = [(sources[i], sources[j])
                    for i in range(len(sources))
                    for j in range(i + 1, len(sources))]

    # ── Mean ± std ──────────────────────────────────────────────────────────
    print("\n" + "═" * 100)
    print("  MEAN ± STD PER SOURCE")
    print("═" * 100)
    rows = []
    for metric in metric_cols:
        label, _ = METRIC_META[metric]
        row = {"Metric": label}
        for src in sources:
            vals = all_metrics.loc[all_metrics["source"] == src, metric].dropna()
            row[src] = f"{vals.mean():.3f} ± {vals.std():.3f}" if len(vals) else "—"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("Metric").to_string())

    # ── Wasserstein distances ────────────────────────────────────────────────
    print("\n" + "═" * 100)
    print("  WASSERSTEIN-1 DISTANCES BETWEEN SOURCE DISTRIBUTIONS")
    print("═" * 100)
    rows = []
    for metric in metric_cols:
        label, scale = METRIC_META[metric]
        row = {"Metric": label}
        for a, b in pairs:
            va = all_metrics.loc[all_metrics["source"] == a, metric].values
            vb = all_metrics.loc[all_metrics["source"] == b, metric].values
            if scale == "log":
                va, vb = np.log1p(va), np.log1p(vb)
            row[f"{a} vs {b}"] = f"{_wasserstein(va, vb):.4f}"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("Metric").to_string())
    print()


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_distributions(
    all_metrics: pd.DataFrame,
    output_path: str = "characterisation.png",
) -> None:
    metric_cols = [m for m in METRIC_META if m in all_metrics.columns]
    n          = len(metric_cols)
    ncols      = 3
    nrows      = (n + ncols - 1) // ncols
    sources    = list(SOURCES.keys())

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    fig.suptitle(
        "Table Characterisation: Distribution by Source",
        fontsize=14, fontweight="bold", y=1.01,
    )
    axes = axes.flatten()

    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        label, scale = METRIC_META[metric]

        for src in sources:
            vals = all_metrics.loc[all_metrics["source"] == src, metric].dropna().values
            if len(vals) == 0:
                continue
            plot_vals = np.log1p(vals) if scale == "log" else vals
            ax.hist(plot_vals, bins=40, alpha=0.5, label=src,
                    color=COLORS[src], density=True)

        ax.set_title(label + (" [log1p scale]" if scale == "log" else ""), fontsize=8.5)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot      → {output_path}")


# ── Pairwise scatter / density plots ───────────────────────────────────────────

SCATTER_PAIRS = [
    ("n_cols",        "cat_fraction",          "log",    "linear"),
    ("mean_skewness", "mean_kurtosis",         "log",    "log"),
    ("cat_fraction",  "mean_cat_cardinality", "linear", "log"),
]


def plot_pairwise(
    all_metrics: pd.DataFrame,
    output_path: str = "characterisation_pairwise.png",
) -> None:
    sources = list(SOURCES.keys())
    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 4.5))
    fig.suptitle(
        "Table Characterisation: Pairwise Density",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, (xk, yk, xscale, yscale) in zip(axes, SCATTER_PAIRS):
        xlabel, _ = METRIC_META[xk]
        ylabel, _ = METRIC_META[yk]

        for src in sources:
            sub = all_metrics[all_metrics["source"] == src]
            xv = sub[xk].dropna().values
            yv = sub[yk].reindex(sub[xk].dropna().index).values
            mask = ~np.isnan(xv) & ~np.isnan(yv)
            xv, yv = xv[mask], yv[mask]
            if xscale == "log":
                xv = np.log1p(xv)
            if yscale == "log":
                yv = np.log1p(yv)
            ax.scatter(xv, yv, s=6, alpha=0.25, color=COLORS[src], label=src, linewidths=0)
            if len(xv) > 10:
                try:
                    kde = gaussian_kde(np.vstack([xv, yv]))
                    xg = np.linspace(xv.min(), xv.max(), 80)
                    yg = np.linspace(yv.min(), yv.max(), 80)
                    Xg, Yg = np.meshgrid(xg, yg)
                    Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                    ax.contour(Xg, Yg, Zg, levels=4, colors=[COLORS[src]], linewidths=0.8, alpha=0.8)
                except Exception:
                    pass

        ax.set_xlabel(xlabel + (" [log1p]" if xscale == "log" else ""), fontsize=8.5)
        ax.set_ylabel(ylabel + (" [log1p]" if yscale == "log" else ""), fontsize=8.5)
        ax.set_title(f"{xlabel} vs {ylabel}", fontsize=9)
        ax.legend(fontsize=7, markerscale=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved pairwise  → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    run = wandb.init(
        entity  = "hierarchical-diffusion",
        project = "TabFMs-Characterise",
        name    = "-vs-".join(SOURCES.keys()),
        config  = {"max_tables": MAX_TABLES, "max_rows": MAX_ROWS,
                   "sources": list(SOURCES.keys())},
    )

    dfs = []
    for source_name, parquet_dir in SOURCES.items():
        if not parquet_dir.exists():
            print(f"[WARN] {parquet_dir} not found — skipping {source_name}")
            continue
        print(f"\nLoading {source_name} …")
        df = load_source_metrics(parquet_dir, source_name, MAX_TABLES, MAX_ROWS)
        print(f"  → {len(df)} tables")
        dfs.append(df)

    if not dfs:
        print("No data loaded. Exiting.")
        run.finish()
        return

    all_metrics = pd.concat(dfs, ignore_index=True)

    print_summary(all_metrics)

    dist_path    = "characterisation.png"
    pairwise_path = "characterisation_pairwise.png"
    plot_distributions(all_metrics, output_path=dist_path)
    plot_pairwise(all_metrics, output_path=pairwise_path)

    csv_path = "characterisation_metrics.csv"
    all_metrics.to_csv(csv_path, index=False)
    print(f"  Saved raw data  → {csv_path}")

    # ── wandb logging ────────────────────────────────────────────────────────
    sources      = list(SOURCES.keys())
    metric_cols  = [m for m in METRIC_META if m in all_metrics.columns]
    log_dict: dict = {}
    for metric in metric_cols:
        label, _ = METRIC_META[metric]
        for src in sources:
            vals = all_metrics.loc[all_metrics["source"] == src, metric].dropna()
            if len(vals):
                log_dict[f"{src}/{label}/mean"] = float(vals.mean())
                log_dict[f"{src}/{label}/std"]  = float(vals.std())

    log_dict["distributions"] = wandb.Image(dist_path)
    log_dict["pairwise"]      = wandb.Image(pairwise_path)
    log_dict["metrics_table"] = wandb.Table(dataframe=all_metrics)
    run.log(log_dict)
    run.finish()


if __name__ == "__main__":
    main()
