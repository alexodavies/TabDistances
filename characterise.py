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
from scipy.stats import wasserstein_distance
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

SOURCES = {
    "CleanTabLib": Path("tables/parquet"),
    "TabICL":      Path("tables-tabicl/parquet"),
    "TabFM":       Path("tables-tabfm/parquet"),
}
COLORS = {
    "CleanTabLib": "#4C72B0",
    "TabICL":      "#DD8452",
    "TabFM":       "#55A868",
}

MAX_TABLES = 500   # cap per source for speed; set to None to use all
MAX_ROWS   = 1000  # max rows sampled per table

# Common target-column names (case-sensitive first match)
TARGET_NAMES = {"target", "label", "class", "y", "Target", "Label", "Class"}

# ── Per-table metrics ──────────────────────────────────────────────────────────

def _find_target(df: pd.DataFrame) -> str | None:
    for name in TARGET_NAMES:
        if name in df.columns:
            return name
    return None


def _missingness_structure(df: pd.DataFrame) -> float:
    """Mean absolute pairwise correlation of per-column missingness indicators."""
    miss = df.isnull()
    cols = miss.columns[miss.any()]
    if len(cols) < 2:
        return 0.0
    arr = miss[cols].corr().abs().to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, np.nan)
    return float(np.nanmean(arr))


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
    cat_cols     = df.select_dtypes(exclude="number").columns.tolist()
    feat_cols    = [c for c in numeric_cols if c != target_col]

    # ── Structural ──────────────────────────────────────────────────────────
    missingness_rate = float(df.isnull().mean().mean())
    miss_structure   = _missingness_structure(df)
    numeric_fraction = len(numeric_cols) / n_cols if n_cols else 0.0

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
        corr_matrix = df[feat_cols].corr(method="pearson").abs()
        np.fill_diagonal(corr_matrix.values, np.nan)
        mean_pairwise_corr = float(np.nanmean(corr_matrix.values))
    else:
        mean_pairwise_corr = float("nan")

    # Class balance (normalised entropy of target)
    class_balance = _normalised_entropy(df[target_col]) if target_col else float("nan")

    # Target leakage: max |Pearson r| between any numeric feature and target
    if target_col and feat_cols:
        try:
            t = pd.to_numeric(df[target_col], errors="coerce")
            corrs = [
                abs(df[c].corr(t))
                for c in feat_cols
                if df[c].notna().sum() > 10
            ]
            max_target_corr = float(max(corrs)) if corrs else float("nan")
        except Exception:
            max_target_corr = float("nan")
    else:
        max_target_corr = float("nan")

    return {
        "n_rows":               n_rows,
        "n_cols":               n_cols,
        "missingness_rate":     missingness_rate,
        "miss_structure":       miss_structure,
        "numeric_fraction":     numeric_fraction,
        "mean_cat_cardinality": mean_cat_cardinality,
        "max_cat_cardinality":  max_cat_cardinality,
        "mean_skewness":        mean_skewness,
        "mean_kurtosis":        mean_kurtosis,
        "mean_pairwise_corr":   mean_pairwise_corr,
        "class_balance":        class_balance,
        "max_target_corr":      max_target_corr,
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
        try:
            df = pd.read_parquet(f)
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42)
            m = compute_table_metrics(df)
            m["source"] = source_name
            records.append(m)
        except Exception:
            pass

    return pd.DataFrame(records)


# ── Metric metadata (display label, axis scale) ────────────────────────────────

METRIC_META: dict[str, tuple[str, str]] = {
    "n_rows":               ("Row count",                      "log"),
    "n_cols":               ("Column count",                   "linear"),
    "missingness_rate":     ("Missingness rate",               "linear"),
    "miss_structure":       ("Missingness structure",          "linear"),
    "numeric_fraction":     ("Numeric column fraction",        "linear"),
    "mean_cat_cardinality": ("Mean categorical cardinality",   "log"),
    "max_cat_cardinality":  ("Max categorical cardinality",    "log"),
    "mean_skewness":        ("Mean |skewness|",                "log"),
    "mean_kurtosis":        ("Mean |excess kurtosis|",         "log"),
    "mean_pairwise_corr":   ("Mean |pairwise Pearson corr|",  "linear"),
    "class_balance":        ("Class balance (norm. entropy)",  "linear"),
    "max_target_corr":      ("Max feature–target |corr|",     "linear"),
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
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
        return

    all_metrics = pd.concat(dfs, ignore_index=True)

    print_summary(all_metrics)
    plot_distributions(all_metrics)

    csv_path = "characterisation_metrics.csv"
    all_metrics.to_csv(csv_path, index=False)
    print(f"  Saved raw data  → {csv_path}")


if __name__ == "__main__":
    main()
