"""Partial correlation between k-NN distance and TabICL performance after
controlling for dataset-level confounders.

Motivation
----------
Dataset size, dimensionality, and class structure plausibly confound both
k-NN distance and model performance. Large tables may sit far from the prior
simply because the prior generates small tables; high-dimensional tables may
be harder for TabICL for unrelated reasons (context length limits). Partialling
out these covariates tests whether the distance–performance relationship is
real after removing trivial covariate effects.

Covariates
----------
Two analyses are reported, distinguished by covariate set:

  "universal" (all datasets):
    log_n_rows, log_n_cols, prop_categorical

  "full" (classification datasets only; regression rows are dropped because
  n_classes and class_imbalance are NaN for regression):
    log_n_rows, log_n_cols, n_classes, class_imbalance, prop_categorical

Method
------
For each (representation, performance metric) pair:
  1. Regress performance ~ covariates (OLS), take residuals e_perf.
  2. Regress nn_dist  ~ covariates (OLS), take residuals e_dist.
  3. Pearson r(e_perf, e_dist) with two-tailed p-value.
Only rows with all covariates and both outcomes finite are used.

Output
------
  results/partial-correlation/partial_corr_<metric>.csv
    One row per representation; columns: N, partial_r, partial_p.
  (Also printed to stdout.)

Caches required (same as coverage_correlation_summary.py):
  cache/talent_feats_512.npz
  cache/real_feats_icl_25000_512_v2.npz
  cache/talent_col_hists.npz
  cache/icl_col_hists_<n>.npz
  cache/talent_corr_hists.npz
  cache/tabicl_corr_hists.npz
  cache/talent_tabicl_emb_colrow.npz
  cache/icl_tabicl_emb_25000_colrow.npz

Usage
-----
    python benchmark/partial_correlation_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

from tabdistances.analysis.distances import knn_coverage

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_ROOT   = Path(__file__).resolve().parent.parent
CACHE   = _ROOT / "cache"
RESULTS = _ROOT / "results"
DATA    = _ROOT / "data"

RANKINGS_CSV       = _ROOT / "results/tabicl-rankings-full-context.csv"

TALENT_FEATS_CACHE = CACHE / "talent_feats_512.npz"
ICL_FEATS_CACHE    = CACHE / "real_feats_icl_25000_512_v2.npz"
TALENT_COL_CACHE   = CACHE / "talent_col_hists.npz"
TALENT_CORR_CACHE  = CACHE / "talent_corr_hists.npz"
TABICL_CORR_CACHE  = CACHE / "tabicl_corr_hists.npz"
TALENT_EMB_CACHE   = CACHE / "talent_tabicl_emb_colrow.npz"
ICL_EMB_CACHE      = CACHE / "icl_tabicl_emb_25000_colrow.npz"

KNN_K        = 5
RANDOM_STATE = 42


# ------------------------------------------------------------------
# k-NN distance helper
# ------------------------------------------------------------------

def _nn_distances(query: np.ndarray, ref: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Mean k-NN distance from each query point to ref (Min-Max scaled)."""
    query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
    ref   = np.nan_to_num(ref,   nan=0.0, posinf=0.0, neginf=0.0)
    scaler = MinMaxScaler().fit(ref)
    q = scaler.transform(query)
    r = scaler.transform(ref)
    tree = BallTree(r, metric="euclidean")
    dists, _ = tree.query(q, k=min(k, len(r)))
    return dists.mean(axis=1)


# ------------------------------------------------------------------
# Dataset-level covariates
# ------------------------------------------------------------------

def _load_covariates(datasets: list[str]) -> pd.DataFrame:
    """Extract dataset-level covariates from info.json and y_train.npy.

    Returns a DataFrame indexed by dataset name with columns:
        log_n_rows, log_n_cols, n_classes, class_imbalance, prop_categorical
    n_classes and class_imbalance are NaN for regression datasets.
    """
    rows = []
    for ds in datasets:
        ds_dir = DATA / ds
        info_path = ds_dir / "info.json"
        if not info_path.exists():
            rows.append({"dataset": ds})
            continue

        with open(info_path) as f:
            info = json.load(f)

        n_num = int(info.get("n_num_features", 0))
        n_cat = int(info.get("n_cat_features", 0))
        n_cols = n_num + n_cat
        n_rows = (
            int(info.get("train_size", 0))
            + int(info.get("val_size", 0))
            + int(info.get("test_size", 0))
        )
        task_type = info.get("task_type", "")
        is_cls = task_type in ("binclass", "multiclass", "classification",
                               "binary", "multiclass")

        log_n_rows = np.log(n_rows) if n_rows > 0 else np.nan
        log_n_cols = np.log(n_cols) if n_cols > 0 else np.nan
        prop_cat   = n_cat / n_cols if n_cols > 0 else np.nan

        # n_classes from info.json (1 means regression)
        n_classes_raw = int(info.get("n_classes", 1))
        n_classes      = float(n_classes_raw) if is_cls else np.nan

        # Class imbalance: load y_train and compute majority-class proportion
        class_imbalance = np.nan
        y_path = ds_dir / "y_train.npy"
        if is_cls and y_path.exists():
            try:
                y = np.load(y_path)
                if len(y) > 0:
                    vals, counts = np.unique(y, return_counts=True)
                    class_imbalance = float(counts.max() / counts.sum())
            except Exception:
                pass

        rows.append({
            "dataset":         ds,
            "log_n_rows":      log_n_rows,
            "log_n_cols":      log_n_cols,
            "n_classes":       n_classes,
            "class_imbalance": class_imbalance,
            "prop_categorical": prop_cat,
        })

    return pd.DataFrame(rows).set_index("dataset")


# ------------------------------------------------------------------
# Rankings: TALENT
# ------------------------------------------------------------------

def _talent_rankings() -> tuple[dict, dict, dict, list[str]]:
    """Returns (rank_map, rel_auc_map, rel_best_auc_map, datasets)."""
    df = pd.read_csv(RANKINGS_CSV)
    df["AUC"] = df["AUC"].astype(float)
    tc = df[df["model"] == "tabicl"][["dataset", "rank", "AUC"]].copy()
    tc["rank"] = tc["rank"].astype(int)
    df_no_tc = df[df["model"] != "tabicl"]
    mean_auc = df_no_tc.groupby("dataset")["AUC"].mean()
    best_auc = df_no_tc.groupby("dataset")["AUC"].max()
    rank_map = dict(zip(tc["dataset"], tc["rank"]))
    rel_auc_map = {
        ds: auc / mean_auc[ds]
        for ds, auc in zip(tc["dataset"], tc["AUC"])
        if ds in mean_auc and mean_auc[ds] > 0
    }
    rel_best_auc_map = {
        ds: auc / best_auc[ds]
        for ds, auc in zip(tc["dataset"], tc["AUC"])
        if ds in best_auc and best_auc[ds] > 0
    }
    return rank_map, rel_auc_map, rel_best_auc_map, tc["dataset"].tolist()


# ------------------------------------------------------------------
# Partial correlation
# ------------------------------------------------------------------

# Universal covariates (available for all datasets including regression)
COVARIATE_COLS_UNIVERSAL = ["log_n_rows", "log_n_cols", "prop_categorical"]

# Full covariate set (drops regression datasets where n_classes / class_imbalance are NaN)
COVARIATE_COLS_FULL = ["log_n_rows", "log_n_cols", "n_classes",
                       "class_imbalance", "prop_categorical"]


def _partial_corr(
    names: list[str],
    nn_dists: np.ndarray,
    perf: np.ndarray,
    covariates: pd.DataFrame,
    covariate_cols: list[str],
) -> tuple[float, float, int]:
    """Partial Pearson r between nn_dists and perf, partialling out covariates.

    Returns (r, p, n) where n is the number of complete cases used.
    Only rows where dist, perf, and all specified covariate_cols are finite
    are included.
    """
    df = pd.DataFrame({
        "dist": nn_dists,
        "perf": perf,
        "name": names,
    }).set_index("name")

    # Join covariates
    df = df.join(covariates[covariate_cols], how="left")

    # Keep only rows where dist, perf, and all covariates are finite
    mask = np.isfinite(df["dist"]) & np.isfinite(df["perf"])
    for col in covariate_cols:
        mask &= np.isfinite(df[col])
    df = df[mask]

    n = len(df)
    if n < 5:
        return float("nan"), float("nan"), n

    X = df[covariate_cols].values.astype(float)
    y_dist = df["dist"].values.astype(float)
    y_perf = df["perf"].values.astype(float)

    # Residualise both variables against covariates
    reg = LinearRegression()
    e_dist = y_dist - reg.fit(X, y_dist).predict(X)
    e_perf = y_perf - reg.fit(X, y_perf).predict(X)

    r, p = stats.pearsonr(e_dist, e_perf)
    return float(r), float(p), n


# ------------------------------------------------------------------
# Per-representation loaders (return names + arrays)
# ------------------------------------------------------------------

def _load_table_feats(
    rank_map: dict,
    rel_auc_map: dict,
    rel_best_auc_map: dict,
    datasets: list[str],
    rng: np.random.Generator,
) -> dict | None:
    for path in (TALENT_FEATS_CACHE, ICL_FEATS_CACHE):
        if not path.exists():
            print(f"  [skip table feats] cache missing: {path}")
            return None

    data = np.load(TALENT_FEATS_CACHE, allow_pickle=True)
    names_arr = data["names"].tolist()
    feats_arr = data["feats"]
    cached = {n: feats_arr[i] for i, n in enumerate(names_arr)}

    icl_all = np.load(ICL_FEATS_CACHE)["feats"]
    n_icl = min(25_000, len(icl_all))
    idx = rng.choice(len(icl_all), size=n_icl, replace=False)
    icl_feats = icl_all[idx]

    names    = [ds for ds in datasets if ds in cached and ds in rank_map]
    feats    = np.stack([cached[ds] for ds in names])
    nn_dists = _nn_distances(feats, icl_feats)
    ranks    = np.array([rank_map[n]                    for n in names], float)
    rel_auc  = np.array([rel_auc_map.get(n, np.nan)     for n in names], float)
    rel_best = np.array([rel_best_auc_map.get(n, np.nan) for n in names], float)

    return {"names": names, "nn_dists": nn_dists,
            "ranks": ranks, "rel_auc": rel_auc, "rel_best": rel_best}


def _load_col_hists(
    rank_map: dict,
    rel_auc_map: dict,
    rel_best_auc_map: dict,
    rng: np.random.Generator,
) -> dict | None:
    if not TALENT_COL_CACHE.exists():
        print(f"  [skip col hists] cache missing: {TALENT_COL_CACHE}")
        return None

    cdata = np.load(TALENT_COL_CACHE, allow_pickle=True)
    talent_hists = cdata["hists"]
    col_datasets = cdata["col_datasets"]

    n_talent = len(talent_hists)
    icl_col_cache = CACHE / f"icl_col_hists_{n_talent}.npz"
    if not icl_col_cache.exists():
        print(f"  [skip col hists] ICL cache missing: {icl_col_cache}")
        return None

    icl_all = np.load(icl_col_cache)["hists"]
    idx = rng.choice(len(icl_all), size=min(n_talent, len(icl_all)), replace=False)
    icl_hists = icl_all[idx]

    mask = np.array([ds in rank_map for ds in col_datasets])
    hists_m    = talent_hists[mask]
    matched_ds = col_datasets[mask]

    nn_dists_col = _nn_distances(hists_m, icl_hists)

    df_tmp = pd.DataFrame({"ds": matched_ds, "dist": nn_dists_col})
    ds_mean = df_tmp.groupby("ds")["dist"].mean()

    names    = ds_mean.index.tolist()
    nn_dists = ds_mean.values
    ranks    = np.array([rank_map[n]                    for n in names], float)
    rel_auc  = np.array([rel_auc_map.get(n, np.nan)     for n in names], float)
    rel_best = np.array([rel_best_auc_map.get(n, np.nan) for n in names], float)

    return {"names": names, "nn_dists": nn_dists,
            "ranks": ranks, "rel_auc": rel_auc, "rel_best": rel_best}


def _load_corr_hists(
    rank_map: dict,
    rel_auc_map: dict,
    rel_best_auc_map: dict,
    rng: np.random.Generator,
) -> dict | None:
    for path in (TALENT_CORR_CACHE, TABICL_CORR_CACHE):
        if not path.exists():
            print(f"  [skip corr hists] cache missing: {path}")
            return None

    bd = np.load(TALENT_CORR_CACHE, allow_pickle=True)
    bench_hists = bd["hists"]
    bench_names = bd["names"].tolist()

    icl_all = np.load(TABICL_CORR_CACHE)["hists"]
    idx = rng.choice(len(icl_all), size=min(25_000, len(icl_all)), replace=False)
    icl_hists = icl_all[idx]

    mask    = np.array([n in rank_map for n in bench_names])
    hists_m = bench_hists[mask]
    names   = list(np.array(bench_names)[mask])

    nn_dists = _nn_distances(hists_m, icl_hists)
    ranks    = np.array([rank_map[n]                    for n in names], float)
    rel_auc  = np.array([rel_auc_map.get(n, np.nan)     for n in names], float)
    rel_best = np.array([rel_best_auc_map.get(n, np.nan) for n in names], float)

    return {"names": names, "nn_dists": nn_dists,
            "ranks": ranks, "rel_auc": rel_auc, "rel_best": rel_best}


def _load_tabicl_embs(
    rank_map: dict,
    rel_auc_map: dict,
    rel_best_auc_map: dict,
    datasets: list[str],
    rng: np.random.Generator,
    emb_type: str,
) -> dict | None:
    for path in (TALENT_EMB_CACHE, ICL_EMB_CACHE):
        if not path.exists():
            print(f"  [skip tabicl emb ({emb_type})] cache missing: {path}")
            return None

    bd = np.load(TALENT_EMB_CACHE, allow_pickle=True)
    if "feats_col" not in bd or len(bd["feats_col"]) == 0:
        print(f"  [skip tabicl emb ({emb_type})] cache empty")
        return None

    bench_names = bd["names"].tolist()
    bench_col   = bd["feats_col"]
    bench_row   = bd["feats_row"]

    id_ = np.load(ICL_EMB_CACHE, allow_pickle=True)
    if "feats_col" not in id_ or len(id_["feats_col"]) == 0:
        print(f"  [skip tabicl emb ({emb_type})] ICL cache empty")
        return None
    icl_col = id_["feats_col"]
    icl_row = id_["feats_row"]

    n_icl = min(25_000, len(icl_col))
    idx = rng.choice(len(icl_col), size=n_icl, replace=False)
    icl_col = icl_col[idx]
    icl_row = icl_row[idx]

    bench_combined = np.concatenate([bench_col, bench_row], axis=1)
    icl_combined   = np.concatenate([icl_col,   icl_row],   axis=1)

    if emb_type == "col":
        bench_feats, icl_feats = bench_col, icl_col
    elif emb_type == "row":
        bench_feats, icl_feats = bench_row, icl_row
    else:
        bench_feats, icl_feats = bench_combined, icl_combined

    cache_name_set = set(bench_names)
    names = [ds for ds in datasets if ds in cache_name_set and ds in rank_map]
    if not names:
        print(f"  [skip tabicl emb ({emb_type})] no datasets matched to rankings")
        return None

    idx_bench = [bench_names.index(n) for n in names]
    feats    = bench_feats[idx_bench]
    nn_dists = _nn_distances(feats, icl_feats)
    ranks    = np.array([rank_map[n]                    for n in names], float)
    rel_auc  = np.array([rel_auc_map.get(n, np.nan)     for n in names], float)
    rel_best = np.array([rel_best_auc_map.get(n, np.nan) for n in names], float)

    return {"names": names, "nn_dists": nn_dists,
            "ranks": ranks, "rel_auc": rel_auc, "rel_best": rel_best}


# ------------------------------------------------------------------
# Build summary row
# ------------------------------------------------------------------

def _make_row(
    label: str,
    data: dict | None,
    covariates: pd.DataFrame,
    covariate_cols: list[str],
    covariate_tag: str,
) -> dict:
    """Compute partial correlations for all three metrics and return one summary row."""
    nan_row = {
        "Representation":        label,
        "Covariates":            covariate_tag,
        "N":                     0,
        "partial_r (rank)":      float("nan"),
        "partial_p (rank)":      float("nan"),
        "partial_r (rel_auc)":   float("nan"),
        "partial_p (rel_auc)":   float("nan"),
        "partial_r (rel_best)":  float("nan"),
        "partial_p (rel_best)":  float("nan"),
    }
    if data is None:
        return nan_row

    names    = data["names"]
    nn_dists = data["nn_dists"]

    r_rank,  p_rank,  n_rank  = _partial_corr(names, nn_dists, data["ranks"],    covariates, covariate_cols)
    r_auc,   p_auc,   n_auc   = _partial_corr(names, nn_dists, data["rel_auc"],  covariates, covariate_cols)
    r_best,  p_best,  n_best  = _partial_corr(names, nn_dists, data["rel_best"], covariates, covariate_cols)

    def _fmt(v: float) -> float:
        return round(v, 4) if np.isfinite(v) else float("nan")

    return {
        "Representation":        label,
        "Covariates":            covariate_tag,
        "N":                     n_rank,
        "partial_r (rank)":      _fmt(r_rank),
        "partial_p (rank)":      _fmt(p_rank),
        "partial_r (rel_auc)":   _fmt(r_auc),
        "partial_p (rel_auc)":   _fmt(p_auc),
        "partial_r (rel_best)":  _fmt(r_best),
        "partial_p (rel_best)":  _fmt(p_best),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    print("\n=== Loading TALENT rankings ===")
    rank_map, rel_auc_map, rel_best_map, datasets = _talent_rankings()
    print(f"  {len(datasets)} datasets")

    print("\n=== Loading dataset covariates ===")
    covariates = _load_covariates(datasets)
    n_univ = covariates[COVARIATE_COLS_UNIVERSAL].dropna().shape[0]
    n_full = covariates[COVARIATE_COLS_FULL].dropna().shape[0]
    print(f"  Universal covariates complete: {n_univ} / {len(datasets)}")
    print(f"  Full covariates complete:      {n_full} / {len(datasets)} (classification only)")

    print("\n=== Computing partial correlations ===")
    rows = []

    # Load representation data once, reuse for both covariate sets
    print("  Loading representations...")
    repr_data = [
        ("Table features",      _load_table_feats(rank_map, rel_auc_map, rel_best_map, datasets, rng)),
        ("Col histograms",      _load_col_hists(rank_map, rel_auc_map, rel_best_map, rng)),
        ("Corr histograms",     _load_corr_hists(rank_map, rel_auc_map, rel_best_map, rng)),
        ("TabICL col emb",      _load_tabicl_embs(rank_map, rel_auc_map, rel_best_map, datasets, rng, "col")),
        ("TabICL row emb",      _load_tabicl_embs(rank_map, rel_auc_map, rel_best_map, datasets, rng, "row")),
        ("TabICL combined emb", _load_tabicl_embs(rank_map, rel_auc_map, rel_best_map, datasets, rng, "combined")),
    ]

    # Run for both covariate sets
    covariate_sets = [
        ("universal", COVARIATE_COLS_UNIVERSAL,
         "log_n_rows, log_n_cols, prop_categorical (all datasets)"),
        ("full", COVARIATE_COLS_FULL,
         "log_n_rows, log_n_cols, n_classes, class_imbalance, prop_categorical (classification only)"),
    ]

    for label, data in repr_data:
        for tag, cov_cols, cov_desc in covariate_sets:
            rows.append(_make_row(label, data, covariates, cov_cols, tag))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    master = pd.DataFrame(rows)

    out_dir = RESULTS / "partial-correlation"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("rank",     "Mean rank",       "partial_r (rank)",     "partial_p (rank)"),
        ("rel_auc",  "Rel AUC vs mean", "partial_r (rel_auc)",  "partial_p (rel_auc)"),
        ("rel_best", "Rel AUC vs best", "partial_r (rel_best)", "partial_p (rel_best)"),
    ]

    for mkey, mtitle, rcol, pcol in metrics:
        tbl = master[["Representation", "Covariates", "N", rcol, pcol]].copy()
        tbl.columns = ["Representation", "Covariates", "N", "partial_r", "partial_p"]
        tbl = tbl.set_index(["Representation", "Covariates"])

        out_path = out_dir / f"partial_corr_{mkey}.csv"
        tbl.to_csv(out_path)

        print(f"\n{'='*70}")
        print(f"  Performance metric: {mtitle}")
        print(f"{'='*70}")
        with pd.option_context("display.float_format", "{:.4f}".format,
                               "display.max_columns", 10, "display.width", 140):
            print(tbl.to_string())
        print(f"\n  → saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
