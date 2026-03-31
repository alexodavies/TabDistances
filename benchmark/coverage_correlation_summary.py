"""Quantitative summary of coverage and distance–performance correlation.

For every (benchmark × feature representation) combination, computes:
  - Recall / precision coverage via k-NN proximity (knn_coverage)
  - Pearson r between dataset-level k-NN distance to the ICL prior and each
    performance metric (mean rank, rel AUC vs mean, rel AUC vs best)

Outputs one CSV table per performance metric (and prints them to stdout).

Caches used (must exist — run the individual benchmark scripts first):
  TALENT
    cache/talent_feats_512.npz          (names, feats)
    cache/real_feats_icl_25000_512_v2.npz (feats)
    cache/talent_col_hists.npz          (hists, col_datasets)
    cache/icl_col_hists_<n>.npz         (hists)
    cache/talent_corr_hists.npz         (hists, names)
    cache/tabicl_corr_hists.npz         (hists)
    cache/talent_tabicl_emb_colrow.npz  (names, feats_col, feats_row)
    cache/icl_tabicl_emb_25000_colrow.npz (feats_col, feats_row)

  TabArena
    cache/tabarena_feats_512.npz        (names, feats)
    cache/tabarena_col_hists.npz        (hists, col_datasets)
    cache/tabarena_corr_hists.npz       (hists, names)
    (ICL caches shared with TALENT)

Usage
-----
    python benchmark/coverage_correlation_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

from tabdistances.analysis.distances import knn_coverage

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_ROOT  = Path(__file__).resolve().parent.parent
CACHE  = _ROOT / "cache"
RESULTS = _ROOT / "results"

RANKINGS_CSV          = _ROOT / "results/tabicl-rankings-full-context.csv"
TABARENA_RESULTS_CSV  = CACHE / "tabarena_results.csv"

# TALENT caches
TALENT_FEATS_CACHE    = CACHE / "talent_feats_512.npz"
ICL_FEATS_CACHE       = CACHE / "real_feats_icl_25000_512_v2.npz"
TALENT_COL_CACHE      = CACHE / "talent_col_hists.npz"
TALENT_CORR_CACHE     = CACHE / "talent_corr_hists.npz"
TABICL_CORR_CACHE     = CACHE / "tabicl_corr_hists.npz"
TALENT_EMB_CACHE      = CACHE / "talent_tabicl_emb_colrow.npz"
ICL_EMB_CACHE         = CACHE / "icl_tabicl_emb_25000_colrow.npz"

# TabArena caches
TABARENA_FEATS_CACHE  = CACHE / "tabarena_feats_512.npz"
TABARENA_COL_CACHE    = CACHE / "tabarena_col_hists.npz"
TABARENA_CORR_CACHE   = CACHE / "tabarena_corr_hists.npz"
TABARENA_EMB_CACHE    = CACHE / "tabarena_tabicl_emb_colrow.npz"

TABICL_METHOD  = "TABICL_GPU (default)"
KNN_K          = 5
RANDOM_STATE   = 42


# ------------------------------------------------------------------
# k-NN distance (same implementation as the benchmark scripts)
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
# Pearson r helper
# ------------------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Pearson r between x and y, dropping NaN pairs. Returns (r, p, n)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())


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
# Rankings: TabArena
# ------------------------------------------------------------------

def _tabarena_rankings() -> tuple[dict, dict, dict, list[str]]:
    """Returns (rank_map, rel_auc_map, rel_best_auc_map, datasets)."""
    if not TABARENA_RESULTS_CSV.exists():
        print(f"  [skip] TabArena results not found: {TABARENA_RESULTS_CSV}")
        return {}, {}, {}, []
    df = pd.read_csv(TABARENA_RESULTS_CSV)
    agg = (df.groupby(["dataset", "method"])["metric_error"]
             .mean()
             .reset_index())
    agg["rank"] = (agg.groupby("dataset")["metric_error"]
                      .rank(method="min", ascending=True))
    tc = agg[agg["method"] == TABICL_METHOD][["dataset", "rank"]].copy()
    if tc.empty:
        print(f"  [skip] Method '{TABICL_METHOD}' not found in TabArena results")
        return {}, {}, {}, []
    rank_map = dict(zip(tc["dataset"], tc["rank"].astype(int)))
    # AUC maps (classification only)
    cls_datasets = set(
        df[df["metric"].str.contains("roc_auc", na=False)]["dataset"].unique()
    ) if "metric" in df.columns else set(
        df[df["problem_type"].isin(["binary", "multiclass"])]["dataset"].unique()
    )
    agg_cls = agg[agg["dataset"].isin(cls_datasets)].copy()
    agg_cls["auc"] = 1.0 - agg_cls["metric_error"]
    tc_cls = agg_cls[agg_cls["method"] == TABICL_METHOD][["dataset", "auc"]].copy()
    others = agg_cls[agg_cls["method"] != TABICL_METHOD]
    mean_auc = others.groupby("dataset")["auc"].mean()
    best_auc = others.groupby("dataset")["auc"].max()
    rel_auc_map = {
        ds: auc / mean_auc[ds]
        for ds, auc in zip(tc_cls["dataset"], tc_cls["auc"])
        if ds in mean_auc and mean_auc[ds] > 0
    }
    rel_best_auc_map = {
        ds: auc / best_auc[ds]
        for ds, auc in zip(tc_cls["dataset"], tc_cls["auc"])
        if ds in best_auc and best_auc[ds] > 0
    }
    return rank_map, rel_auc_map, rel_best_auc_map, tc["dataset"].tolist()


# ------------------------------------------------------------------
# Per-representation loading helpers
# ------------------------------------------------------------------

def _load_table_feats(
    bench_feats_cache: Path,
    icl_feats_cache: Path,
    rank_map: dict,
    rel_auc_map: dict,
    rel_best_auc_map: dict,
    datasets: list[str],
    rng: np.random.Generator,
    label: str,
) -> dict | None:
    """Load table-level feature vectors. Returns metric dict or None if cache missing."""
    for path in (bench_feats_cache, icl_feats_cache):
        if not path.exists():
            print(f"  [skip {label} table feats] cache missing: {path}")
            return None

    data = np.load(bench_feats_cache, allow_pickle=True)
    names_arr = data["names"].tolist()
    feats_arr = data["feats"]
    cached = {n: feats_arr[i] for i, n in enumerate(names_arr)}

    icl_all = np.load(icl_feats_cache)["feats"]
    n_icl = min(25_000, len(icl_all))
    idx = rng.choice(len(icl_all), size=n_icl, replace=False)
    icl_feats = icl_all[idx]

    names   = [ds for ds in datasets if ds in cached]
    feats   = np.stack([cached[ds] for ds in names])
    ranks   = np.array([rank_map[n]              for n in names], dtype=float)
    rel_auc = np.array([rel_auc_map.get(n, np.nan)      for n in names], dtype=float)
    rel_best= np.array([rel_best_auc_map.get(n, np.nan) for n in names], dtype=float)

    nn_dists = _nn_distances(feats, icl_feats)
    recall, precision = knn_coverage(feats, icl_feats, k=KNN_K)

    return {
        "nn_dists":  nn_dists,
        "ranks":     ranks,
        "rel_auc":   rel_auc,
        "rel_best":  rel_best,
        "recall":    recall,
        "precision": precision,
        "N":         len(names),
    }


def _load_col_hists(
    bench_col_cache: Path,
    icl_col_cache_pattern: str,
    rank_map: dict,
    rel_auc_map: dict,
    rel_best_auc_map: dict,
    rng: np.random.Generator,
    label: str,
) -> dict | None:
    """Load column-histogram representations. Returns metric dict or None."""
    if not bench_col_cache.exists():
        print(f"  [skip {label} col hists] cache missing: {bench_col_cache}")
        return None

    cdata = np.load(bench_col_cache, allow_pickle=True)
    talent_hists = cdata["hists"]
    col_datasets = cdata["col_datasets"]

    n_talent = len(talent_hists)
    icl_col_cache = CACHE / icl_col_cache_pattern.format(n=n_talent)
    if not icl_col_cache.exists():
        print(f"  [skip {label} col hists] ICL cache missing: {icl_col_cache}")
        return None

    icl_hists_all = np.load(icl_col_cache)["hists"]
    idx = rng.choice(len(icl_hists_all), size=min(n_talent, len(icl_hists_all)), replace=False)
    icl_hists = icl_hists_all[idx]

    # Filter to columns with rank data
    mask = np.array([ds in rank_map for ds in col_datasets])
    hists_m    = talent_hists[mask]
    matched_ds = col_datasets[mask]

    nn_dists_col = _nn_distances(hists_m, icl_hists)

    # Aggregate to dataset level (mean distance per dataset)
    df_tmp = pd.DataFrame({"ds": matched_ds, "dist": nn_dists_col})
    ds_mean = df_tmp.groupby("ds")["dist"].mean()

    names   = ds_mean.index.tolist()
    nn_dists = ds_mean.values
    ranks   = np.array([rank_map[n]              for n in names], dtype=float)
    rel_auc = np.array([rel_auc_map.get(n, np.nan)      for n in names], dtype=float)
    rel_best= np.array([rel_best_auc_map.get(n, np.nan) for n in names], dtype=float)

    recall, precision = knn_coverage(hists_m, icl_hists, k=KNN_K)

    return {
        "nn_dists":  nn_dists,
        "ranks":     ranks,
        "rel_auc":   rel_auc,
        "rel_best":  rel_best,
        "recall":    recall,
        "precision": precision,
        "N":         len(names),
    }


def _load_corr_hists(
    bench_corr_cache: Path,
    icl_corr_cache: Path,
    rank_map: dict,
    rel_auc_map: dict,
    rel_best_auc_map: dict,
    rng: np.random.Generator,
    label: str,
) -> dict | None:
    """Load correlation-histogram representations. Returns metric dict or None."""
    for path in (bench_corr_cache, icl_corr_cache):
        if not path.exists():
            print(f"  [skip {label} corr hists] cache missing: {path}")
            return None

    bd = np.load(bench_corr_cache, allow_pickle=True)
    bench_hists = bd["hists"]
    bench_names = bd["names"].tolist()

    icl_all = np.load(icl_corr_cache)["hists"]
    idx = rng.choice(len(icl_all), size=min(25_000, len(icl_all)), replace=False)
    icl_hists = icl_all[idx]

    mask    = np.array([n in rank_map for n in bench_names])
    hists_m = bench_hists[mask]
    names_m = np.array(bench_names)[mask]

    nn_dists = _nn_distances(hists_m, icl_hists)
    ranks   = np.array([rank_map[n]              for n in names_m], dtype=float)
    rel_auc = np.array([rel_auc_map.get(n, np.nan)      for n in names_m], dtype=float)
    rel_best= np.array([rel_best_auc_map.get(n, np.nan) for n in names_m], dtype=float)

    recall, precision = knn_coverage(hists_m, icl_hists, k=KNN_K)

    return {
        "nn_dists":  nn_dists,
        "ranks":     ranks,
        "rel_auc":   rel_auc,
        "rel_best":  rel_best,
        "recall":    recall,
        "precision": precision,
        "N":         len(names_m),
    }


def _load_tabicl_embs(
    bench_emb_cache: Path,
    icl_emb_cache: Path,
    rank_map: dict,
    rel_auc_map: dict,
    rel_best_auc_map: dict,
    datasets: list[str],
    rng: np.random.Generator,
    label: str,
    emb_type: str,  # "col", "row", or "combined"
) -> dict | None:
    """Load pre-trained TabICL embedding representations. Returns metric dict or None."""
    for path in (bench_emb_cache, icl_emb_cache):
        if not path.exists():
            print(f"  [skip {label} tabicl emb ({emb_type})] cache missing: {path}")
            return None

    bd = np.load(bench_emb_cache, allow_pickle=True)
    if "feats_col" not in bd or len(bd["feats_col"]) == 0:
        print(f"  [skip {label} tabicl emb ({emb_type})] cache empty: {bench_emb_cache}")
        return None

    bench_names = bd["names"].tolist()
    bench_col   = bd["feats_col"]
    bench_row   = bd["feats_row"]

    id_ = np.load(icl_emb_cache, allow_pickle=True)
    if "feats_col" not in id_ or len(id_["feats_col"]) == 0:
        print(f"  [skip {label} tabicl emb ({emb_type})] ICL cache empty: {icl_emb_cache}")
        return None
    icl_col = id_["feats_col"]
    icl_row = id_["feats_row"]

    # Subsample ICL
    n_icl = min(25_000, len(icl_col))
    idx = rng.choice(len(icl_col), size=n_icl, replace=False)
    icl_col = icl_col[idx]
    icl_row = icl_row[idx]

    # Build combined
    bench_combined = np.concatenate([bench_col, bench_row], axis=1)
    icl_combined   = np.concatenate([icl_col,   icl_row],   axis=1)

    if emb_type == "col":
        bench_feats, icl_feats = bench_col, icl_col
    elif emb_type == "row":
        bench_feats, icl_feats = bench_row, icl_row
    else:
        bench_feats, icl_feats = bench_combined, icl_combined

    # Match to datasets with rankings
    cache_name_set = set(bench_names)
    names   = [ds for ds in datasets if ds in cache_name_set and ds in rank_map]
    if not names:
        print(f"  [skip {label} tabicl emb ({emb_type})] no datasets matched to rankings")
        return None

    idx_bench = [bench_names.index(n) for n in names]
    feats   = bench_feats[idx_bench]
    ranks   = np.array([rank_map[n]              for n in names], dtype=float)
    rel_auc = np.array([rel_auc_map.get(n, np.nan)      for n in names], dtype=float)
    rel_best= np.array([rel_best_auc_map.get(n, np.nan) for n in names], dtype=float)

    nn_dists = _nn_distances(feats, icl_feats)
    recall, precision = knn_coverage(feats, icl_feats, k=KNN_K)

    return {
        "nn_dists":  nn_dists,
        "ranks":     ranks,
        "rel_auc":   rel_auc,
        "rel_best":  rel_best,
        "recall":    recall,
        "precision": precision,
        "N":         len(names),
    }


# ------------------------------------------------------------------
# Build summary rows
# ------------------------------------------------------------------

def _make_row(label: str, bench: str, data: dict | None) -> dict:
    """Convert a loaded data dict into one summary row."""
    if data is None:
        return {
            "Representation": label,
            "Benchmark":      bench,
            "Recall":         float("nan"),
            "Precision":      float("nan"),
            "Pearson_r (rank)":    float("nan"),
            "Pearson_p (rank)":    float("nan"),
            "Pearson_r (rel_auc)": float("nan"),
            "Pearson_p (rel_auc)": float("nan"),
            "Pearson_r (rel_best)": float("nan"),
            "Pearson_p (rel_best)": float("nan"),
            "N": 0,
        }

    r_rank, p_rank, n_rank       = _pearson(data["nn_dists"], data["ranks"])
    # Correlation with AUC metrics (higher AUC = better → negative r expected with distance)
    r_auc,  p_auc,  n_auc        = _pearson(data["nn_dists"], data["rel_auc"])
    r_best, p_best, n_best       = _pearson(data["nn_dists"], data["rel_best"])

    return {
        "Representation":       label,
        "Benchmark":            bench,
        "Recall":               round(data["recall"],    4),
        "Precision":            round(data["precision"], 4),
        "Pearson_r (rank)":     round(r_rank, 4) if np.isfinite(r_rank) else float("nan"),
        "Pearson_p (rank)":     round(p_rank, 4) if np.isfinite(p_rank) else float("nan"),
        "Pearson_r (rel_auc)":  round(r_auc,  4) if np.isfinite(r_auc)  else float("nan"),
        "Pearson_p (rel_auc)":  round(p_auc,  4) if np.isfinite(p_auc)  else float("nan"),
        "Pearson_r (rel_best)": round(r_best, 4) if np.isfinite(r_best) else float("nan"),
        "Pearson_p (rel_best)": round(p_best, 4) if np.isfinite(p_best) else float("nan"),
        "N":                    data["N"],
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    # --- TALENT rankings ---
    print("\n=== Loading TALENT rankings ===")
    t_rank, t_rel_auc, t_rel_best, t_datasets = _talent_rankings()
    print(f"  {len(t_datasets)} datasets loaded")

    # --- TabArena rankings ---
    print("\n=== Loading TabArena rankings ===")
    a_rank, a_rel_auc, a_rel_best, a_datasets = _tabarena_rankings()
    print(f"  {len(a_datasets)} datasets loaded")

    # --- Collect all rows ---
    rows = []

    print("\n=== TALENT ===")
    print("--- Table features ---")
    rows.append(_make_row(
        "Table features", "TALENT",
        _load_table_feats(TALENT_FEATS_CACHE, ICL_FEATS_CACHE,
                          t_rank, t_rel_auc, t_rel_best, t_datasets, rng, "TALENT"),
    ))
    print("--- Column histograms ---")
    rows.append(_make_row(
        "Col histograms", "TALENT",
        _load_col_hists(TALENT_COL_CACHE, "icl_col_hists_{n}.npz",
                        t_rank, t_rel_auc, t_rel_best, rng, "TALENT"),
    ))
    print("--- Correlation histograms ---")
    rows.append(_make_row(
        "Corr histograms", "TALENT",
        _load_corr_hists(TALENT_CORR_CACHE, TABICL_CORR_CACHE,
                         t_rank, t_rel_auc, t_rel_best, rng, "TALENT"),
    ))
    for emb_type, label in [("col", "TabICL col emb"), ("row", "TabICL row emb"),
                             ("combined", "TabICL combined emb")]:
        print(f"--- TabICL emb ({emb_type}) ---")
        rows.append(_make_row(
            label, "TALENT",
            _load_tabicl_embs(TALENT_EMB_CACHE, ICL_EMB_CACHE,
                              t_rank, t_rel_auc, t_rel_best, t_datasets, rng, "TALENT", emb_type),
        ))

    print("\n=== TabArena ===")
    print("--- Table features ---")
    rows.append(_make_row(
        "Table features", "TabArena",
        _load_table_feats(TABARENA_FEATS_CACHE, ICL_FEATS_CACHE,
                          a_rank, a_rel_auc, a_rel_best, a_datasets, rng, "TabArena"),
    ))
    print("--- Column histograms ---")
    rows.append(_make_row(
        "Col histograms", "TabArena",
        _load_col_hists(TABARENA_COL_CACHE, "icl_col_hists_{n}.npz",
                        a_rank, a_rel_auc, a_rel_best, rng, "TabArena"),
    ))
    print("--- Correlation histograms ---")
    rows.append(_make_row(
        "Corr histograms", "TabArena",
        _load_corr_hists(TABARENA_CORR_CACHE, TABICL_CORR_CACHE,
                         a_rank, a_rel_auc, a_rel_best, rng, "TabArena"),
    ))
    for emb_type, label in [("col", "TabICL col emb"), ("row", "TabICL row emb"),
                             ("combined", "TabICL combined emb")]:
        print(f"--- TabICL emb ({emb_type}) ---")
        rows.append(_make_row(
            label, "TabArena",
            _load_tabicl_embs(TABARENA_EMB_CACHE, ICL_EMB_CACHE,
                              a_rank, a_rel_auc, a_rel_best, a_datasets, rng, "TabArena", emb_type),
        ))

    # --- Build master DataFrame ---
    master = pd.DataFrame(rows)

    # --- Output one table per metric ---
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS / "coverage-summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("rank",     "Mean rank",          "Pearson_r (rank)",     "Pearson_p (rank)"),
        ("rel_auc",  "Rel AUC vs mean",     "Pearson_r (rel_auc)",  "Pearson_p (rel_auc)"),
        ("rel_best", "Rel AUC vs best",     "Pearson_r (rel_best)", "Pearson_p (rel_best)"),
    ]

    repr_order = list(dict.fromkeys(r["Representation"] for r in rows))
    bench_order = ["TALENT", "TabArena"]

    for mkey, mtitle, rcol, pcol in metrics:
        # Build wide table manually: one row per representation, columns per benchmark
        wide_rows = []
        for rep in repr_order:
            row_out: dict = {"Representation": rep}
            for bench in bench_order:
                match = [r for r in rows if r["Representation"] == rep and r["Benchmark"] == bench]
                if match:
                    r = match[0]
                    row_out[f"{bench} N"]         = r["N"]
                    row_out[f"{bench} Recall"]    = r["Recall"]
                    row_out[f"{bench} Precision"] = r["Precision"]
                    row_out[f"{bench} Pearson_r"] = r[rcol]
                    row_out[f"{bench} Pearson_p"] = r[pcol]
                else:
                    for col in ("N", "Recall", "Precision", "Pearson_r", "Pearson_p"):
                        row_out[f"{bench} {col}"] = float("nan")
            wide_rows.append(row_out)

        tbl_wide = pd.DataFrame(wide_rows).set_index("Representation")

        out_path = out_dir / f"summary_{mkey}.csv"
        tbl_wide.to_csv(out_path)

        print(f"\n{'='*70}")
        print(f"  Performance metric: {mtitle}")
        print(f"{'='*70}")
        with pd.option_context("display.float_format", "{:.4f}".format,
                               "display.max_columns", 20, "display.width", 200):
            print(tbl_wide.to_string())
        print(f"\n  → saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
