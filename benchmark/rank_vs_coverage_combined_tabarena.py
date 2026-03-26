"""3-panel combined histogram + mean-rank figure for TabArena datasets.

Same analysis as rank_vs_coverage_combined_talent.py but uses TabArena benchmark
results to derive ranks instead of tabicl-rankings-full-context.csv.

One panel per distance representation:
  (1) Table feature vectors
  (2) Column histograms
  (3) Correlation histograms

Each panel:
  - Left y-axis:  normalised NN-distance density (TabArena datasets + TabICL Prior self)
  - Right y-axis: mean TabICL rank ± 1 SD per distance bin (inverted, lower = better)

Ranks are computed from the TabArena results CSV (average metric_error over
folds/seeds per method, then rank methods per dataset; lower rank = better).

Usage
-----
    # Option A: provide a local results CSV and dataset directory
    python benchmark/rank_vs_coverage_combined_tabarena.py \
        --results-csv  /path/to/tabarena_results.csv \
        --arena-data-dir /path/to/tabarena/datasets \
        [--output-dir results/rank-coverage-tabarena] \
        [--n-bins 30]

    # Option B: download results via the tabarena Python package
    python benchmark/rank_vs_coverage_combined_tabarena.py \
        --download-results \
        --arena-data-dir /path/to/tabarena/datasets \
        [--output-dir results/rank-coverage-tabarena]

    # Option C: download everything (results + datasets via OpenML)
    python benchmark/rank_vs_coverage_combined_tabarena.py \
        --download-results \
        --download-datasets \
        [--output-dir results/rank-coverage-tabarena]

Dataset files
-------------
--arena-data-dir should contain one file per dataset named exactly after the
dataset as it appears in the results CSV, e.g. APSFailure.csv or
APSFailure.parquet.  Only numeric columns are used.

If --download-datasets is given, datasets are fetched from OpenML using the
`openml` package (pip install openml) and cached in --arena-data-dir.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from tabdistances.discriminator.features import df_to_vec
from tabdistances.columns.column_features import get_histogram

# ------------------------------------------------------------------
# ICL prior cache builders (self-contained — no TALENT data needed)
# ------------------------------------------------------------------

def _build_icl_feats_cache(rng: np.random.Generator) -> np.ndarray:
    """Featurise TabICL parquet tables and write ICL_FEATS_CACHE. Returns array."""
    if not ICL_DIR.exists():
        raise FileNotFoundError(
            f"TabICL parquet dir not found: {ICL_DIR}\n"
            f"Run:  python fetch_data.py --icl"
        )
    paths = np.array(sorted(ICL_DIR.glob("*.parquet")))
    rng.shuffle(paths)
    feats, errors = [], 0
    for p in tqdm(paths, desc="Featurising TabICL tables"):
        try:
            df = pd.read_parquet(p).select_dtypes(include="number")
            if df.shape[0] == 0 or df.shape[1] == 0:
                errors += 1
                continue
            feats.append(df_to_vec(df.iloc[:MAX_ROWS]))
        except Exception:
            errors += 1
        if len(feats) >= N_ICL:
            break
    print(f"  Featurised {len(feats)} TabICL tables ({errors} skipped)")
    arr = np.array(feats)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(ICL_FEATS_CACHE, feats=arr)
    print(f"  Cached -> {ICL_FEATS_CACHE}")
    return arr


def _build_icl_col_cache(n_query: int, rng: np.random.Generator) -> np.ndarray:
    """Build column-histogram cache for the ICL prior. Returns array."""
    if not ICL_DIR.exists():
        raise FileNotFoundError(
            f"TabICL parquet dir not found: {ICL_DIR}\n"
            f"Run:  python fetch_data.py --icl"
        )
    all_files = sorted(ICL_DIR.glob("*.parquet"))
    rng_local = np.random.default_rng(RANDOM_STATE)
    rng_local.shuffle(all_files := np.array(all_files))
    hists = []
    for p in tqdm(all_files, desc="ICL column histograms"):
        try:
            df = pd.read_parquet(p).select_dtypes(include="number").iloc[:MAX_ROWS]
            for col in df.columns:
                hists.append(get_histogram(df[col]))
            if len(hists) >= n_query * 10:   # build a large pool to sample from
                break
        except Exception:
            pass
    arr = np.array(hists)
    out = CACHE / f"icl_col_hists_{n_query}.npz"
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, hists=arr)
    print(f"  ICL col hists cached ({len(arr)} cols) -> {out}")
    return arr

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
_ROOT  = Path(__file__).resolve().parent.parent
CACHE  = _ROOT / "cache"

ICL_DIR           = _ROOT / "tables-tabicl/parquet"
ICL_FEATS_CACHE   = CACHE / "real_feats_icl_25000_512_v2.npz"
TABICL_CORR_CACHE = CACHE / "tabicl_corr_hists.npz"

TABICL_METHOD   = "TABICL_GPU (default)"
TABARENA_RESULTS_CACHE = CACHE / "tabarena_results.csv"

MAX_ROWS     = 512
KNN_K        = 5
N_ICL        = 25_000
RANDOM_STATE = 42

# Correlation histogram config (matches rank_vs_coverage_combined_talent.py)
N_CORR_BINS  = 50
CORR_EDGES   = np.linspace(-1.0, 1.0, N_CORR_BINS + 1)
MIN_NUM_COLS = 2

_PAPER_RC = {
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.5,
    "figure.dpi":      300,
}

_BLUE   = "#2980b9"
_PURPLE = "#8e44ad"
_RED    = "#ef0e0e"


# ------------------------------------------------------------------
# Shared utilities (identical to the TALENT version)
# ------------------------------------------------------------------

def _stairs(ax, density, bins, color, label, alpha_fill=0.15, lw=1.5):
    patch = ax.stairs(density, bins, fill=True, label=label)
    patch.set_facecolor((*mpl.colors.to_rgb(color), alpha_fill))
    patch.set_edgecolor(color)
    patch.set_linewidth(lw)


def _nn_distances(query: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    """Mean k-NN distance from each query point to the reference pool."""
    query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
    ref   = np.nan_to_num(ref,   nan=0.0, posinf=0.0, neginf=0.0)
    scaler = MinMaxScaler().fit(ref)
    q = scaler.transform(query)
    r = scaler.transform(ref)
    tree = BallTree(r, metric="euclidean")
    dists, _ = tree.query(q, k=min(k, len(r)))
    return dists.mean(axis=1)


def _bin_metric(nn_dists: np.ndarray, ranks: np.ndarray,
                n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Bin data, zero out singleton bins, then rebin over trimmed range."""
    bins1 = np.linspace(nn_dists.min(), nn_dists.max(), n_bins + 1)
    idx1  = np.clip(np.digitize(nn_dists, bins1) - 1, 0, n_bins - 1)
    cts1  = np.array([(idx1 == i).sum() for i in range(n_bins)])
    nz    = np.where(cts1 > 0)[0]
    x_lo  = bins1[nz[0]]      if len(nz) else bins1[0]
    x_hi  = bins1[nz[-1] + 1] if len(nz) else bins1[-1]

    inner = (nn_dists >= x_lo) & (nn_dists <= x_hi)
    nd, rk = nn_dists[inner], ranks[inner]
    bins = np.linspace(x_lo, x_hi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    idx  = np.clip(np.digitize(nd, bins) - 1, 0, n_bins - 1)
    cts  = np.array([(idx == i).sum() for i in range(n_bins)])
    density = cts.astype(float)
    density /= density.max() if density.max() > 0 else 1.0
    means = np.array([np.nanmean(rk[idx == i]) if cts[i] > 0 else np.nan for i in range(n_bins)])
    stds  = np.array([np.nanstd(rk[idx == i])  if cts[i] > 1 else 0.0   for i in range(n_bins)])
    return bins, bin_centers, density, means, stds, x_lo, x_hi


# ------------------------------------------------------------------
# TabArena results download (requires `tabarena` package)
# ------------------------------------------------------------------

def _download_tabarena_results() -> Path:
    """Download TabArena results via the tabarena package and cache locally.

    Requires the tabarena install:
        git clone https://github.com/autogluon/tabarena.git
        cd tabarena && pip install -e ./bencheval && pip install -e ./tabarena

    Results are fetched as a DataFrame with columns:
        dataset, fold, method, metric_error, time_train_s, time_infer_s,
        seed, problem_type, metric
    """
    if TABARENA_RESULTS_CACHE.exists():
        print(f"  TabArena results: using cache {TABARENA_RESULTS_CACHE}")
        return TABARENA_RESULTS_CACHE

    try:
        from tabarena.nips2025_utils.tabarena_context import TabArenaContext  # type: ignore
    except ImportError:
        raise ImportError(
            "The `tabarena` package is not installed.\n"
            "  git clone https://github.com/autogluon/tabarena.git\n"
            "  cd tabarena && pip install -e ./bencheval && pip install -e ./tabarena\n"
            "Or supply --results-csv with a locally saved CSV."
        )

    # Fix namespace-package collision: if tabarena was imported as a namespace
    # package (tabarena.__file__ is None), patch __file__ so TabArenaContext
    # can locate its bundled metadata files.
    import tabarena as _ta
    if _ta.__file__ is None:
        import tabarena.nips2025_utils.tabarena_context as _ctx_mod
        from pathlib import Path as _Path
        _ta.__file__ = str(_Path(_ctx_mod.__file__).parent.parent / "__init__.py")

    print("  Downloading TabArena results …")
    ctx = TabArenaContext()
    df = ctx.load_results_paper(download_results="auto")
    # Normalise column name: new API uses 'framework', script expects 'method'
    if "framework" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"framework": "method"})
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABARENA_RESULTS_CACHE, index=False)
    print(f"  Saved to {TABARENA_RESULTS_CACHE}")
    return TABARENA_RESULTS_CACHE


# ------------------------------------------------------------------
# OpenML dataset download
# ------------------------------------------------------------------

TABARENA_SUITE   = "tabarena-v0.1"
TASK_MAP_CACHE   = CACHE / "tabarena_task_map.json"


def _get_task_map() -> dict[str, int]:
    """Return {dataset_name: openml_task_id} for the TabArena benchmark suite.

    The mapping is built once from the OpenML suite and cached locally.
    """
    import json
    import openml  # type: ignore

    if TASK_MAP_CACHE.exists():
        with open(TASK_MAP_CACHE) as f:
            return json.load(f)

    print(f"  Fetching TabArena task list from OpenML suite '{TABARENA_SUITE}' …")
    suite = openml.study.get_suite(TABARENA_SUITE)
    task_map: dict[str, int] = {}
    for task_id in tqdm(suite.tasks, desc="Resolving task names"):
        try:
            task = openml.tasks.get_task(task_id, download_data=False)
            ds   = task.get_dataset()
            task_map[ds.name] = task_id
        except Exception as e:
            print(f"    Warning: could not resolve task {task_id}: {e}")

    CACHE.mkdir(parents=True, exist_ok=True)
    with open(TASK_MAP_CACHE, "w") as f:
        json.dump(task_map, f, indent=2)
    print(f"  Task map cached ({len(task_map)} datasets) -> {TASK_MAP_CACHE}")
    return task_map


def _download_dataset_openml(dataset: str, data_dir: Path,
                              task_map: dict[str, int] | None = None) -> bool:
    """Fetch a TabArena dataset from OpenML by task ID, save numeric cols as parquet."""
    out = data_dir / f"{dataset}.parquet"
    if out.exists():
        return True

    try:
        import openml  # type: ignore
    except ImportError:
        raise ImportError(
            "The `openml` package is not installed. "
            "Run `pip install openml` or provide --arena-data-dir with local files."
        )

    if task_map is None:
        task_map = _get_task_map()

    task_id = task_map.get(dataset)
    if task_id is None:
        print(f"    OpenML: '{dataset}' not in TabArena suite, skipping")
        return False

    try:
        task = openml.tasks.get_task(task_id)
        ds   = task.get_dataset()
        X, _, _, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
        num = X.select_dtypes(include="number")
        data_dir.mkdir(parents=True, exist_ok=True)
        num.to_parquet(out, index=False)
        return True
    except Exception as e:
        print(f"    OpenML: failed to download '{dataset}' (task {task_id}): {e}")
        return False


# ------------------------------------------------------------------
# Rankings from TabArena results CSV
# ------------------------------------------------------------------

def _load_rankings(results_csv: Path) -> tuple[dict, dict, dict, dict, dict, list[str]]:
    """Parse TabArena results CSV and compute per-dataset TabICL rank.

    Returns (rank_map, rel_auc_map, rel_best_auc_map,
             tabicl_auc_map, mean_auc_map, datasets).
    """
    df = pd.read_csv(results_csv)

    # Average metric_error over folds and seeds
    agg = (df.groupby(["dataset", "method"])["metric_error"]
             .mean()
             .reset_index())

    # Rank methods per dataset (lower metric_error = better = lower rank)
    agg["rank"] = (agg.groupby("dataset")["metric_error"]
                      .rank(method="min", ascending=True))

    tc = agg[agg["method"] == TABICL_METHOD][["dataset", "rank"]].copy()
    if tc.empty:
        raise ValueError(
            f"No rows found for method '{TABICL_METHOD}' in {results_csv}. "
            f"Available methods: {sorted(agg['method'].unique().tolist())}"
        )

    rank_map = dict(zip(tc["dataset"], tc["rank"].astype(int)))

    # AUC maps: only valid for classification datasets (metric_error = 1 - roc_auc)
    cls_datasets = set(
        df[df["metric"].str.contains("roc_auc", na=False)]["dataset"].unique()
    ) if "metric" in df.columns else set(
        df[df["problem_type"].isin(["binary", "multiclass"])]["dataset"].unique()
    )
    agg_cls = agg[agg["dataset"].isin(cls_datasets)].copy()
    agg_cls["auc"] = 1.0 - agg_cls["metric_error"]

    tc_cls = agg_cls[agg_cls["method"] == TABICL_METHOD][["dataset", "auc"]].copy()
    others_cls = agg_cls[agg_cls["method"] != TABICL_METHOD]
    mean_auc = others_cls.groupby("dataset")["auc"].mean()
    best_auc = others_cls.groupby("dataset")["auc"].max()

    tabicl_auc_map = dict(zip(tc_cls["dataset"], tc_cls["auc"]))
    mean_auc_map   = mean_auc.to_dict()

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

    datasets = tc["dataset"].tolist()
    print(f"Rankings: {len(rank_map)} TabArena datasets ({len(tabicl_auc_map)} classification), "
          f"TABICL method: '{TABICL_METHOD}'")
    return rank_map, rel_auc_map, rel_best_auc_map, tabicl_auc_map, mean_auc_map, datasets


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------

def _load_dataset_df(dataset: str, data_dir: Path) -> pd.DataFrame | None:
    """Load a TabArena dataset as a DataFrame (numeric columns only)."""
    for ext in (".parquet", ".csv"):
        p = data_dir / f"{dataset}{ext}"
        if p.exists():
            df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            return df.select_dtypes(include="number")
    return None


# ------------------------------------------------------------------
# Panel 1 — table feature vectors
# ------------------------------------------------------------------

def _load_table_panel(rank_map, rel_auc_map, rel_best_auc_map,
                      tabicl_auc_map, mean_auc_map, datasets, data_dir, rng,
                      cache_prefix: str = "tabarena"):
    feats_cache = CACHE / f"{cache_prefix}_feats_512.npz"
    cached: dict[str, np.ndarray] = {}
    if feats_cache.exists():
        data = np.load(feats_cache, allow_pickle=True)
        cached = {n: data["feats"][i] for i, n in enumerate(data["names"].tolist())
                  if n in set(datasets)}
        print(f"  Table feats: cache hit {len(cached)}/{len(datasets)} datasets")

    missing = [ds for ds in datasets if ds not in cached]
    if missing:
        print(f"  Table feats: computing {len(missing)} missing datasets …")
        skipped = 0
        for ds in tqdm(missing, desc="Featurising TabArena datasets"):
            df = _load_dataset_df(ds, data_dir)
            if df is None or df.shape[0] == 0 or df.shape[1] == 0:
                skipped += 1
                continue
            cached[ds] = df_to_vec(df.iloc[:MAX_ROWS])
        print(f"  Table feats: computed {len(missing) - skipped} ({skipped} skipped)")
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(feats_cache,
                 names=np.array(list(cached.keys())),
                 feats=np.array(list(cached.values())))
        print(f"  Table feats: cache updated -> {feats_cache}")

    names         = [ds for ds in datasets if ds in cached]
    arena_feats   = np.stack([cached[ds] for ds in names])
    ranks         = np.array([rank_map[n]                     for n in names], dtype=float)
    rel_aucs      = np.array([rel_auc_map.get(n, np.nan)      for n in names], dtype=float)
    rel_best_aucs = np.array([rel_best_auc_map.get(n, np.nan) for n in names], dtype=float)
    tabicl_aucs   = np.array([tabicl_auc_map.get(n, np.nan)   for n in names], dtype=float)
    mean_aucs     = np.array([mean_auc_map.get(n, np.nan)      for n in names], dtype=float)
    print(f"  Table feats: {len(names)} TabArena datasets")

    if ICL_FEATS_CACHE.exists():
        icl_all = np.load(ICL_FEATS_CACHE)["feats"]
    else:
        print(f"  Table feats: ICL cache missing, building from {ICL_DIR} …")
        icl_all = _build_icl_feats_cache(rng)
    idx = rng.choice(len(icl_all), size=min(N_ICL, len(icl_all)), replace=False)
    icl_feats = icl_all[idx]
    print(f"  Table feats: {len(icl_feats)} ICL vectors")

    nn_dists = _nn_distances(arena_feats, icl_feats, KNN_K)
    half = len(icl_feats) // 2
    nn_self = _nn_distances(icl_feats[:half], icl_feats[half:], KNN_K)
    return nn_dists, nn_self, ranks, rel_aucs, rel_best_aucs, tabicl_aucs, mean_aucs


# ------------------------------------------------------------------
# Panel 2 — column histograms
# ------------------------------------------------------------------

def _load_column_panel(rank_map, rel_auc_map, rel_best_auc_map,
                       tabicl_auc_map, mean_auc_map, datasets, data_dir, rng,
                       cache_prefix: str = "tabarena"):
    col_cache = CACHE / f"{cache_prefix}_col_hists.npz"

    if col_cache.exists():
        cdata = np.load(col_cache, allow_pickle=True)
        arena_hists  = cdata["hists"]
        col_datasets = cdata["col_datasets"]
        print(f"  Col hists: {len(arena_hists)} TabArena columns from cache")
    else:
        print(f"  Col hists: computing column histograms …")
        hists, col_ds = [], []
        for ds in tqdm(datasets, desc="Column histograms"):
            df = _load_dataset_df(ds, data_dir)
            if df is None:
                continue
            df = df.iloc[:MAX_ROWS]
            for col in df.columns:
                h = get_histogram(df[col])
                hists.append(h)
                col_ds.append(ds)
        arena_hists  = np.array(hists)
        col_datasets = np.array(col_ds)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(col_cache, hists=arena_hists, col_datasets=col_datasets)
        print(f"  Col hists: {len(arena_hists)} columns cached -> {col_cache}")

    # ICL column hists (reuse TALENT cache if available, else build)
    n_arena = len(arena_hists)
    icl_col_cache = CACHE / f"icl_col_hists_{n_arena}.npz"
    if not icl_col_cache.exists():
        # Fall back to any existing icl_col_hists_*.npz (built by TALENT pipeline)
        existing = sorted(CACHE.glob("icl_col_hists_*.npz"))
        if existing:
            icl_col_cache = existing[-1]
            print(f"  Col hists: using existing ICL col cache {icl_col_cache.name}")
    if icl_col_cache.exists():
        icl_hists = np.load(icl_col_cache)["hists"]
    else:
        print(f"  Col hists: ICL cache missing, building from {ICL_DIR} …")
        icl_hists = _build_icl_col_cache(n_arena, rng)
    idx = rng.choice(len(icl_hists), size=min(n_arena, len(icl_hists)), replace=False)
    icl_hists = icl_hists[idx]
    print(f"  Col hists: {len(icl_hists)} ICL columns")

    mask = np.array([ds in rank_map for ds in col_datasets])
    arena_hists_m = arena_hists[mask]
    matched_ds    = col_datasets[mask]
    ranks         = np.array([rank_map[ds]                     for ds in matched_ds], dtype=float)
    rel_aucs      = np.array([rel_auc_map.get(ds, np.nan)      for ds in matched_ds], dtype=float)
    rel_best_aucs = np.array([rel_best_auc_map.get(ds, np.nan) for ds in matched_ds], dtype=float)
    tabicl_aucs   = np.array([tabicl_auc_map.get(ds, np.nan)   for ds in matched_ds], dtype=float)
    mean_aucs     = np.array([mean_auc_map.get(ds, np.nan)      for ds in matched_ds], dtype=float)

    nn_dists = _nn_distances(arena_hists_m, icl_hists, KNN_K)
    half = len(icl_hists) // 2
    nn_self = _nn_distances(icl_hists[:half], icl_hists[half:], KNN_K)
    print(f"  Col hists: {mask.sum()} columns matched to ranks")
    return nn_dists, nn_self, ranks, rel_aucs, rel_best_aucs, tabicl_aucs, mean_aucs


# ------------------------------------------------------------------
# Panel 3 — correlation histograms
# ------------------------------------------------------------------

def _corr_pmf(arr2d: np.ndarray) -> np.ndarray | None:
    df = pd.DataFrame(arr2d)
    df = df.loc[:, (df.nunique() / max(len(df), 1)) >= 0.2]
    if df.shape[1] < MIN_NUM_COLS:
        return None
    corr = df.corr(method="pearson").to_numpy()
    n = corr.shape[0]
    vals = corr[np.triu_indices(n, k=1)]
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None
    hist = np.histogram(vals, bins=CORR_EDGES)[0].astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else hist


def _load_tabicl_corr_hists() -> np.ndarray:
    if TABICL_CORR_CACHE.exists():
        arr = np.load(TABICL_CORR_CACHE)["hists"]
        if arr.shape[1] == N_CORR_BINS:
            print(f"  Corr hists: {len(arr)} TabICL tables from cache")
            return arr
        print(f"  Corr hists: cache has {arr.shape[1]} bins (expected {N_CORR_BINS}), recomputing …")

    if not ICL_DIR.exists():
        raise FileNotFoundError(f"TabICL parquet dir not found: {ICL_DIR}")

    all_files = sorted(ICL_DIR.glob("*.parquet"))
    _rng = np.random.default_rng(RANDOM_STATE)
    if len(all_files) > 10_000:
        all_files = _rng.choice(all_files, size=10_000, replace=False).tolist()

    hists = []
    for p in tqdm(all_files, desc="TabICL corr hists"):
        try:
            df = pd.read_parquet(p)
            num = df.select_dtypes(include="number")
            num = num.loc[:, (num.nunique() / max(len(num), 1)) >= 0.2]
            if num.shape[1] < MIN_NUM_COLS:
                continue
            h = _corr_pmf(num.to_numpy())
            if h is not None:
                hists.append(h)
        except Exception:
            pass

    arr = np.array(hists) if hists else np.zeros((0, N_CORR_BINS), dtype=np.float32)
    TABICL_CORR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TABICL_CORR_CACHE, hists=arr)
    print(f"  Corr hists: {len(arr)} TabICL tables computed + cached")
    return arr


def _load_corr_panel(rank_map, rel_auc_map, rel_best_auc_map,
                     tabicl_auc_map, mean_auc_map, datasets, data_dir, rng,
                     cache_prefix: str = "tabarena"):
    corr_cache = CACHE / f"{cache_prefix}_corr_hists.npz"

    if corr_cache.exists():
        d = np.load(corr_cache, allow_pickle=True)
        if d["hists"].shape[1] == N_CORR_BINS:
            arena_hists = d["hists"]
            names       = d["names"].tolist()
            print(f"  Corr hists: {len(arena_hists)} TabArena tables from cache")
        else:
            print(f"  Corr hists: cache bin mismatch, recomputing …")
            arena_hists, names = None, None
    else:
        arena_hists, names = None, None

    if arena_hists is None:
        hists, ns = [], []
        for ds in tqdm(datasets, desc="TabArena corr hists"):
            df = _load_dataset_df(ds, data_dir)
            if df is None:
                continue
            h = _corr_pmf(df.iloc[:MAX_ROWS].to_numpy())
            if h is not None:
                hists.append(h)
                ns.append(ds)
        arena_hists = np.array(hists)
        names = ns
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(corr_cache, hists=arena_hists, names=np.array(names))
        print(f"  Corr hists: {len(arena_hists)} tables cached -> {corr_cache}")

    tabicl_hists = _load_tabicl_corr_hists()
    idx = rng.choice(len(tabicl_hists), size=min(N_ICL, len(tabicl_hists)), replace=False)
    tabicl_hists = tabicl_hists[idx]
    print(f"  Corr hists: {len(tabicl_hists)} TabICL tables as reference")

    mask      = np.array([n in rank_map for n in names])
    hists_m   = arena_hists[mask]
    matched   = np.array(names)[mask]
    ranks         = np.array([rank_map[n]                     for n in matched], dtype=float)
    rel_aucs      = np.array([rel_auc_map.get(n, np.nan)      for n in matched], dtype=float)
    rel_best_aucs = np.array([rel_best_auc_map.get(n, np.nan) for n in matched], dtype=float)
    tabicl_aucs   = np.array([tabicl_auc_map.get(n, np.nan)   for n in matched], dtype=float)
    mean_aucs     = np.array([mean_auc_map.get(n, np.nan)      for n in matched], dtype=float)

    nn_dists = _nn_distances(hists_m, tabicl_hists, KNN_K)
    half = len(tabicl_hists) // 2
    nn_self = _nn_distances(tabicl_hists[:half], tabicl_hists[half:], KNN_K)
    print(f"  Corr hists: {mask.sum()} tables matched to ranks")
    return nn_dists, nn_self, ranks, rel_aucs, rel_best_aucs, tabicl_aucs, mean_aucs


# ------------------------------------------------------------------
# Figures (identical logic to the TALENT version)
# ------------------------------------------------------------------

def plot_combined(panels: list[dict], output_path: Path, n_bins: int = 30,
                  metric_label: str = "Mean rank",
                  y_label: str = "TabICL mean rank",
                  invert_y: bool = True,
                  query_label: str = "TabArena datasets",
                  clip_lo: float | None = None) -> None:
    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True,
                                 constrained_layout=False)
        fig.subplots_adjust(left=0.07, right=0.93, bottom=0.26, top=0.91,
                            wspace=0.08)

        right_axes = []
        for col, (ax1, p) in enumerate(zip(axes, panels)):
            bins, bin_centers, arena_density, means, stds, x_lo, x_hi = \
                _bin_metric(p["nn_dists"], p["ranks"], n_bins)

            si_idx     = np.clip(np.digitize(p["nn_self"], bins) - 1, 0, n_bins - 1)
            si_cts     = np.array([(si_idx == i).sum() for i in range(n_bins)])
            si_density = si_cts / si_cts.max()

            _stairs(ax1, arena_density, bins, _BLUE,   query_label)
            _stairs(ax1, si_density,    bins, _PURPLE, "TabICL Prior (self)")
            ax1.set_xlim(x_lo, x_hi)
            ax1.set_xlabel("Mean k-NN distance (normalised)")
            if col == 0:
                ax1.set_ylabel("Relative density")
            ax1.set_title(p["title"])

            ax2 = ax1.twinx()
            right_axes.append(ax2)
            mask = ~np.isnan(means)
            lo = means - stds if clip_lo is None else np.clip(means - stds, clip_lo, None)
            hi = means + stds
            ax2.plot(bin_centers[mask], means[mask],
                     color=_RED, lw=1.5, marker="o", ms=3,
                     label=metric_label)
            ax2.fill_between(bin_centers[mask], lo[mask], hi[mask],
                             color=_RED, alpha=0.2)
            if invert_y:
                ax2.invert_yaxis()

            ax2.set_ylabel(y_label, color=_RED,
                           visible=(col == len(panels) - 1))
            ax2.yaxis.set_major_locator(
                MaxNLocator(integer=invert_y, nbins=5, prune="upper"))
            ax2.tick_params(axis="y", labelcolor=_RED,
                            labelleft=False, labelright=False,
                            direction="in")

            if mask.sum() >= 2:
                r = np.corrcoef(bin_centers[mask], means[mask])[0, 1]
                right_axes[-1]._corr_r = r

        h_left, l_left = axes[0].get_legend_handles_labels()
        h_right = [right_axes[0].get_legend_handles_labels()[0][0]]
        l_right = [metric_label]
        fig.legend(
            h_left + h_right, l_left + l_right,
            loc="lower center", ncols=3,
            bbox_to_anchor=(0.5, 0.01), frameon=True, borderaxespad=0.5,
        )

        fig.canvas.draw()
        _bbox = dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=1.0)
        _trans_yd = mpl.transforms.blended_transform_factory
        for ax2 in right_axes:
            trans = _trans_yd(ax2.transAxes, ax2.transData)
            ylim  = sorted(ax2.get_ylim())
            for loc in ax2.yaxis.get_majorticklocs():
                if ylim[0] <= loc <= ylim[1]:
                    label = str(int(round(loc))) if invert_y else f"{loc:.2f}"
                    ax2.text(0.97, loc, label,
                             transform=trans, ha="right", va="center",
                             fontsize=7, color=_RED, zorder=10, bbox=_bbox)
            if hasattr(ax2, "_corr_r"):
                ax2.text(0.5, 0.03, f"$r = {ax2._corr_r:.2f}$",
                         transform=ax2.transAxes, va="bottom", ha="center",
                         fontsize=7, color="black", zorder=10, bbox=_bbox)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {output_path}")


def plot_auc_kde(panels: list[dict], output_path: Path) -> None:
    """3-panel KDE: x = kNN distance to ICL prior, y = AUC."""
    from scipy.stats import gaussian_kde, pearsonr, spearmanr

    _ORANGE = "#e67e22"
    _N_GRID = 200

    print("\n=== AUC vs kNN distance correlations ===")

    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True,
                                 constrained_layout=False)
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.26, top=0.91,
                            wspace=0.08)

        for col, (ax, p) in enumerate(zip(axes, panels)):
            nn   = p["nn_dists"]
            ticl = p["tabicl_aucs"]
            base = p["mean_aucs"]

            valid = ~(np.isnan(ticl) | np.isnan(base) | np.isnan(nn))
            nn, ticl, base = nn[valid], ticl[valid], base[valid]

            x_grid = np.linspace(nn.min(), nn.max(), _N_GRID)

            for vals, color, label in [
                (base, _ORANGE, "Mean baseline AUC"),
                (ticl, _BLUE,   "TabICL AUC"),
            ]:
                xy  = np.vstack([nn, vals])
                kde = gaussian_kde(xy, bw_method="scott")
                xg, yg = np.meshgrid(
                    np.linspace(nn.min(),   nn.max(),   _N_GRID),
                    np.linspace(vals.min(), vals.max(), _N_GRID),
                )
                z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(_N_GRID, _N_GRID)
                ax.contourf(xg, yg, z, levels=6, colors=[color], alpha=0.12)
                ax.contour(xg, yg, z, levels=6, colors=[color], linewidths=0.6, alpha=0.7)
                ax.scatter([], [], color=color, s=12, label=label)

                z1 = np.polyfit(nn, vals, 1)
                ax.plot(x_grid, np.polyval(z1, x_grid),
                        color=color, lw=1.2, linestyle="--", alpha=0.9)

            pr_ticl, _ = pearsonr(nn, ticl)
            sr_ticl, _ = spearmanr(nn, ticl)
            pr_base, _ = pearsonr(nn, base)
            sr_base, _ = spearmanr(nn, base)

            print(f"  {p['title']}")
            print(f"    TabICL AUC    — Pearson r={pr_ticl:+.3f}  Spearman ρ={sr_ticl:+.3f}")
            print(f"    Baseline AUC  — Pearson r={pr_base:+.3f}  Spearman ρ={sr_base:+.3f}")

            _bbox = dict(boxstyle="square,pad=0.15", facecolor="white",
                         edgecolor="none", alpha=0.85)
            ax.text(0.97, 0.97,
                    f"$r_{{\\rm ICL}}={pr_ticl:+.2f}$\n$r_{{\\rm base}}={pr_base:+.2f}$",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6, color="black", bbox=_bbox)

            ax.set_xlabel("Mean k-NN distance (normalised)")
            if col == 0:
                ax.set_ylabel("AUC")
            ax.set_title(p["title"])

        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncols=2,
                   bbox_to_anchor=(0.5, 0.01), frameon=True, borderaxespad=0.5)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {output_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    results_group = parser.add_mutually_exclusive_group(required=True)
    results_group.add_argument("--results-csv", type=Path,
                               help="TabArena results CSV (columns: "
                                    "dataset, method, metric_error, fold, seed)")
    results_group.add_argument("--download-results", action="store_true",
                               help="Download results via the tabarena Python package")

    parser.add_argument("--arena-data-dir", type=Path,
                        default=_ROOT / "data/tabarena",
                        help="Directory with one {dataset}.csv/.parquet per dataset "
                             "(default: data/tabarena)")
    parser.add_argument("--download-datasets", action="store_true",
                        help="Download missing datasets from OpenML into --arena-data-dir")
    parser.add_argument("--output-dir", type=Path,
                        default=_ROOT / "results/rank-coverage-tabarena")
    parser.add_argument("--n-bins", type=int, default=30)
    parser.add_argument("--cache-prefix", type=str, default="tabarena",
                        help="Prefix for cache files (default: tabarena)")
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_STATE)

    results_csv = args.results_csv if args.results_csv else _download_tabarena_results()
    rank_map, rel_auc_map, rel_best_auc_map, tabicl_auc_map, mean_auc_map, datasets = \
        _load_rankings(results_csv)

    if args.download_datasets:
        print(f"\n=== Downloading missing datasets from OpenML -> {args.arena_data_dir} ===")
        task_map = _get_task_map()
        for ds in tqdm(datasets, desc="OpenML downloads"):
            _download_dataset_openml(ds, args.arena_data_dir, task_map)

    print("\n=== Panel 1: table feature vectors ===")
    nn1, self1, rk1, ra1, rb1, ticl1, base1 = _load_table_panel(
        rank_map, rel_auc_map, rel_best_auc_map, tabicl_auc_map, mean_auc_map,
        datasets, args.arena_data_dir, rng, args.cache_prefix)

    print("\n=== Panel 2: column histograms ===")
    nn2, self2, rk2, ra2, rb2, ticl2, base2 = _load_column_panel(
        rank_map, rel_auc_map, rel_best_auc_map, tabicl_auc_map, mean_auc_map,
        datasets, args.arena_data_dir, rng, args.cache_prefix)

    print("\n=== Panel 3: correlation histograms ===")
    nn3, self3, rk3, ra3, rb3, ticl3, base3 = _load_corr_panel(
        rank_map, rel_auc_map, rel_best_auc_map, tabicl_auc_map, mean_auc_map,
        datasets, args.arena_data_dir, rng, args.cache_prefix)

    titles = ["Table features", "Column histograms", "Correlation histograms"]

    panels_rank = [
        dict(nn_dists=nn, nn_self=ns, ranks=rk, title=t)
        for nn, ns, rk, t in zip(
            [nn1, nn2, nn3], [self1, self2, self3], [rk1, rk2, rk3], titles)
    ]
    panels_rel_auc = [
        dict(nn_dists=nn, nn_self=ns, ranks=ra, title=t)
        for nn, ns, ra, t in zip(
            [nn1, nn2, nn3], [self1, self2, self3], [ra1, ra2, ra3], titles)
    ]
    panels_rel_best = [
        dict(nn_dists=nn, nn_self=ns, ranks=rb, title=t)
        for nn, ns, rb, t in zip(
            [nn1, nn2, nn3], [self1, self2, self3], [rb1, rb2, rb3], titles)
    ]
    panels_auc_scatter = [
        dict(nn_dists=nn, tabicl_aucs=ticl, mean_aucs=base, title=t)
        for nn, ticl, base, t in zip(
            [nn1, nn2, nn3], [ticl1, ticl2, ticl3], [base1, base2, base3], titles)
    ]

    print("\n=== Generating figures ===")
    plot_combined(
        panels_rank,
        args.output_dir / "rank_vs_coverage_combined.png",
        args.n_bins,
        metric_label="Mean rank",
        y_label="TabICL mean rank",
        invert_y=True,
        clip_lo=1,
    )
    plot_combined(
        panels_rel_auc,
        args.output_dir / "rel_auc_vs_coverage_combined.png",
        args.n_bins,
        metric_label="Rel. AUC (vs mean)",
        y_label="TabICL AUC / mean AUC",
        invert_y=False,
    )
    plot_combined(
        panels_rel_best,
        args.output_dir / "rel_best_auc_vs_coverage_combined.png",
        args.n_bins,
        metric_label="Rel. AUC (vs best)",
        y_label="TabICL AUC / best AUC",
        invert_y=False,
    )
    plot_auc_kde(
        panels_auc_scatter,
        args.output_dir / "auc_scatter_vs_coverage.png",
    )


if __name__ == "__main__":
    main()
