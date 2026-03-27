"""2-panel combined histogram + mean-rank figure using pymfe meta-features.

One panel per dataset source:
  (1) TALENT datasets   (ranks from tabicl-rankings-full-context.csv)
  (2) TabArena datasets (ranks from a TabArena results CSV)

Each panel:
  - Left y-axis:  normalised NN-distance density (datasets + TabICL Prior self)
  - Right y-axis: mean TabICL rank ± 1 SD per distance bin (inverted, lower = better)

Meta-features are computed with pymfe (groups: general, statistical, info-theory).
Each dataset is featurised as (X, y) where X are numeric features and y is the target.
For ICL prior tables (parquet), the last column is used as y.
For TALENT datasets, y_*.npy files are used.
For TabArena datasets, the last numeric column is used as y proxy.

Usage
-----
    # Provide a local TabArena results CSV
    python benchmark/rank_vs_coverage_combined_pymfe.py \
        --results-csv /path/to/tabarena_results.csv \
        [--arena-data-dir /path/to/tabarena/datasets] \
        [--output-dir results/rank-coverage-pymfe] \
        [--n-bins 30]

    # Download TabArena results via the tabarena package
    python benchmark/rank_vs_coverage_combined_pymfe.py \
        --download-results \
        [--arena-data-dir /path/to/tabarena/datasets] \
        [--download-datasets]
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

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
_ROOT  = Path(__file__).resolve().parent.parent
CACHE  = _ROOT / "cache"

RANKINGS_CSV  = _ROOT / "results/tabicl-rankings-full-context.csv"
TALENT_DIR    = _ROOT / "data"
ICL_DIR       = _ROOT / "tables-tabicl/parquet"

TABARENA_RESULTS_CACHE = CACHE / "tabarena_results.csv"
TASK_MAP_CACHE         = CACHE / "tabarena_task_map.json"
TABARENA_SUITE         = "tabarena-v0.1"
TABICL_METHOD          = "TABICL_GPU (default)"

PYMFE_ICL_CACHE    = CACHE / "pymfe_icl_feats.npz"
PYMFE_GROUPS       = ["general", "statistical", "info-theory"]

MAX_ROWS      = 512
KNN_K         = 5
N_ICL         = 2_000   # pymfe vectors are richer than histograms; smaller pool suffices
RANDOM_STATE  = 42
PYMFE_TIMEOUT = 60      # seconds per dataset before skipping
N_JOBS        = -1      # parallel workers; -1 = os.cpu_count()

_PAPER_RC = {
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "lines.linewidth":  1.5,
    "figure.dpi":       300,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor": "white",
}

_BLUE   = "#2980b9"
_PURPLE = "#8e44ad"
_RED    = "#ef0e0e"


# ------------------------------------------------------------------
# pymfe featurisation
# ------------------------------------------------------------------

def _pymfe_vec(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Compute pymfe meta-feature vector from X (2-D) and y (1-D).

    Returns a 1-D float array (NaN entries for features that could not be
    computed), or None if pymfe itself raises an unrecoverable error.
    List-valued features are summarised via mean and sd (pymfe default).
    """
    try:
        from pymfe.mfe import MFE
    except ImportError:
        raise ImportError(
            "pymfe is required: pip install pymfe"
        )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    # pymfe needs at least 2 rows and 1 column
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        return None

    try:
        mfe = MFE(groups=PYMFE_GROUPS, random_state=RANDOM_STATE)
        mfe.fit(X, y)
        _, vals = mfe.extract()
        arr = np.array(vals, dtype=float)
        return arr
    except Exception:
        return None


def _align_pymfe_vecs(vecs: list[np.ndarray]) -> np.ndarray:
    """Stack a list of 1-D pymfe vectors, padding / truncating to the modal
    length so that the result is a 2-D array.  Any position that is out of
    range for a shorter vector is filled with NaN."""
    lengths = [len(v) for v in vecs]
    # use the most common length (should be identical for a fixed MFE config)
    from collections import Counter
    target_len = Counter(lengths).most_common(1)[0][0]
    out = np.full((len(vecs), target_len), np.nan, dtype=float)
    for i, v in enumerate(vecs):
        n = min(len(v), target_len)
        out[i, :n] = v[:n]
    return out


# ------------------------------------------------------------------
# Parallel workers  (module-level so they are picklable)
# ------------------------------------------------------------------

def _icl_path_worker(path_str: str) -> np.ndarray | None:
    """Read one ICL parquet file and return its pymfe vector (or None)."""
    try:
        df = pd.read_parquet(path_str).select_dtypes(include="number").iloc[:MAX_ROWS]
        if df.shape[0] < 2 or df.shape[1] < 2:
            return None
        return _pymfe_vec(df.iloc[:, :-1].to_numpy(dtype=float),
                          df.iloc[:, -1].to_numpy())
    except Exception:
        return None


def _xy_worker(args: tuple) -> np.ndarray | None:
    """Compute pymfe vector from a (X, y) tuple."""
    X, y = args
    return _pymfe_vec(X, y)


def _run_parallel(fn, items, desc: str) -> list:
    """Run fn(item) in parallel with PYMFE_TIMEOUT per item.

    Returns a list of results in the same order as *items*; failed or timed-out
    items yield None.
    """
    import os
    from multiprocessing import Pool

    workers = os.cpu_count() if N_JOBS < 0 else N_JOBS
    results = [None] * len(items)
    with Pool(processes=workers) as pool:
        handles = [(i, pool.apply_async(fn, (item,)))
                   for i, item in enumerate(items)]
        for i, ar in tqdm(handles, desc=desc):
            try:
                results[i] = ar.get(timeout=PYMFE_TIMEOUT)
            except Exception:
                pass
    return results


# ------------------------------------------------------------------
# Shared utilities
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


def _trim_distances(nn_dists: np.ndarray, ranks: np.ndarray,
                    nn_self: np.ndarray,
                    lo_pct: float = 0.0, hi_pct: float = 95.0,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip to the [lo_pct, hi_pct] percentile range of nn_dists.

    nn_self is filtered to the same window so both histograms share the x-axis.
    """
    lo, hi = np.percentile(nn_dists, [lo_pct, hi_pct])
    mask = (nn_dists >= lo) & (nn_dists <= hi)
    self_mask = (nn_self >= lo) & (nn_self <= hi)
    return nn_dists[mask], ranks[mask], nn_self[self_mask]


def _bin_metric(nn_dists: np.ndarray, ranks: np.ndarray,
                n_bins: int) -> tuple:
    """Bin pre-trimmed distances into n_bins equal-width bins."""
    bins = np.linspace(nn_dists.min(), nn_dists.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    idx  = np.clip(np.digitize(nn_dists, bins) - 1, 0, n_bins - 1)
    cts  = np.array([(idx == i).sum() for i in range(n_bins)])
    density = cts.astype(float)
    density /= density.max() if density.max() > 0 else 1.0
    means = np.array([np.nanmean(ranks[idx == i]) if cts[i] > 0 else np.nan for i in range(n_bins)])
    stds  = np.array([np.nanstd(ranks[idx == i])  if cts[i] > 1 else 0.0   for i in range(n_bins)])
    return bins, bin_centers, density, means, stds


# ------------------------------------------------------------------
# ICL prior pymfe cache
# ------------------------------------------------------------------

def _build_icl_pymfe_cache(rng: np.random.Generator) -> np.ndarray:
    """Featurise TabICL parquet tables with pymfe, cache, and return array.

    Each parquet file is treated as (X = all-but-last numeric column, y = last
    numeric column), truncated to MAX_ROWS rows.
    """
    if not ICL_DIR.exists():
        raise FileNotFoundError(
            f"TabICL parquet dir not found: {ICL_DIR}\n"
            "Run:  python fetch_data.py --icl"
        )
    paths = np.array(sorted(ICL_DIR.glob("*.parquet")))
    rng.shuffle(paths)
    # Only dispatch as many files as needed (with some headroom for failures)
    candidate_paths = [str(p) for p in paths[:N_ICL * 2]]
    raw = _run_parallel(_icl_path_worker, candidate_paths, desc="pymfe: ICL prior tables")
    feats = [v for v in raw if v is not None][:N_ICL]
    print(f"  pymfe ICL: {len(feats)} tables featurised ({len(raw) - len(feats)} skipped/failed)")
    arr = _align_pymfe_vecs(feats)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(PYMFE_ICL_CACHE, feats=arr)
    print(f"  Cached -> {PYMFE_ICL_CACHE}")
    return arr


def _load_icl_pymfe(rng: np.random.Generator) -> np.ndarray:
    if PYMFE_ICL_CACHE.exists():
        arr = np.load(PYMFE_ICL_CACHE)["feats"]
        print(f"  pymfe ICL: {len(arr)} vectors from cache")
        return arr
    return _build_icl_pymfe_cache(rng)


# ------------------------------------------------------------------
# TALENT rankings
# ------------------------------------------------------------------

def _load_talent_rankings() -> tuple[dict, list[str]]:
    """Returns (rank_map, datasets) from tabicl-rankings-full-context.csv."""
    df = pd.read_csv(RANKINGS_CSV)
    df["AUC"] = df["AUC"].astype(float)
    tc = df[df["model"] == "tabicl"][["dataset", "rank"]].copy()
    tc["rank"] = tc["rank"].astype(int)
    rank_map = dict(zip(tc["dataset"], tc["rank"]))
    print(f"TALENT rankings: {len(rank_map)} datasets")
    return rank_map, tc["dataset"].tolist()


# ------------------------------------------------------------------
# TabArena rankings + dataset download
# ------------------------------------------------------------------

def _download_tabarena_results() -> Path:
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
    import tabarena as _ta
    if _ta.__file__ is None:
        import tabarena.nips2025_utils.tabarena_context as _ctx_mod
        _ta.__file__ = str(Path(_ctx_mod.__file__).parent.parent / "__init__.py")

    print("  Downloading TabArena results …")
    ctx = TabArenaContext()
    df = ctx.load_results_paper(download_results="auto")
    if "framework" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"framework": "method"})
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABARENA_RESULTS_CACHE, index=False)
    print(f"  Saved to {TABARENA_RESULTS_CACHE}")
    return TABARENA_RESULTS_CACHE


def _load_tabarena_rankings(results_csv: Path) -> tuple[dict, list[str]]:
    """Returns (rank_map, datasets) from TabArena results CSV."""
    df = pd.read_csv(results_csv)
    agg = (df.groupby(["dataset", "method"])["metric_error"]
             .mean()
             .reset_index())
    agg["rank"] = (agg.groupby("dataset")["metric_error"]
                      .rank(method="min", ascending=True))
    tc = agg[agg["method"] == TABICL_METHOD][["dataset", "rank"]].copy()
    if tc.empty:
        raise ValueError(
            f"No rows found for method '{TABICL_METHOD}' in {results_csv}. "
            f"Available methods: {sorted(agg['method'].unique().tolist())}"
        )
    rank_map = dict(zip(tc["dataset"], tc["rank"].astype(int)))
    print(f"TabArena rankings: {len(rank_map)} datasets (method: '{TABICL_METHOD}')")
    return rank_map, tc["dataset"].tolist()


def _get_task_map() -> dict[str, int]:
    import json, openml  # type: ignore
    if TASK_MAP_CACHE.exists():
        with open(TASK_MAP_CACHE) as f:
            return json.load(f)
    print(f"  Fetching TabArena task list from OpenML suite '{TABARENA_SUITE}' …")
    suite = openml.study.get_suite(TABARENA_SUITE)
    task_map: dict[str, int] = {}
    for task_id in tqdm(suite.tasks, desc="Resolving task names"):
        try:
            task = openml.tasks.get_task(task_id, download_data=False)
            ds = task.get_dataset()
            task_map[ds.name] = task_id
        except Exception as e:
            print(f"    Warning: could not resolve task {task_id}: {e}")
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(TASK_MAP_CACHE, "w") as f:
        import json
        json.dump(task_map, f, indent=2)
    print(f"  Task map cached ({len(task_map)} datasets) -> {TASK_MAP_CACHE}")
    return task_map


def _download_dataset_openml(dataset: str, data_dir: Path,
                              task_map: dict[str, int] | None = None) -> bool:
    out = data_dir / f"{dataset}.parquet"
    if out.exists():
        return True
    try:
        import openml  # type: ignore
    except ImportError:
        raise ImportError("pip install openml")
    if task_map is None:
        task_map = _get_task_map()
    task_id = task_map.get(dataset)
    if task_id is None:
        print(f"    OpenML: '{dataset}' not in TabArena suite, skipping")
        return False
    try:
        task = openml.tasks.get_task(task_id)
        ds = task.get_dataset()
        X, _, _, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
        num = X.select_dtypes(include="number")
        data_dir.mkdir(parents=True, exist_ok=True)
        num.to_parquet(out, index=False)
        return True
    except Exception as e:
        print(f"    OpenML: failed to download '{dataset}' (task {task_id}): {e}")
        return False


# ------------------------------------------------------------------
# TALENT panel
# ------------------------------------------------------------------

def _load_talent_panel(rank_map: dict, datasets: list[str],
                       rng: np.random.Generator) -> tuple:
    """Returns (nn_dists, nn_self, ranks) for TALENT datasets."""
    cache_path = CACHE / "pymfe_talent_feats.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached = {n: data["feats"][i]
                  for i, n in enumerate(data["names"].tolist())
                  if n in set(datasets)}
        print(f"  pymfe TALENT: cache hit {len(cached)}/{len(datasets)} datasets")

    missing = [ds for ds in datasets if ds not in cached]
    if missing:
        print(f"  pymfe TALENT: computing {len(missing)} missing datasets …")
        # Pre-load arrays in the main process (small, bounded by MAX_ROWS)
        xy_items, valid_ds = [], []
        for ds in missing:
            ds_path = TALENT_DIR / ds
            if not ds_path.exists():
                continue
            X_parts, y_parts = [], []
            for split in ("train", "val", "test"):
                npy_x = ds_path / f"N_{split}.npy"
                npy_y = ds_path / f"y_{split}.npy"
                if npy_x.exists():
                    arr = np.load(npy_x, allow_pickle=True).astype(float)
                    X_parts.append(arr.reshape(-1, 1) if arr.ndim == 1 else arr)
                    if npy_y.exists():
                        y_parts.append(np.load(npy_y, allow_pickle=True).ravel())
            if not X_parts:
                continue
            X = np.concatenate(X_parts, axis=0)[:MAX_ROWS]
            y = np.concatenate(y_parts)[:MAX_ROWS] if y_parts else X[:, -1]
            if not y_parts:
                X = X[:, :-1]
            xy_items.append((X, y))
            valid_ds.append(ds)

        raw = _run_parallel(_xy_worker, xy_items, desc="pymfe: TALENT datasets")
        skipped = sum(1 for v in raw if v is None)
        for ds, v in zip(valid_ds, raw):
            if v is not None:
                cached[ds] = v
        print(f"  pymfe TALENT: computed {len(valid_ds) - skipped} ({len(missing) - len(valid_ds) + skipped} skipped)")
        CACHE.mkdir(parents=True, exist_ok=True)
        vecs = list(cached.values())
        np.savez(cache_path,
                 names=np.array(list(cached.keys())),
                 feats=_align_pymfe_vecs(vecs))
        print(f"  pymfe TALENT: cache updated -> {cache_path}")

    names = [ds for ds in datasets if ds in cached]
    vecs  = [cached[ds] for ds in names]
    if not vecs:
        raise RuntimeError("No TALENT datasets could be featurised with pymfe")
    talent_feats = _align_pymfe_vecs(vecs)
    ranks = np.array([rank_map[n] for n in names], dtype=float)
    print(f"  pymfe TALENT: {len(names)} datasets")

    icl_feats = _load_icl_pymfe(rng)
    # align ICL to same feature width as TALENT
    target_w = talent_feats.shape[1]
    if icl_feats.shape[1] != target_w:
        w = min(icl_feats.shape[1], target_w)
        icl_feats    = icl_feats[:, :w]
        talent_feats = talent_feats[:, :w]

    idx = rng.choice(len(icl_feats), size=min(N_ICL, len(icl_feats)), replace=False)
    icl_sample = icl_feats[idx]

    nn_dists = _nn_distances(talent_feats, icl_sample, KNN_K)
    half     = len(icl_sample) // 2
    nn_self  = _nn_distances(icl_sample[:half], icl_sample[half:], KNN_K)
    return nn_dists, nn_self, ranks


# ------------------------------------------------------------------
# TabArena panel
# ------------------------------------------------------------------

def _load_dataset_df(dataset: str, data_dir: Path) -> pd.DataFrame | None:
    for ext in (".parquet", ".csv"):
        p = data_dir / f"{dataset}{ext}"
        if p.exists():
            df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            return df.select_dtypes(include="number")
    return None


def _load_tabarena_panel(rank_map: dict, datasets: list[str],
                         data_dir: Path, rng: np.random.Generator,
                         cache_prefix: str = "tabarena") -> tuple:
    """Returns (nn_dists, nn_self, ranks) for TabArena datasets."""
    cache_path = CACHE / f"pymfe_{cache_prefix}_feats.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached = {n: data["feats"][i]
                  for i, n in enumerate(data["names"].tolist())
                  if n in set(datasets)}
        print(f"  pymfe TabArena: cache hit {len(cached)}/{len(datasets)} datasets")

    missing = [ds for ds in datasets if ds not in cached]
    if missing:
        print(f"  pymfe TabArena: computing {len(missing)} missing datasets …")
        xy_items, valid_ds = [], []
        for ds in missing:
            df = _load_dataset_df(ds, data_dir)
            if df is None or df.shape[0] < 2 or df.shape[1] < 2:
                continue
            df = df.iloc[:MAX_ROWS]
            xy_items.append((df.iloc[:, :-1].to_numpy(dtype=float),
                             df.iloc[:, -1].to_numpy()))
            valid_ds.append(ds)

        raw = _run_parallel(_xy_worker, xy_items, desc="pymfe: TabArena datasets")
        skipped = sum(1 for v in raw if v is None)
        for ds, v in zip(valid_ds, raw):
            if v is not None:
                cached[ds] = v
        print(f"  pymfe TabArena: computed {len(missing) - skipped} ({skipped} skipped)")
        CACHE.mkdir(parents=True, exist_ok=True)
        vecs = list(cached.values())
        np.savez(cache_path,
                 names=np.array(list(cached.keys())),
                 feats=_align_pymfe_vecs(vecs))
        print(f"  pymfe TabArena: cache updated -> {cache_path}")

    names = [ds for ds in datasets if ds in cached]
    vecs  = [cached[ds] for ds in names]
    if not vecs:
        raise RuntimeError("No TabArena datasets could be featurised with pymfe")
    arena_feats = _align_pymfe_vecs(vecs)
    ranks = np.array([rank_map[n] for n in names], dtype=float)
    print(f"  pymfe TabArena: {len(names)} datasets")

    icl_feats = _load_icl_pymfe(rng)
    target_w = arena_feats.shape[1]
    if icl_feats.shape[1] != target_w:
        w = min(icl_feats.shape[1], target_w)
        icl_feats   = icl_feats[:, :w]
        arena_feats = arena_feats[:, :w]

    idx = rng.choice(len(icl_feats), size=min(N_ICL, len(icl_feats)), replace=False)
    icl_sample = icl_feats[idx]

    nn_dists = _nn_distances(arena_feats, icl_sample, KNN_K)
    half     = len(icl_sample) // 2
    nn_self  = _nn_distances(icl_sample[:half], icl_sample[half:], KNN_K)
    return nn_dists, nn_self, ranks


# ------------------------------------------------------------------
# Figure — 2-panel layout
# ------------------------------------------------------------------

def plot_combined(panels: list[dict], output_path: Path, n_bins: int = 30,
                  metric_label: str = "Mean rank",
                  y_label: str = "TabICL mean rank",
                  invert_y: bool = True) -> None:
    """panels: list of 2 dicts, each with keys nn_dists, nn_self, ranks, title."""
    with mpl.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.6), sharey=True,
                                 constrained_layout=False)
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.26, top=0.91,
                            wspace=0.10)

        right_axes = []
        for col, (ax1, p) in enumerate(zip(axes, panels)):
            nd, rk, ns = _trim_distances(p["nn_dists"], p["ranks"], p["nn_self"])
            bins, bin_centers, density, means, stds = \
                _bin_metric(nd, rk, n_bins)

            si_idx     = np.clip(np.digitize(ns, bins) - 1, 0, n_bins - 1)
            si_cts     = np.array([(si_idx == i).sum() for i in range(n_bins)])
            si_density = si_cts / (si_cts.max() if si_cts.max() > 0 else 1.0)

            _stairs(ax1, density,    bins, _BLUE,   p.get("query_label", "Datasets"))
            _stairs(ax1, si_density, bins, _PURPLE, "TabICL Prior (self)")
            ax1.set_xlim(bins[0], bins[-1])
            ax1.set_xlabel("Mean k-NN distance (normalised)")
            if col == 0:
                ax1.set_ylabel("Relative density")
            ax1.set_title(p["title"])

            ax2 = ax1.twinx()
            right_axes.append(ax2)
            mask = ~np.isnan(means)
            lo = means - stds
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

        # Legend below both panels
        h_left, l_left = axes[0].get_legend_handles_labels()
        h_right = [right_axes[0].get_legend_handles_labels()[0][0]]
        l_right = [metric_label]
        fig.legend(
            h_left + h_right,
            l_left + l_right,
            loc="lower center",
            ncols=3,
            bbox_to_anchor=(0.5, 0.01),
            frameon=True,
            borderaxespad=0.5,
        )

        fig.canvas.draw()
        _bbox = dict(boxstyle="square,pad=0.15", facecolor="white",
                     edgecolor="none", alpha=1.0)
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


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    results_group = parser.add_mutually_exclusive_group(required=False)
    results_group.add_argument("--results-csv", type=Path,
                               help="TabArena results CSV (columns: "
                                    "dataset, method, metric_error, fold, seed). "
                                    f"Defaults to the shared cache at cache/tabarena_results.csv "
                                    f"if it already exists.")
    results_group.add_argument("--download-results", action="store_true",
                               help="Download results via the tabarena Python package")

    parser.add_argument("--arena-data-dir", type=Path,
                        default=_ROOT / "data/tabarena",
                        help="Directory with one {dataset}.csv/.parquet per dataset")
    parser.add_argument("--download-datasets", action="store_true",
                        help="Download missing TabArena datasets from OpenML")
    parser.add_argument("--output-dir", type=Path,
                        default=_ROOT / "results/rank-coverage-pymfe")
    parser.add_argument("--n-bins", type=int, default=30)
    parser.add_argument("--cache-prefix", type=str, default="tabarena")
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_STATE)

    # --- TALENT panel ---
    print("\n=== Panel 1: TALENT datasets (pymfe) ===")
    talent_rank_map, talent_datasets = _load_talent_rankings()
    nn_talent, self_talent, rk_talent = _load_talent_panel(
        talent_rank_map, talent_datasets, rng)

    # --- TabArena panel ---
    print("\n=== Panel 2: TabArena datasets (pymfe) ===")
    if args.results_csv:
        results_csv = args.results_csv
    elif TABARENA_RESULTS_CACHE.exists():
        results_csv = TABARENA_RESULTS_CACHE
        print(f"  TabArena results: using shared cache {TABARENA_RESULTS_CACHE}")
    else:
        results_csv = _download_tabarena_results()
    arena_rank_map, arena_datasets = _load_tabarena_rankings(results_csv)

    if args.download_datasets:
        print(f"\n=== Downloading missing datasets -> {args.arena_data_dir} ===")
        task_map = _get_task_map()
        for ds in tqdm(arena_datasets, desc="OpenML downloads"):
            _download_dataset_openml(ds, args.arena_data_dir, task_map)

    nn_arena, self_arena, rk_arena = _load_tabarena_panel(
        arena_rank_map, arena_datasets,
        args.arena_data_dir, rng, args.cache_prefix)

    # --- Figure ---
    panels = [
        dict(nn_dists=nn_talent, nn_self=self_talent, ranks=rk_talent,
             title="TALENT datasets",
             query_label="TALENT datasets"),
        dict(nn_dists=nn_arena,  nn_self=self_arena,  ranks=rk_arena,
             title="TabArena datasets",
             query_label="TabArena datasets"),
    ]

    print("\n=== Generating figures ===")
    plot_combined(
        panels,
        args.output_dir / "rank_vs_coverage_pymfe.png",
        args.n_bins,
        metric_label="Mean rank",
        y_label="TabICL mean rank",
        invert_y=True,
    )


if __name__ == "__main__":
    main()
