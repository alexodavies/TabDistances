"""Aggregate feature extraction from DataFrames.

Converts a ``pd.DataFrame`` into a fixed-length numpy vector suitable for
tabular classifiers (used by :class:`AggregateDiscriminator`).

Features are chosen to be invariant to per-column normalisation — they
capture distributional *shape* and inter-column *structure* rather than
location or scale.

Feature layout (70 dimensions)
--------------------------------
0       n_cols            — number of columns
1       type_ratio        — fraction of categorical columns
2       missing_rate      — fraction of missing values across all cells
3       mean_col_skew     — mean of per-column skewness (numeric cols)
4       std_col_skew      — std  of per-column skewness
5       mean_col_kurt     — mean of per-column kurtosis (numeric cols)
6       std_col_kurt      — std  of per-column kurtosis
7       mean_abs_corr     — mean absolute pairwise Pearson correlation
8       std_abs_corr      — std  of absolute pairwise correlations
9       frac_high_corr    — fraction of column pairs with |r| > 0.7
10..39  histogram_mean    — mean of each 30-bin cumulative histogram across numeric columns only
40..69  histogram_std     — std  of each 30-bin cumulative histogram across numeric columns only
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from tabdistances.columns.column_features import get_histogram
from tabdistances.columns.typing import ColumnType, infer_col_types

_N_HIST_BINS = 30
_N_SCALAR    = 10
FEATURE_DIM  = _N_SCALAR + 2 * _N_HIST_BINS   # 70


def df_to_vec(df: pd.DataFrame) -> np.ndarray:
    """Extract a fixed-length aggregate feature vector from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    np.ndarray, shape (FEATURE_DIM,)
    """
    if df.empty or df.shape[1] == 0:
        return np.zeros(FEATURE_DIM)

    col_types  = infer_col_types(df)
    n_cols     = len(col_types)
    n_cat      = sum(1 for ct in col_types.values() if ct == ColumnType.CATEGORICAL)
    type_ratio = n_cat / n_cols if n_cols > 0 else 0.0

    missing_rate = float(df.isnull().values.mean())

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 0:
        mean_col_skew = float(num_df.skew().fillna(0.0).mean())
        std_col_skew  = float(num_df.skew().fillna(0.0).std()) if num_df.shape[1] > 1 else 0.0
        mean_col_kurt = float(num_df.kurt().fillna(0.0).mean())
        std_col_kurt  = float(num_df.kurt().fillna(0.0).std()) if num_df.shape[1] > 1 else 0.0

        if num_df.shape[1] > 1:
            corr = num_df.corr().values
            mask = ~np.eye(corr.shape[0], dtype=bool)
            abs_corrs     = np.abs(corr[mask])
            valid         = abs_corrs[~np.isnan(abs_corrs)]
            mean_abs_corr = float(valid.mean()) if len(valid) else 0.0
            std_abs_corr  = float(valid.std())  if len(valid) > 1 else 0.0
            frac_high_corr = float((valid > 0.7).mean()) if len(valid) else 0.0
        else:
            mean_abs_corr = std_abs_corr = frac_high_corr = 0.0
    else:
        mean_col_skew = std_col_skew = mean_col_kurt = std_col_kurt = 0.0
        mean_abs_corr = std_abs_corr = frac_high_corr = 0.0

    num_cols = [col for col, ct in col_types.items() if ct != ColumnType.CATEGORICAL]
    if num_cols:
        histograms = np.stack([get_histogram(df[col]) for col in num_cols])  # (n_num, 30)
        hist_mean  = histograms.mean(axis=0)
        hist_std   = histograms.std(axis=0) if len(num_cols) > 1 else np.zeros(_N_HIST_BINS)
    else:
        hist_mean = hist_std = np.zeros(_N_HIST_BINS)

    return np.concatenate([
        [n_cols, type_ratio, missing_rate,
         mean_col_skew, std_col_skew,
         mean_col_kurt, std_col_kurt,
         mean_abs_corr, std_abs_corr, frac_high_corr],
        hist_mean,
        hist_std,
    ])


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------

#: Human-readable names for the 70-dimensional feature vector produced by df_to_vec.
FEATURE_NAMES: list[str] = [
    "n_cols",
    "type_ratio",
    "missing_rate",
    "mean_col_skew",
    "std_col_skew",
    "mean_col_kurt",
    "std_col_kurt",
    "mean_abs_corr",
    "std_abs_corr",
    "frac_high_corr",
    *[f"hist_mean[{i}]" for i in range(_N_HIST_BINS)],
    *[f"hist_std[{i}]"  for i in range(_N_HIST_BINS)],
]

#: Named subsets of feature indices for use with :func:`select_features`.
#: Index 0 (n_cols) is excluded from all modes — it is used for conditioning
#: (column-count matching) but not as a discriminative feature.
FEATURE_MODES: dict[str, list[int]] = {
    "full":        list(range(1, FEATURE_DIM)),        # all distributional features
    "scalar_only": list(range(1, _N_SCALAR)),          # 9 scalar moments / correlations
    "hist_only":   list(range(_N_SCALAR, FEATURE_DIM)),# 60 histogram features
}


def featurize_tables(
    tables: Iterable[tuple[int, str, pd.DataFrame]],
    cache_path: Path | None = None,
    max_rows: int = 512,
    desc: str = "Featurising tables",
) -> tuple[list[str], np.ndarray]:
    """Compute :func:`df_to_vec` for each table, with optional disk caching.

    Loads previously cached feature vectors from *cache_path* (if it exists)
    and only computes the missing ones.  The updated cache is written back
    automatically.

    Parameters
    ----------
    tables : iterable of (index, name, df)
        Any ``iter_tables()``-compatible source — e.g. ``TabFMDataset``,
        ``T4Dataset``, or ``iter_prior_tables()``.
    cache_path : Path or None
        Path to a ``.npz`` file for caching.  If ``None``, results are not
        cached to disk.
    max_rows : int
        Cap each table at this many rows before featurisation.
    desc : str
        Label shown on the tqdm progress bar.

    Returns
    -------
    names : list[str]
        Table identifiers, one per row of *feats*.
    feats : np.ndarray, shape (n, FEATURE_DIM)
        Aggregate feature matrix.

    Examples
    --------
    >>> from tabdistances import TabFMDataset, featurize_tables
    >>> from tabdistances.config import Paths
    >>> names, feats = featurize_tables(
    ...     TabFMDataset().iter_tables(),
    ...     cache_path=Paths.CACHE / "tabfm_feats.npz",
    ... )
    >>> feats.shape
    (n_tables, 70)
    """
    # ------------------------------------------------------------------ load cache
    cached: dict[str, np.ndarray] = {}
    if cache_path is not None and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        cached = {name: data["feats"][i] for i, name in enumerate(data["names"].tolist())}

    # ------------------------------------------------------------------ compute missing
    new_names: list[str] = []
    new_feats: list[np.ndarray] = []
    all_names = list(cached.keys())
    cached_set = set(all_names)

    for _, name, df in tqdm(tables, desc=desc):
        if name in cached_set:
            continue
        feat = df_to_vec(df.iloc[:max_rows])
        cached[name] = feat
        new_names.append(name)
        new_feats.append(feat)

    # ------------------------------------------------------------------ update cache
    if cache_path is not None and new_names:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            names=np.array(list(cached.keys())),
            feats=np.array(list(cached.values())),
        )

    names = list(cached.keys())
    feats = np.array(list(cached.values())) if names else np.zeros((0, FEATURE_DIM))
    return names, feats


def select_features(
    feats: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, list[str]]:
    """Return (filtered_feats, filtered_names) for the given feature mode.

    Parameters
    ----------
    feats : np.ndarray, shape (N, FEATURE_DIM)
        Feature matrix from :func:`df_to_vec`.
    mode : str
        One of ``"full"``, ``"scalar_only"``, ``"hist_only"``.

    Returns
    -------
    filtered_feats : np.ndarray, shape (N, len(idx))
    filtered_names : list[str]
    """
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode {mode!r}. Choose from {list(FEATURE_MODES)}")
    idx = FEATURE_MODES[mode]
    return feats[:, idx], [FEATURE_NAMES[i] for i in idx]
