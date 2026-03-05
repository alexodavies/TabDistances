"""Aggregate feature extraction from DataFrames.

Converts a ``pd.DataFrame`` into a fixed-length numpy vector suitable for
tabular classifiers (used by :class:`AggregateDiscriminator`).

Feature layout (63 dimensions)
--------------------------------
0       n_cols           — number of columns
1       type_ratio       — fraction of categorical columns
2..31   histogram_mean   — mean of each 30-bin cumulative histogram across columns
32..61  histogram_std    — std  of each 30-bin cumulative histogram across columns
62      n_rows           — number of rows (log1p-scaled)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabdistances.columns.node_features import get_histogram
from tabdistances.columns.typing import ColumnType, infer_col_types

_N_HIST_BINS = 30
FEATURE_DIM  = 3 + 2 * _N_HIST_BINS   # 63


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

    histograms = np.stack([get_histogram(df[col]) for col in df.columns])  # (n_cols, 30)
    hist_mean  = histograms.mean(axis=0)   # (30,)
    hist_std   = histograms.std(axis=0)    # (30,)

    return np.concatenate([
        [n_cols, type_ratio],
        hist_mean,
        hist_std,
        [np.log1p(len(df))],
    ])
