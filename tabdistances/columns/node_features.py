import numpy as np
import pandas as pd

from .typing import ColumnType

_HISTOGRAM_BINS = 30

_TYPE_INT_MAP = {
    ColumnType.NUMERIC: np.array([0.0]),
    ColumnType.CATEGORICAL: np.array([1.0]),
}


def get_type_int(col_type: ColumnType) -> np.ndarray:
    """Return a 1-d vector encoding the column type: NUMERIC -> [0.0], CATEGORICAL -> [1.0]."""
    return _TYPE_INT_MAP[col_type].copy()


def get_histogram(col: pd.Series) -> np.ndarray:
    """Return a normalised 30-bin cumulative histogram of the column values.

    Non-numeric columns are integer-factorized before binning.
    """
    raw = col.dropna().to_numpy()
    try:
        values = raw.astype(float)
    except (ValueError, TypeError):
        values = pd.factorize(raw)[0].astype(float)
    counts, _ = np.histogram(values, bins=_HISTOGRAM_BINS)
    total = counts.sum()
    if total == 0:
        return np.zeros(_HISTOGRAM_BINS)
    return np.cumsum(counts) / total
