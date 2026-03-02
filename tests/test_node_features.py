import numpy as np
import pandas as pd

from tabdistances.columns.typing import ColumnType
from tabdistances.columns.node_features import get_type_int, get_histogram


def test_numeric_type_int():
    np.testing.assert_array_equal(get_type_int(ColumnType.NUMERIC), [0.0])


def test_categorical_type_int():
    np.testing.assert_array_equal(get_type_int(ColumnType.CATEGORICAL), [1.0])


def test_type_int_same_dimension():
    assert get_type_int(ColumnType.NUMERIC).shape == get_type_int(ColumnType.CATEGORICAL).shape


def test_type_int_returns_copy():
    f1 = get_type_int(ColumnType.NUMERIC)
    f2 = get_type_int(ColumnType.NUMERIC)
    f1[0] = 99.0
    assert f2[0] == 0.0


def test_histogram_length():
    col = pd.Series(np.linspace(0, 1, 100))
    assert get_histogram(col).shape == (30,)


def test_histogram_sums_to_one():
    col = pd.Series(np.random.default_rng(0).normal(size=200))
    np.testing.assert_almost_equal(get_histogram(col).sum(), 1.0)


def test_histogram_empty_column():
    col = pd.Series([np.nan, np.nan])
    np.testing.assert_array_equal(get_histogram(col), np.zeros(30))
