"""Edge-case tests for tabdistances.columns.typing."""

import numpy as np
import pandas as pd
import pytest

from tabdistances.columns.typing import ColumnType, infer_col_type, infer_col_types


# ---------------------------------------------------------------------------
# cardinality_threshold reclassification
# ---------------------------------------------------------------------------

def test_low_cardinality_float_reclassified_as_categorical():
    # 5 unique values out of 100 rows → ratio = 0.05 < default 0.2
    col = pd.Series(np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 20).astype("float64"))
    assert infer_col_type(col) == ColumnType.CATEGORICAL


def test_high_cardinality_float_stays_numeric():
    # All unique values → ratio = 1.0 > 0.2
    col = pd.Series(np.linspace(0, 1, 100).astype("float64"))
    assert infer_col_type(col) == ColumnType.NUMERIC


def test_cardinality_threshold_none_disables_reclassification():
    col = pd.Series(np.repeat([1.0, 2.0], 50).astype("float64"))
    # With threshold=None, float64 always → NUMERIC
    assert infer_col_type(col, cardinality_threshold=None) == ColumnType.NUMERIC


# ---------------------------------------------------------------------------
# max_cat_classes reclassification
# ---------------------------------------------------------------------------

def test_high_cardinality_int_reclassified_as_numeric():
    col = pd.Series(range(100), dtype="int64")
    assert infer_col_type(col) == ColumnType.NUMERIC


def test_low_cardinality_int_stays_categorical():
    col = pd.Series([0, 1, 2, 3] * 10, dtype="int64")
    assert infer_col_type(col) == ColumnType.CATEGORICAL


def test_max_cat_classes_none_disables_reclassification():
    col = pd.Series(range(100), dtype="int64")
    assert infer_col_type(col, max_cat_classes=None) == ColumnType.CATEGORICAL


# ---------------------------------------------------------------------------
# String columns
# ---------------------------------------------------------------------------

def test_string_column_is_categorical():
    col = pd.Series(["apple", "banana", "cherry"])
    assert infer_col_type(col) == ColumnType.CATEGORICAL


def test_high_cardinality_string_reclassified_as_numeric():
    col = pd.Series([f"id_{i}" for i in range(50)])
    assert infer_col_type(col) == ColumnType.NUMERIC


# ---------------------------------------------------------------------------
# infer_col_types — full DataFrame
# ---------------------------------------------------------------------------

def test_infer_col_types_all_columns_present():
    df = pd.DataFrame({
        "a": np.linspace(0, 1, 20).astype("float64"),
        "b": list(range(20)),
        "c": ["x", "y"] * 10,
    })
    types = infer_col_types(df)
    assert set(types.keys()) == {"a", "b", "c"}


def test_infer_col_types_empty_dataframe():
    types = infer_col_types(pd.DataFrame())
    assert types == {}
