import pandas as pd

from tabdistances.columns.typing import ColumnType, infer_col_type, infer_col_types


def test_float64_is_numeric():
    assert infer_col_type(pd.Series([1.0, 2.0, 3.0], dtype="float64")) == ColumnType.NUMERIC


def test_int32_is_categorical():
    assert infer_col_type(pd.Series([1, 2, 3], dtype="int32")) == ColumnType.CATEGORICAL


def test_int64_is_categorical():
    assert infer_col_type(pd.Series([1, 2, 3], dtype="int64")) == ColumnType.CATEGORICAL


def test_infer_col_types_returns_dict():
    df = pd.DataFrame({"a": pd.array([1.0, 2.0], dtype="float64"), "b": pd.array([1, 2], dtype="int32")})
    types = infer_col_types(df)
    assert types["a"] == ColumnType.NUMERIC
    assert types["b"] == ColumnType.CATEGORICAL
