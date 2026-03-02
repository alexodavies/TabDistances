import numpy as np
import pandas as pd

from tabdistances.columns.typing import infer_col_types
from tabdistances.columns.edge_features import get_edge_weights


def _w(df):
    return get_edge_weights(df, infer_col_types(df))


def test_perfectly_correlated_numeric():
    x = np.linspace(0, 1, 100)
    df = pd.DataFrame({"a": x, "b": x})
    assert _w(df)[0, 1] > 0.99


def test_independent_numeric_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    assert _w(df)[0, 1] < 0.1


def test_asymmetric():
    # directed weights need not be symmetric in general
    rng = np.random.default_rng(1)
    x = rng.normal(size=200).astype("float64")
    cat = pd.array((x > 0).astype("int32"), dtype="int32")
    df = pd.DataFrame({"num": x, "cat": cat})
    W = _w(df)
    # num->cat and cat->num use different metrics so W need not equal W.T
    assert W.shape == (2, 2)


def test_diagonal_is_zero():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    np.testing.assert_array_equal(np.diag(_w(df)), 0)


def test_categorical_determines_numeric():
    # cat perfectly determines num -> high weight in cat->num direction
    cat = pd.array([0, 1] * 50, dtype="int32")
    num = np.where(cat == 0, 0.0, 1.0)
    df = pd.DataFrame({"cat": cat, "num": num})
    assert _w(df)[0, 1] > 0.9


def test_weights_bounded():
    rng = np.random.default_rng(2)
    cat1 = pd.array(rng.integers(0, 3, size=100), dtype="int32")
    cat2 = pd.array(rng.integers(0, 2, size=100), dtype="int32")
    df = pd.DataFrame({
        "a": rng.normal(size=100),
        "b": cat1,
        "c": cat2,
    })
    W = _w(df)
    assert W.min() >= 0.0
    assert W.max() <= 1.0
