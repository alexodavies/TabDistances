"""Tests for tabdistances.discriminator.features."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabdistances.discriminator.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_MODES,
    df_to_vec,
    featurize_tables,
    select_features,
)


# ---------------------------------------------------------------------------
# df_to_vec
# ---------------------------------------------------------------------------

def _simple_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(size=n).astype("float64"),
        "b": rng.normal(size=n).astype("float64"),
        "c": rng.integers(0, 5, size=n),
    })


def test_df_to_vec_output_shape():
    vec = df_to_vec(_simple_df())
    assert vec.shape == (FEATURE_DIM,)


def test_df_to_vec_empty_df():
    vec = df_to_vec(pd.DataFrame())
    assert vec.shape == (FEATURE_DIM,)
    assert np.all(vec == 0.0)


def test_df_to_vec_no_numeric_columns():
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
    vec = df_to_vec(df)
    assert vec.shape == (FEATURE_DIM,)
    assert not np.any(np.isnan(vec))


def test_df_to_vec_type_ratio():
    # 1 numeric float64 col, 2 categorical int cols → type_ratio = 2/3
    df = pd.DataFrame({
        "n": pd.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float64"),
        "c1": [0, 1, 2, 3, 4],
        "c2": [0, 1, 0, 1, 0],
    })
    vec = df_to_vec(df)
    # feature index 1 is type_ratio
    np.testing.assert_almost_equal(vec[1], 2 / 3, decimal=5)


def test_df_to_vec_missing_rate():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
    vec = df_to_vec(df)
    # feature index 2 is missing_rate: 2/4 = 0.5
    np.testing.assert_almost_equal(vec[2], 0.5, decimal=5)


def test_df_to_vec_no_nans_in_output():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.standard_normal((200, 10)))
    vec = df_to_vec(df)
    assert not np.any(np.isnan(vec))


def test_df_to_vec_deterministic():
    df = _simple_df()
    assert np.array_equal(df_to_vec(df), df_to_vec(df))


# ---------------------------------------------------------------------------
# FEATURE_NAMES / FEATURE_MODES consistency
# ---------------------------------------------------------------------------

def test_feature_names_length():
    assert len(FEATURE_NAMES) == FEATURE_DIM


def test_select_features_full():
    feats = np.arange(FEATURE_DIM, dtype=float).reshape(1, -1)
    out, names = select_features(feats, "full")
    assert out.shape[1] == len(names)
    # "full" excludes index 0 (n_cols)
    assert out.shape[1] == FEATURE_DIM - 1


def test_select_features_scalar_only():
    feats = np.ones((5, FEATURE_DIM))
    out, names = select_features(feats, "scalar_only")
    assert out.shape == (5, len(names))
    assert "n_cols" not in names


def test_select_features_hist_only():
    feats = np.ones((5, FEATURE_DIM))
    out, names = select_features(feats, "hist_only")
    assert out.shape == (5, len(names))
    assert all(n.startswith("hist_") for n in names)


def test_select_features_invalid_mode():
    feats = np.ones((3, FEATURE_DIM))
    with pytest.raises(ValueError, match="Unknown feature mode"):
        select_features(feats, "bogus")


# ---------------------------------------------------------------------------
# featurize_tables
# ---------------------------------------------------------------------------

def _fake_tables(n=5):
    rng = np.random.default_rng(7)
    for i in range(n):
        df = pd.DataFrame(rng.standard_normal((50, 4)))
        yield i, f"table_{i:03d}", df


def test_featurize_tables_shape():
    names, feats = featurize_tables(_fake_tables(5))
    assert len(names) == 5
    assert feats.shape == (5, FEATURE_DIM)


def test_featurize_tables_names_match_rows():
    names, feats = featurize_tables(_fake_tables(8))
    assert len(names) == feats.shape[0]


def test_featurize_tables_cache_write_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir) / "feats.npz"

        names1, feats1 = featurize_tables(_fake_tables(4), cache_path=cache)
        assert cache.exists()

        # Second call: nothing in the iterator, but cache has 4 entries
        names2, feats2 = featurize_tables(iter([]), cache_path=cache)
        assert names2 == names1
        np.testing.assert_array_equal(feats2, feats1)


def test_featurize_tables_cache_no_recompute():
    """Tables already in cache must not be reprocessed (names are deduped)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir) / "feats.npz"
        featurize_tables(_fake_tables(4), cache_path=cache)

        # Feed the same tables again — cache should not grow
        names, feats = featurize_tables(_fake_tables(4), cache_path=cache)
        assert len(names) == 4


def test_featurize_tables_empty_source():
    names, feats = featurize_tables(iter([]))
    assert names == []
    assert feats.shape == (0, FEATURE_DIM)
