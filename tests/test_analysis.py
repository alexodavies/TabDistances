"""Tests for tabdistances.analysis.distances."""

import numpy as np
import pytest

from tabdistances.analysis.distances import compute_nn_distances, knn_coverage


def _cluster(center, n=50, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return center + rng.standard_normal((n, len(center))) * spread


# ---------------------------------------------------------------------------
# compute_nn_distances
# ---------------------------------------------------------------------------

def test_compute_nn_distances_shape():
    query = np.random.default_rng(0).standard_normal((10, 8))
    ref   = np.random.default_rng(1).standard_normal((20, 8))
    dists = compute_nn_distances(query, ref, k=3)
    assert dists.shape == (10,)


def test_compute_nn_distances_non_negative():
    query = np.random.default_rng(2).standard_normal((15, 5))
    ref   = np.random.default_rng(3).standard_normal((30, 5))
    dists = compute_nn_distances(query, ref, k=5)
    assert np.all(dists >= 0)


def test_compute_nn_distances_closer_when_same_distribution():
    """Points drawn from the same distribution should be closer than points
    drawn from a shifted distribution."""
    rng = np.random.default_rng(42)
    ref   = rng.standard_normal((100, 4))
    near  = rng.standard_normal((50, 4))          # same distribution
    far   = rng.standard_normal((50, 4)) + 10.0   # shifted by 10

    d_near = compute_nn_distances(near, ref, k=5).mean()
    d_far  = compute_nn_distances(far,  ref, k=5).mean()
    assert d_near < d_far


def test_compute_nn_distances_nan_input():
    """NaN values must not propagate to the output."""
    query = np.full((5, 4), np.nan)
    ref   = np.random.default_rng(0).standard_normal((20, 4))
    dists = compute_nn_distances(query, ref, k=3)
    assert not np.any(np.isnan(dists))


def test_compute_nn_distances_no_scale():
    query = np.random.default_rng(0).standard_normal((5, 3))
    ref   = np.random.default_rng(1).standard_normal((10, 3))
    dists = compute_nn_distances(query, ref, k=2, scale=False)
    assert dists.shape == (5,)
    assert np.all(dists >= 0)


def test_compute_nn_distances_k_clipped_to_ref_size():
    """k larger than the reference set must not raise."""
    query = np.random.default_rng(0).standard_normal((3, 4))
    ref   = np.random.default_rng(1).standard_normal((2, 4))
    dists = compute_nn_distances(query, ref, k=10)
    assert dists.shape == (3,)


# ---------------------------------------------------------------------------
# knn_coverage
# ---------------------------------------------------------------------------

def test_knn_coverage_returns_two_floats():
    real  = np.random.default_rng(0).standard_normal((30, 4))
    synth = np.random.default_rng(1).standard_normal((30, 4))
    recall, precision = knn_coverage(real, synth, k=3)
    assert isinstance(recall, float)
    assert isinstance(precision, float)


def test_knn_coverage_range():
    real  = np.random.default_rng(0).standard_normal((40, 4))
    synth = np.random.default_rng(1).standard_normal((40, 4))
    recall, precision = knn_coverage(real, synth, k=5, quantile=0.95)
    assert 0.0 <= recall    <= 1.0
    assert 0.0 <= precision <= 1.0


def test_knn_coverage_identical_distributions():
    """When real and synthetic come from the same distribution, coverage should
    be reasonably high (well above 0.5 for large enough samples)."""
    rng   = np.random.default_rng(99)
    real  = rng.standard_normal((200, 4))
    synth = rng.standard_normal((200, 4))
    recall, precision = knn_coverage(real, synth, k=5, quantile=0.95)
    assert recall    > 0.5
    assert precision > 0.5


def test_knn_coverage_disjoint_distributions():
    """When synthetic is far from real, recall should be low."""
    real  = _cluster(np.zeros(4), n=80)
    synth = _cluster(np.ones(4) * 100, n=80)  # 100 units away
    recall, _ = knn_coverage(real, synth, k=5, quantile=0.95)
    assert recall < 0.1


def test_knn_coverage_perfect_overlap():
    """When synthetic == real, recall and precision should be ~1.0."""
    pts = np.random.default_rng(7).standard_normal((60, 4))
    recall, precision = knn_coverage(pts, pts.copy(), k=5, quantile=0.95)
    assert recall    > 0.9
    assert precision > 0.9
