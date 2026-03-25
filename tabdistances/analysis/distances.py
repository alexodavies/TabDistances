"""k-NN distance and coverage metrics for tabular dataset comparison.

These functions were previously copy-pasted (with minor variations) across
``analysis/rank_vs_coverage*.py``, ``optimisation/``, and ``grid/``.
The canonical implementation is preserved here.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


def compute_nn_distances(
    query_feats: np.ndarray,
    ref_feats: np.ndarray,
    k: int = 5,
    scale: bool = True,
) -> np.ndarray:
    """Mean k-NN distance from each query point to the reference distribution.

    Before computing distances, NaN/Inf values are replaced with 0 and all
    features are Min-Max scaled to ``[0, 1]`` using the reference distribution
    as the fitting set (so the query is transformed into the same space).

    Parameters
    ----------
    query_feats : np.ndarray, shape (N, D)
        Feature vectors to query (e.g. TALENT datasets).
    ref_feats : np.ndarray, shape (M, D)
        Reference distribution (e.g. TabICL prior samples).
    k : int
        Number of nearest neighbours.
    scale : bool
        If True, apply Min-Max scaling using the reference set.

    Returns
    -------
    np.ndarray, shape (N,)
        Mean distance from each query point to its k nearest neighbours in
        the reference set.
    """
    query_feats = np.nan_to_num(query_feats, nan=0.0, posinf=0.0, neginf=0.0)
    ref_feats   = np.nan_to_num(ref_feats,   nan=0.0, posinf=0.0, neginf=0.0)

    if scale:
        scaler = MinMaxScaler().fit(ref_feats)
        query_feats = scaler.transform(query_feats)
        ref_feats   = scaler.transform(ref_feats)

    tree = BallTree(ref_feats, metric="euclidean")
    dists, _ = tree.query(query_feats, k=min(k, len(ref_feats)))
    return dists.mean(axis=1)


def knn_coverage(
    real_feats: np.ndarray,
    synth_feats: np.ndarray,
    k: int = 5,
    quantile: float = 0.95,
) -> tuple[float, float]:
    """Compute (recall, precision) coverage via k-NN proximity.

    The coverage threshold is derived from the synthetic distribution's own
    internal k-NN structure (at the given *quantile*), making it adaptive to
    the density of the synthetic set.

    Parameters
    ----------
    real_feats : np.ndarray, shape (N, D)
        Feature vectors for real tables.
    synth_feats : np.ndarray, shape (M, D)
        Feature vectors for synthetic tables.
    k : int
        Number of nearest neighbours used to estimate local density.
    quantile : float
        Percentile (0–1) of the synthetic self-distances used as the coverage
        radius.  0.95 means a real point must be within the 95th-percentile
        neighbourhood of its nearest synthetic point to be "covered".

    Returns
    -------
    recall : float
        Fraction of real samples within the synthetic neighbourhood (coverage
        of the real distribution by the synthetic one).
    precision : float
        Fraction of synthetic samples within the real neighbourhood (coverage
        of the synthetic distribution by the real one).
    """
    recall    = _one_sided_coverage(real_feats,  synth_feats, k=k, quantile=quantile)
    precision = _one_sided_coverage(synth_feats, real_feats,  k=k, quantile=quantile)
    return recall, precision


def _one_sided_coverage(
    ref: np.ndarray,
    cand: np.ndarray,
    k: int,
    quantile: float,
) -> float:
    """Fraction of *ref* points covered by *cand*.

    A ref point is "covered" if its mean distance to its k nearest neighbours
    in *cand* is below the threshold derived from *cand*'s self-distances.
    """
    lo  = np.minimum(ref.min(0), cand.min(0))
    hi  = np.maximum(ref.max(0), cand.max(0))
    rng = hi - lo
    rng[rng == 0] = 1.0
    ref_n  = (ref  - lo) / rng
    cand_n = (cand - lo) / rng

    # Adaptive threshold from cand's internal structure
    nn_ss = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(cand_n)
    d_ss, _ = nn_ss.kneighbors(cand_n)
    threshold = np.percentile(d_ss[:, 1:].mean(axis=1), quantile * 100)

    # How many ref points fall within that threshold?
    nn_rs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(cand_n)
    d_rs, _ = nn_rs.kneighbors(ref_n)
    return float(1.0 - (d_rs.mean(axis=1) > threshold).mean())
