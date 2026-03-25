"""tabdistances.data — data loaders for real and synthetic tabular datasets.

All loaders expose a common ``iter_tables()`` interface that yields
``(index, name, df)`` tuples so downstream code can treat real and synthetic
data identically.

Available loaders
-----------------
``TabFMDataset``  — 338 Kaggle datasets (requires downloaded CSVs)
``T4Dataset``     — ~3.1M web-scraped tables streamed from HuggingFace
``SCMPrior``      — synthetic tables via structural causal models (TabICL)

For SCMPrior, use :func:`iter_prior_tables` to get the same ``iter_tables()``
interface instead of working with the raw tensor output of ``get_batch()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tabdistances.data.priors.tabicl.dataset import SCMPrior


def iter_prior_tables(
    prior: "SCMPrior",
    n: int,
) -> Iterator[tuple[int, str, pd.DataFrame]]:
    """Yield ``(index, name, df)`` from an :class:`SCMPrior`, matching the
    ``iter_tables()`` interface of the real-data loaders.

    Converts the tensor output of ``SCMPrior.get_batch()`` into individual
    DataFrames, so synthetic data can be passed to the same downstream code
    (e.g. :func:`tabdistances.featurize_tables`) as real-data sources.

    Parameters
    ----------
    prior : SCMPrior
        A configured prior instance.  Set all hyperparameters (``prior_type``,
        ``max_classes``, ``max_seq_len``, etc.) on the prior before calling
        this function.
    n : int
        Number of tables to yield.

    Yields
    ------
    index : int
        Sequential index starting from 0.
    name : str
        Synthetic identifier of the form ``"tabicl_000042"``.
    df : pd.DataFrame
        Table with numeric feature columns ``"f0", "f1", …`` and a
        ``"target"`` column.

    Examples
    --------
    >>> from tabdistances.data.priors.tabicl.dataset import SCMPrior
    >>> from tabdistances.data import iter_prior_tables
    >>> from tabdistances import featurize_tables
    >>>
    >>> prior = SCMPrior(batch_size=8, prior_type="mix_scm")
    >>> names, feats = featurize_tables(iter_prior_tables(prior, n=500))
    >>> feats.shape
    (500, 70)
    """
    idx = 0
    while idx < n:
        X, y, d, seq_lens, _ = prior.get_batch()
        for i in range(X.shape[0]):
            if idx >= n:
                return
            n_feat  = int(d[i].item())
            seq_len = int(seq_lens[i].item())
            if seq_len == 0 or n_feat == 0:
                continue
            X_i = X[i, :seq_len, :n_feat].numpy().astype(float)
            y_i = y[i, :seq_len].numpy().astype(int)
            df  = pd.DataFrame(X_i, columns=[f"f{j}" for j in range(n_feat)])
            df["target"] = y_i
            yield idx, f"tabicl_{idx:06d}", df
            idx += 1
