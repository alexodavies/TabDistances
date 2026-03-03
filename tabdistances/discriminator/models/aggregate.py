"""Aggregate-feature discriminator: graph → fixed vec → sklearn classifier."""

from __future__ import annotations

import networkx as nx
import numpy as np

from tabdistances.discriminator.base import BaseDiscriminator
from tabdistances.discriminator.features import graph_to_vec


class AggregateDiscriminator(BaseDiscriminator):
    """Discriminator that featurizes each graph with aggregate statistics
    and delegates classification to an arbitrary sklearn estimator.

    Parameters
    ----------
    clf :
        Any sklearn-compatible classifier with ``fit`` / ``predict`` methods.
    """

    def __init__(self, clf) -> None:
        self._clf = clf

    def _featurize(self, X: list[nx.DiGraph]) -> np.ndarray:
        return np.stack([graph_to_vec(G) for G in X])

    def fit(self, X: list[nx.DiGraph], y: np.ndarray) -> "AggregateDiscriminator":
        self._clf.fit(self._featurize(X), y)
        return self

    def predict(self, X: list[nx.DiGraph]) -> np.ndarray:
        return self._clf.predict(self._featurize(X))

    def predict_proba(self, X: list[nx.DiGraph]) -> np.ndarray:
        """Return class probabilities, shape (n, n_classes).

        Requires the underlying classifier to support ``predict_proba``.
        """
        return self._clf.predict_proba(self._featurize(X))
