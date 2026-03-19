"""Aggregate-feature discriminator: DataFrame  $\\in$  fixed vec  $\\in$  sklearn classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabdistances.discriminator.base import BaseDiscriminator
from tabdistances.discriminator.features import df_to_vec


class AggregateDiscriminator(BaseDiscriminator):
    """Discriminator that featurizes each DataFrame with aggregate statistics
    and delegates classification to an arbitrary sklearn estimator.

    Parameters
    ----------
    clf :
        Any sklearn-compatible classifier with ``fit`` / ``predict`` methods.
    """

    def __init__(self, clf) -> None:
        self._clf = clf

    def _featurize(self, X: list[pd.DataFrame]) -> np.ndarray:
        return np.stack([df_to_vec(df) for df in X])

    def fit(self, X: list[pd.DataFrame], y: np.ndarray) -> "AggregateDiscriminator":
        self._clf.fit(self._featurize(X), y)
        return self

    def predict(self, X: list[pd.DataFrame]) -> np.ndarray:
        return self._clf.predict(self._featurize(X))

    def predict_proba(self, X: list[pd.DataFrame]) -> np.ndarray:
        """Return class probabilities, shape (n, n_classes).

        Requires the underlying classifier to support ``predict_proba``.
        """
        return self._clf.predict_proba(self._featurize(X))
