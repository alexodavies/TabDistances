"""Abstract base class for table-source discriminators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class BaseDiscriminator(ABC):
    """ABC for discriminators that classify DataFrames by their source.

    Concrete subclasses receive lists of ``pd.DataFrame`` objects and produce
    integer class predictions.

    The interface mirrors sklearn's estimator convention so that subclasses
    can be evaluated with :func:`sklearn.metrics.classification_report` and
    similar utilities.
    """

    @abstractmethod
    def fit(self, X: list[pd.DataFrame], y: np.ndarray) -> "BaseDiscriminator":
        """Train on a labelled collection of DataFrames.

        Parameters
        ----------
        X : list[pd.DataFrame]
            Training tables.
        y : np.ndarray, shape (n,)
            Integer class labels aligned with X.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: list[pd.DataFrame]) -> np.ndarray:
        """Predict class labels for a collection of DataFrames.

        Parameters
        ----------
        X : list[pd.DataFrame]

        Returns
        -------
        np.ndarray, shape (n,)
            Predicted integer class labels.
        """

    def score(self, X: list[pd.DataFrame], y: np.ndarray) -> float:
        """Return accuracy on the given DataFrames and labels.

        Subclasses may override to use a different primary metric.
        """
        return float(accuracy_score(y, self.predict(X)))
