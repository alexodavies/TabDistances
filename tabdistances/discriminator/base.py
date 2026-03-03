"""Abstract base class for graph-source discriminators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score


class BaseDiscriminator(ABC):
    """ABC for discriminators that classify graphs by their source.

    Concrete subclasses receive lists of ``nx.DiGraph`` objects (each with
    node attrs ``col_type``, ``type_int``, ``histogram`` and edge attr
    ``weight``) and produce integer class predictions.

    The interface mirrors sklearn's estimator convention so that subclasses
    can be evaluated with :func:`sklearn.metrics.classification_report` and
    similar utilities.
    """

    @abstractmethod
    def fit(self, X: list[nx.DiGraph], y: np.ndarray) -> "BaseDiscriminator":
        """Train on a labelled collection of graphs.

        Parameters
        ----------
        X : list[nx.DiGraph]
            Training graphs.
        y : np.ndarray, shape (n,)
            Integer class labels aligned with X.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: list[nx.DiGraph]) -> np.ndarray:
        """Predict class labels for a collection of graphs.

        Parameters
        ----------
        X : list[nx.DiGraph]

        Returns
        -------
        np.ndarray, shape (n,)
            Predicted integer class labels.
        """

    def score(self, X: list[nx.DiGraph], y: np.ndarray) -> float:
        """Return accuracy on the given graphs and labels.

        Subclasses may override to use a different primary metric.
        """
        return float(accuracy_score(y, self.predict(X)))
