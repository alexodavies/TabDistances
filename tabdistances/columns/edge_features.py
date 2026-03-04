import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .typing import ColumnType


def _onehot(col: pd.Series) -> np.ndarray:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    return enc.fit_transform(col.to_numpy().reshape(-1, 1))


def _pseudo_r2(X: np.ndarray, y: pd.Series) -> float:
    """McFadden's pseudo-R² for logistic regression of categorical y on X."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 0.0
    null_proba = np.tile(counts / counts.sum(), (len(y), 1))
    ll_null = log_loss(y, null_proba, labels=classes)
    if ll_null == 0.0:
        return 0.0
    X = StandardScaler().fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    ll_model = log_loss(y, model.predict_proba(X), labels=classes)
    return float(max(0.0, 1.0 - ll_model / ll_null))


def _r2(X: np.ndarray, Y: np.ndarray) -> float:
    """Proportion of variance in Y explained by X (multioutput mean)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if np.all(Y == Y[0]):  # zero variance in target
        return 0.0
    model = LinearRegression()
    model.fit(X, Y)
    return float(model.score(X, Y))


def _directed_weight(
    ci: pd.Series,
    cj: pd.Series,
    ti: ColumnType,
    tj: ColumnType,
) -> float:
    """Weight of directed edge ci -> cj: sqrt of R²-like metric for how well ci predicts cj."""
    mask = ci.notna() & cj.notna()
    ci, cj = ci[mask], cj[mask]
    if len(ci) < 2:
        return 0.0

    if ti == ColumnType.NUMERIC and tj == ColumnType.NUMERIC:
        r2 = _r2(ci.to_numpy(), cj.to_numpy())
    elif ti == ColumnType.NUMERIC and tj == ColumnType.CATEGORICAL:
        r2 = _pseudo_r2(ci.to_numpy().reshape(-1, 1), cj)
    elif ti == ColumnType.CATEGORICAL and tj == ColumnType.NUMERIC:
        r2 = _r2(_onehot(ci), cj.to_numpy())
    else:  # CATEGORICAL -> CATEGORICAL
        r2 = _pseudo_r2(_onehot(ci), cj)

    return math.sqrt(r2)


def get_edge_weights(
    df: pd.DataFrame,
    col_types: dict[str, ColumnType],
    max_rows: int = 2000,
    random_state: int = 0,
) -> np.ndarray:
    """Return a directed (n_cols, n_cols) matrix of pairwise prediction weights.

    W[i, j] is the sqrt of the R²-like metric for how well column i predicts column j:
    - sqrt(R²) of linear regression for NUMERIC -> NUMERIC
    - sqrt(η²) for CATEGORICAL -> NUMERIC
    - sqrt(McFadden's pseudo-R²) for NUMERIC -> CATEGORICAL or CATEGORICAL -> CATEGORICAL
    Diagonal is 0.

    If the DataFrame has more than ``max_rows`` rows it is randomly sampled
    before fitting, keeping edge-weight computation tractable on large tables.
    """
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state)

    cols = list(df.columns)
    n = len(cols)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            W[i, j] = _directed_weight(
                df[cols[i]], df[cols[j]], col_types[cols[i]], col_types[cols[j]]
            )
    return W
