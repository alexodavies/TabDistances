from enum import Enum

import pandas as pd


class ColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


def infer_col_type(
    col: pd.Series,
    cardinality_threshold: float | None = 0.2,
    max_cat_classes: int | None = 20,
) -> ColumnType:
    """Infer column type from dtype, with optional cardinality fallbacks.

    float64 (standard-scaled continuous) -> NUMERIC
    other   (integer-encoded or string)  -> CATEGORICAL

    ``cardinality_threshold`` (0–1): reclassify a float64 column as CATEGORICAL
    when its unique-value ratio falls below the threshold.  Useful for data
    where categorical features have been standard-scaled to floats (e.g. TabICL
    synthetic data).

    ``max_cat_classes``: reclassify a CATEGORICAL column as NUMERIC when its
    unique-value count exceeds this limit.  High-cardinality columns (IDs,
    free-text keys, etc.) are better treated as ordinal/numeric for predictive
    measures — logistic regression with hundreds of classes is both slow and
    misleading.
    """
    if col.dtype == "float64":
        if cardinality_threshold is not None:
            ratio = col.nunique() / max(len(col), 1)
            if ratio < cardinality_threshold:
                return ColumnType.CATEGORICAL
        return ColumnType.NUMERIC

    # Non-float columns default to CATEGORICAL; reclassify if too many classes.
    if max_cat_classes is not None and col.nunique() > max_cat_classes:
        return ColumnType.NUMERIC
    return ColumnType.CATEGORICAL


def infer_col_types(
    df: pd.DataFrame,
    cardinality_threshold: float | None = 0.2,
    max_cat_classes: int | None = 20,
) -> dict[str, ColumnType]:
    """Infer the :class:`ColumnType` of every column in *df*.

    Applies the same two-rule heuristic as :func:`infer_col_type` to each
    column independently:

    1. **float64 → NUMERIC** by default.  Exception: if the fraction of unique
       values is below *cardinality_threshold* the column is reclassified as
       CATEGORICAL (catches categorical features that have been standard-scaled
       to floats, which is common in TabICL synthetic data).

    2. **everything else → CATEGORICAL** by default.  Exception: if the number
       of unique values exceeds *max_cat_classes* the column is reclassified as
       NUMERIC (high-cardinality string keys and IDs are better treated
       ordinally).

    Parameters
    ----------
    df : pd.DataFrame
    cardinality_threshold : float or None
        Unique-value fraction below which a float64 column is reclassified as
        CATEGORICAL.  Set to ``None`` to disable.  Default: ``0.2``.
    max_cat_classes : int or None
        Unique-value count above which a non-float column is reclassified as
        NUMERIC.  Set to ``None`` to disable.  Default: ``20``.

    Returns
    -------
    dict[str, ColumnType]
        Mapping from column name to inferred type.

    Examples
    --------
    >>> import pandas as pd
    >>> from tabdistances.columns.typing import infer_col_types, ColumnType
    >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    >>> infer_col_types(df)
    {'x': <ColumnType.NUMERIC: 'numeric'>, 'cat': <ColumnType.CATEGORICAL: 'categorical'>}

    To disable cardinality reclassification (e.g. for purely numeric data):

    >>> infer_col_types(df, cardinality_threshold=None)
    """
    return {
        col: infer_col_type(df[col], cardinality_threshold, max_cat_classes)
        for col in df.columns
    }
