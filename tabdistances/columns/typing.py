from enum import Enum

import pandas as pd


class ColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


def infer_col_type(
    col: pd.Series,
    cardinality_threshold: float | None = None,
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
    cardinality_threshold: float | None = None,
    max_cat_classes: int | None = 20,
) -> dict[str, ColumnType]:
    return {
        col: infer_col_type(df[col], cardinality_threshold, max_cat_classes)
        for col in df.columns
    }
