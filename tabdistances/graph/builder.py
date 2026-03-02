import networkx as nx
import pandas as pd

from tabdistances.columns.typing import infer_col_types
from tabdistances.columns.node_features import get_type_int, get_histogram
from tabdistances.columns.edge_features import get_edge_weights


def build_graph(
    df: pd.DataFrame,
    threshold: float = 0.0,
    cardinality_threshold: float | None = None,
    max_rows: int = 2000,
    max_cat_classes: int | None = 20,
) -> nx.Graph:
    """Convert a DataFrame into a weighted directed column-relationship graph.

    Each column becomes a node with a feature vector encoding its type.
    A directed edge ci -> cj is added when the sqrt(R²-like) weight of ci
    predicting cj exceeds `threshold`.

    Args:
        df: Input table.
        threshold: Minimum edge weight to include (default 0 = all edges).
        cardinality_threshold: If set, float64 columns with unique-value ratio
            below this value are classified as CATEGORICAL (see infer_col_type).
        max_rows: Randomly sample this many rows before computing edge weights
            to keep fitting tractable on large tables (default 2000).
        max_cat_classes: Reclassify CATEGORICAL columns with more unique values
            than this as NUMERIC, so linear R² is used instead of logistic
            pseudo-R² (default 20).

    Returns:
        nx.DiGraph with node attrs ``feature`` (np.ndarray) and ``col_type``,
        and edge attr ``weight`` (float in [0, 1]).
    """
    col_types = infer_col_types(df, cardinality_threshold, max_cat_classes)
    cols = list(df.columns)
    W = get_edge_weights(df, col_types, max_rows=max_rows)

    G = nx.DiGraph()
    for col in cols:
        G.add_node(col, type_int=get_type_int(col_types[col]), col_type=col_types[col], histogram=get_histogram(df[col]))

    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i == j:
                continue
            w = W[i, j]
            if w > threshold:
                G.add_edge(ci, cj, weight=w)

    return G
