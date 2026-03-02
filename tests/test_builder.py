import numpy as np
import pandas as pd
import networkx as nx

from tabdistances.columns.typing import ColumnType
from tabdistances.graph.builder import build_graph


def _make_df():
    x = np.linspace(0, 1, 50)
    return pd.DataFrame({"num_a": x, "num_b": x, "cat": ["a", "b"] * 25})


def test_returns_graph():
    assert isinstance(build_graph(_make_df()), nx.DiGraph)


def test_nodes_match_columns():
    df = _make_df()
    assert set(build_graph(df).nodes) == set(df.columns)


def test_node_type_int_is_array():
    G = build_graph(_make_df())
    for node in G.nodes:
        assert isinstance(G.nodes[node]["type_int"], np.ndarray)


def test_node_histogram_is_array():
    G = build_graph(_make_df())
    for node in G.nodes:
        h = G.nodes[node]["histogram"]
        assert isinstance(h, np.ndarray)
        assert h.shape == (30,)


def test_node_col_type():
    G = build_graph(_make_df())
    assert G.nodes["num_a"]["col_type"] == ColumnType.NUMERIC
    assert G.nodes["cat"]["col_type"] == ColumnType.CATEGORICAL


def test_edge_weight_in_range():
    G = build_graph(_make_df())
    for _, _, data in G.edges(data=True):
        assert 0.0 <= data["weight"] <= 1.0


def test_correlated_columns_have_high_weight():
    x = np.linspace(0, 1, 100)
    df = pd.DataFrame({"a": x, "b": x})
    G = build_graph(df)
    assert G["a"]["b"]["weight"] > 0.99


def test_threshold_removes_weak_edges():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    G_all = build_graph(df, threshold=0.0)
    G_strict = build_graph(df, threshold=0.5)
    assert G_strict.number_of_edges() <= G_all.number_of_edges()
