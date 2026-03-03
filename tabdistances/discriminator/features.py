"""Aggregate feature extraction from column-relationship graphs.

Converts an ``nx.DiGraph`` produced by :func:`tabdistances.graph.builder.build_graph`
into a fixed-length numpy vector suitable for tabular classifiers.

Feature layout (68 dimensions)
--------------------------------
0       n_nodes          — number of columns
1       n_edges          — number of directed edges
2       edge_density     — n_edges / (n_nodes*(n_nodes-1)), or 0 for n_nodes<=1
3       weight_mean      — mean edge weight (0 if no edges)
4       weight_std       — std  edge weight (0 if no edges)
5       weight_max       — max  edge weight (0 if no edges)
6       weight_q25       — 25th-percentile edge weight (0 if no edges)
7       weight_q75       — 75th-percentile edge weight (0 if no edges)
8..37   histogram_mean   — mean of each 30-bin cumulative histogram across nodes
38..67  histogram_std    — std  of each 30-bin cumulative histogram across nodes
"""

from __future__ import annotations

import networkx as nx
import numpy as np

_N_HIST_BINS = 30
FEATURE_DIM  = 8 + 2 * _N_HIST_BINS   # 68


def graph_to_vec(G: nx.DiGraph) -> np.ndarray:
    """Extract a fixed-length aggregate feature vector from a graph.

    Parameters
    ----------
    G : nx.DiGraph
        Graph produced by :func:`tabdistances.graph.builder.build_graph`.
        Each node must have ``col_type`` (ColumnType) and ``histogram``
        (np.ndarray of length 30) attributes.

    Returns
    -------
    np.ndarray, shape (FEATURE_DIM,)
    """
    nodes = list(G.nodes(data=True))
    n = len(nodes)

    # ------------------------------------------------------------------
    # Node-level aggregates
    # ------------------------------------------------------------------
    if n == 0:
        return np.zeros(FEATURE_DIM)

    histograms = np.stack([d["histogram"] for _, d in nodes])  # (n, 30)
    hist_mean = histograms.mean(axis=0)                        # (30,)
    hist_std  = histograms.std(axis=0)                         # (30,)

    # ------------------------------------------------------------------
    # Edge-level aggregates
    # ------------------------------------------------------------------
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    n_edges = len(weights)

    max_edges   = n * (n - 1)
    edge_density = n_edges / max_edges if max_edges > 0 else 0.0

    if n_edges > 0:
        w_mean = weights.mean()
        w_std  = weights.std()
        w_max  = weights.max()
        w_q25, w_q75 = np.percentile(weights, [25, 75])
    else:
        w_mean = w_std = w_max = w_q25 = w_q75 = 0.0

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    return np.concatenate([
        [n, n_edges, edge_density,
         w_mean, w_std, w_max, w_q25, w_q75],
        hist_mean,
        hist_std,
    ])
