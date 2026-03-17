# TabDistances — Exploration
#
# Build a graph from a table and inspect its structure.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from tabdistances.graph.builder import build_graph

# # Load a table
# df = pd.read_csv("data/real/example.csv")
# df.head()

# G = build_graph(df, threshold=0.1)
# print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
# print("\nNode features:")
# for node, data in G.nodes(data=True):
#     print(f"  {node}: type={data['col_type'].value}, feature={data['feature']}")

# edge_weights = nx.get_edge_attributes(G, "weight")
# pos = nx.spring_layout(G, seed=0)
# nx.draw(G, pos, with_labels=True, width=[w * 5 for w in edge_weights.values()])
# nx.draw_networkx_edge_labels(G, pos, {k: f"{v:.2f}" for k, v in edge_weights.items()})
# plt.show()

# ---------------------------------------------------------------------------
# ## sklearn Datasets
# ---------------------------------------------------------------------------

from tabdistances.columns.typing import ColumnType

NODE_COLOURS = {
    ColumnType.NUMERIC: "steelblue",
    ColumnType.CATEGORICAL: "darkorange",
}
EDGE_COLOURS = {
    ("numeric", "numeric"):         "steelblue",
    ("categorical", "numeric"):     "mediumpurple",
    ("numeric", "categorical"):     "mediumpurple",
    ("categorical", "categorical"): "darkorange",
}

def plot_graph(df, title, threshold=0.7):
    G = build_graph(df, threshold=threshold)
    print(f"\n{title}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_colours = [NODE_COLOURS[G.nodes[n]["col_type"]] for n in G.nodes]

    edge_list = list(G.edges())
    edge_weights = [G[u][v]["weight"] for u, v in edge_list]
    edge_colours = [
        EDGE_COLOURS[(G.nodes[u]["col_type"].value, G.nodes[v]["col_type"].value)]
        for u, v in edge_list
    ]

    pos = nx.spring_layout(G, seed=0)
    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colours, ax=ax, node_size=600)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white")
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_colours,
                           width=[w * 5 for w in edge_weights], ax=ax)
    nx.draw_networkx_edge_labels(G, pos,
                                 {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edge_list},
                                 font_size=7, ax=ax)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=f"node: {l}")
        for l, c in [("numeric", "steelblue"), ("categorical", "darkorange")]
    ] + [
        Line2D([0], [0], color=c, linewidth=2, label=f"edge: {l}")
        for l, c in [("num–num", "steelblue"), ("num–cat", "mediumpurple"), ("cat–cat", "darkorange")]
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# ## Synthetic Dataset
#
# Known structure:
# - `num_a` → `num_b` (linear + small noise): strong num–num edge expected
# - `num_a` → `cat_hi_lo` (thresholded): strong num–cat edge expected
# - `cat_hi_lo` → `cat_label` (noisy copy): strong cat–cat edge expected
# - `num_free`: independent noise — no strong edges expected
# - `cat_random`: independent random category — no strong edges expected
# ---------------------------------------------------------------------------

import numpy as np

def make_synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)

    # num_a: base signal
    num_a = rng.normal(size=n)

    # num_b: linear function of num_a + small noise  →  strong num–num
    num_b = 2.0 * num_a + rng.normal(scale=0.2, size=n)

    # num_free: pure noise  →  no strong edges
    num_free = rng.normal(size=n)

    # cat_hi_lo: thresholded on num_a  →  strong num–cat
    cat_hi_lo = pd.Categorical(np.where(num_a > 0, "high", "low"))

    # cat_label: noisy copy of cat_hi_lo  →  strong cat–cat
    flip = rng.random(size=n) < 0.05          # flip 5 % of labels
    raw = np.where(num_a > 0, "high", "low")
    raw[flip] = np.where(raw[flip] == "high", "low", "high")
    cat_label = pd.Categorical(raw)

    # cat_random: independent  →  no strong edges
    cat_random = pd.Categorical(rng.choice(["x", "y", "z"], size=n))

    return pd.DataFrame({
        "num_a":      num_a,
        "num_b":      num_b,
        "num_free":   num_free,
        "cat_hi_lo":  cat_hi_lo,
        "cat_label":  cat_label,
        "cat_random": cat_random,
    })

synthetic_df = make_synthetic()
print(synthetic_df.head())

plot_graph(synthetic_df, "synthetic")

# ---------------------------------------------------------------------------
# sklearn Datasets
# ---------------------------------------------------------------------------

from sklearn.datasets import load_iris, load_wine, load_breast_cancer

def sklearn_to_df(loader, include_target=False):
    data = loader()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    if include_target:
        df["target"] = pd.Categorical([data.target_names[t] for t in data.target])
    return df

datasets = {
    "iris":          sklearn_to_df(load_iris, include_target=True),
    "wine":          sklearn_to_df(load_wine, include_target=True),
    "breast_cancer": sklearn_to_df(load_breast_cancer),
}

for name, df in datasets.items():
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} cols — {list(df.columns)}")

for name, df in datasets.items():
    plot_graph(df, name)
