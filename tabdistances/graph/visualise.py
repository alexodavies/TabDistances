import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from tabdistances.columns.typing import ColumnType

_NODE_COLORS = {
    ColumnType.NUMERIC: "steelblue",
    ColumnType.CATEGORICAL: "darkorange",
}

_EDGE_CMAP = cm.bone_r
_HIST_W = 0.07   # histogram inset width  (figure fraction)
_HIST_H = 0.045  # histogram inset height (figure fraction)


def draw_graph(
    G: nx.DiGraph,
    ax=None,
    title: str = "",
    seed: int = 0,
) -> plt.Axes:
    """Visualise a column-relationship DiGraph.

    Node colour: steelblue = NUMERIC, darkorange = CATEGORICAL.
    Edge colour: bone colormap scaled to weight, with colorbar.
    Inset beside each node: normalised 30-bin histogram of column values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    if len(G) == 0:
        ax.axis("off")
        return ax

    pos = nx.kamada_kawai_layout(G) if len(G) > 2 else nx.spring_layout(G, seed=seed)

    node_colors = [_NODE_COLORS[G.nodes[n]["col_type"]] for n in G.nodes]

    edges = list(G.edges())
    weights = np.array([G[u][v]["weight"] for u, v in edges]) if edges else np.array([])
    max_w = weights.max() if len(weights) > 0 else 1.0
    min_w = weights.min() if len(weights) > 0 else 0.0
    # vmin set so min_w maps to colormap position 0.5
    cmap_vmax = max_w
    cmap_vmin = 2 * min_w - max_w if max_w > min_w else min_w - 1.0
    norm = mcolors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
    # alpha: min_w -> 0.3, max_w -> 1.0 (same normalisation, stretched from 0.5 to 1.0)
    edge_colors = [
        (*_EDGE_CMAP(norm(w))[:3], 0.3 + (norm(w) - 0.5) * 1.4)
        for w in weights
    ]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=200, alpha=0.95)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=edges,
        width=1.5,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.12",
    )

    # Colorbar for edge weights
    sm = cm.ScalarMappable(cmap=_EDGE_CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02, label="edge weight")

    patches = [mpatches.Patch(color=c, label=ct.value) for ct, c in _NODE_COLORS.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=9)
    if title:
        ax.set_title(title)
    ax.axis("off")

    # Place mini histograms beside each node
    fig.canvas.draw()  # establish transforms before coordinate conversion
    for node in G.nodes:
        hist = G.nodes[node]["histogram"]
        col_type = G.nodes[node]["col_type"]

        disp = ax.transData.transform(pos[node])
        fig_xy = fig.transFigure.inverted().transform(disp)

        x0 = float(np.clip(fig_xy[0] + 0.03, 0.01, 1.0 - _HIST_W - 0.01))
        y0 = float(np.clip(fig_xy[1] + 0.02, 0.01, 1.0 - _HIST_H - 0.01))

        inset = fig.add_axes([x0, y0, _HIST_W, _HIST_H])
        inset.bar(range(len(hist)), hist, color=_NODE_COLORS[col_type], alpha=0.75, width=1.0)
        inset.set_xlim(-2, len(hist) + 1)
        inset.tick_params(labelsize=5, length=2, pad=1)
        inset.yaxis.set_major_locator(plt.MaxNLocator(2))
        inset.xaxis.set_visible(False)
        for spine in ("top", "right"):
            inset.spines[spine].set_visible(False)

    return ax
