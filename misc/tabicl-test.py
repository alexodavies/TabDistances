import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import numpy as np
import networkx as nx
import pandas as pd
import torch

from tabdistances.data.priors.tabicl.dataset import SCMPrior

from tabdistances.graph.builder import build_graph
from tabdistances.graph.visualise import draw_graph
from tabdistances.columns.edge_features import get_edge_weights
from tabdistances.columns.typing import infer_col_types

N_TABLES = 50
BATCH_SIZE = 8       # datasets per batch (stage1 uses batch_size_per_gp=4; 8 is fine for testing)
MIN_FEATURES = 3     # skip trivially narrow tables
OUT_DIR = Path("graphs-tabicl")
VIS_DIR = OUT_DIR / "vis"
JSON_DIR = OUT_DIR / "json"
VIS_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)


def graph_to_dict(G: nx.DiGraph) -> dict:
    """Serialise a graph to a plain dict (JSON-safe)."""
    nodes = []
    for n, data in G.nodes(data=True):
        nodes.append({
            "id": n,
            "col_type": data["col_type"].value,
            "type_int": data["type_int"].tolist(),
            "histogram": data["histogram"].tolist(),
        })
    edges = [
        {"source": u, "target": v, "weight": float(d["weight"])}
        for u, v, d in G.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


prior = SCMPrior(
    batch_size=BATCH_SIZE,
    batch_size_per_gp=4,        # stage1 default
    min_features=MIN_FEATURES,
    max_features=100,           # stage1 default
    max_classes=10,             # stage1 default
    max_seq_len=1024,           # stage1 default
    min_train_size=0.1,         # stage1 default
    max_train_size=0.9,         # stage1 default
    prior_type="mix_scm",       # stage1 default
    n_jobs=1,
    device="cpu",
)

processed = 0
batch_idx = 0

while processed < N_TABLES:
    X, y, d, seq_lens, train_sizes = prior.get_batch()
    # X: (batch_size, seq_len, max_features)
    # y: (batch_size, seq_len)
    # d: (batch_size,)  — number of active features
    # seq_lens: (batch_size,)

    for i in range(X.shape[0]):
        n_features = int(d[i].item())
        seq_len = int(seq_lens[i].item())

        if n_features < MIN_FEATURES or seq_len == 0:
            continue

        X_i = X[i, :seq_len, :n_features].numpy().astype(np.float64)
        y_i = y[i, :seq_len].numpy().astype(int)

        col_names = [f"feature_{j}" for j in range(n_features)]
        df = pd.DataFrame(X_i, columns=col_names)
        df["target"] = y_i

        table_id = f"tabicl_batch{batch_idx:04d}_idx{i:02d}"
        n_classes = int(y_i.max()) + 1

        print(f"\n{'='*60}")
        print(f"Table: {table_id}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols  ({n_features} features + target, {n_classes} classes)")
        print(df.head().to_string())

        try:
            G = build_graph(df, threshold=0.7, cardinality_threshold=0.2)
            col_types = {n: G.nodes[n]["col_type"] for n in G.nodes}
            print(f"Types: { {c: t.value for c, t in col_types.items()} }")
            print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

            if G.number_of_edges() > 0:
                edges = sorted(G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)
                print("Top edges by weight:")
                for u, v, data in edges[:5]:
                    print(f"  {u!r} — {v!r}: {data['weight']:.3f}")

            fig, ax = plt.subplots(figsize=(12, 8))
            draw_graph(G, ax=ax, title=f"{table_id}  ({df.shape[0]} rows × {df.shape[1]} cols)")
            vis_path = VIS_DIR / f"{table_id}.png"
            fig.savefig(vis_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved vis: {vis_path}")

            json_path = JSON_DIR / f"{table_id}.json"
            json_path.write_text(json.dumps(graph_to_dict(G), indent=2))
            print(f"Saved json: {json_path}")

            processed += 1
        except Exception as e:
            print(f"Table {table_id} failed: {e}")

        if processed >= N_TABLES:
            break

    batch_idx += 1

print(f"\n{'='*60}")
print(f"Done. Processed {processed} synthetic tables.")
print(f"Visualisations saved to {VIS_DIR.resolve()}/")
print(f"JSON graphs saved to {JSON_DIR.resolve()}/")
