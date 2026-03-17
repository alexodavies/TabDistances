import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tabdistances.data.cleantablib.dataset import CleanTabLibDataset
from tabdistances.graph.builder import build_graph
from tabdistances.graph.visualise import draw_graph
from tabdistances.columns.edge_features import get_edge_weights
from tabdistances.columns.typing import infer_col_types

import json
import random

import numpy as np
import networkx as nx

N_TABLES = 50
OUT_DIR = Path("graphs")
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


dataset = CleanTabLibDataset(config="stage4", split="train", deduplicate=True)
processed = 0
for i, table_id, df in dataset.iter_tables():
    try:
        if df.shape[1] <= 2 or random.random() >= 0.005:
            continue

        print(f"\n{'='*60}")
        print(f"Table {i}: {table_id}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"Columns: {list(df.columns)}")
        print(df.head().to_string())

        col_types = infer_col_types(df)
        print(f"Types: { {c: t.value for c, t in col_types.items()} }")

        G = build_graph(df, threshold=0.7)  # threshold=0 to see all weights
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if G.number_of_edges() > 0:
            edges = sorted(G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)
            print("Top edges by weight:")
            for u, v, data in edges[:5]:
                print(f"  {u!r} — {v!r}: {data['weight']:.3f}")

        safe_id = table_id.replace('/', '_')

        fig, ax = plt.subplots(figsize=(12, 8))
        draw_graph(G, ax=ax, title=f"{table_id}  ({df.shape[0]} rows × {df.shape[1]} cols)")
        vis_path = VIS_DIR / f"{safe_id}.png"
        fig.savefig(vis_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved vis: {vis_path}")

        json_path = JSON_DIR / f"{safe_id}.json"
        json_path.write_text(json.dumps(graph_to_dict(G), indent=2))
        print(f"Saved json: {json_path}")

        processed += 1
        if processed >= N_TABLES:
            break
    except Exception as e:
        print(f"Table {i} failed: {e}")

print(f"\n{'='*60}")
print(f"Done. Processed {processed} tables.")
print(f"Visualisations saved to {VIS_DIR.resolve()}/")
print(f"JSON graphs saved to {JSON_DIR.resolve()}/")
