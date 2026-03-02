import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tabdistances.data.tabfm.dataset import TabFMDataset
from tabdistances.graph.builder import build_graph
from tabdistances.graph.visualise import draw_graph
from tabdistances.columns.typing import infer_col_types

import json
import networkx as nx

N_TABLES  = 50
DATA_PATH = Path("data/tabfm")
OUT_DIR = Path("graphs-tabfm")
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


dataset = TabFMDataset(data_path=DATA_PATH, split="both")
processed = 0

for i, dataset_name, df in dataset.iter_tables():
    try:
        if df.shape[1] <= 2:
            continue

        print(f"\n{'='*60}")
        print(f"Dataset {i}: {dataset_name}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"Columns: {list(df.columns)}")
        print(df.head().to_string())

        col_types = infer_col_types(df)
        print(f"Types: { {c: t.value for c, t in col_types.items()} }")

        G = build_graph(df, threshold=0.7, cardinality_threshold=0.2)
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if G.number_of_edges() > 0:
            edges = sorted(G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)
            print("Top edges by weight:")
            for u, v, data in edges[:5]:
                print(f"  {u!r} — {v!r}: {data['weight']:.3f}")

        safe_name = dataset_name.replace("/", "_")

        fig, ax = plt.subplots(figsize=(12, 8))
        draw_graph(G, ax=ax, title=f"{dataset_name}  ({df.shape[0]} rows × {df.shape[1]} cols)")
        vis_path = VIS_DIR / f"{safe_name}.png"
        fig.savefig(vis_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved vis: {vis_path}")

        json_path = JSON_DIR / f"{safe_name}.json"
        json_path.write_text(json.dumps(graph_to_dict(G), indent=2))
        print(f"Saved json: {json_path}")

        processed += 1
        if processed >= N_TABLES:
            break

    except Exception as e:
        print(f"Dataset {i} ({dataset_name}) failed: {e}")

print(f"\n{'='*60}")
print(f"Done. Processed {processed} tables.")
print(f"Visualisations saved to {VIS_DIR.resolve()}/")
print(f"JSON graphs saved to {JSON_DIR.resolve()}/")
