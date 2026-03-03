import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from tabdistances.data.cleantablib.dataset import CleanTabLibDataset
from tabdistances.graph.builder import build_graph

import json
import random
import networkx as nx

N_TABLES = 4000
OUT_DIR  = Path("graphs")
JSON_DIR = OUT_DIR / "json"
JSON_DIR.mkdir(parents=True, exist_ok=True)


def graph_to_dict(G: nx.DiGraph) -> dict:
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
processed = len(list(JSON_DIR.glob("*.json")))
if processed:
    print(f"Resuming: {processed} graphs already in {JSON_DIR}/")

for i, table_id, df in dataset.iter_tables():
    try:
        if df.shape[1] < 2 or random.random() >= 0.005:
            continue

        safe_id = table_id.replace("/", "_")
        json_path = JSON_DIR / f"{safe_id}.json"
        if json_path.exists():
            continue

        G = build_graph(df, threshold=0.7)

        json_path.write_text(json.dumps(graph_to_dict(G), indent=2))
        print(f"[{processed+1}/{N_TABLES}] {table_id}  ({df.shape[0]}×{df.shape[1]})  "
              f"{G.number_of_nodes()} nodes  {G.number_of_edges()} edges  -> {json_path}")

        processed += 1
        if processed >= N_TABLES:
            break
    except Exception as e:
        print(f"Table {i} failed: {e}")

print(f"\nDone. {processed} tables written to {JSON_DIR.resolve()}/")
