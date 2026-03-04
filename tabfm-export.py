import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from tabdistances.data.tabfm.dataset import TabFMDataset
from tabdistances.graph.builder import build_graph

import json
import numpy as np
import pandas as pd
import networkx as nx

N_TABLES  = 1000
DATA_PATH = Path("data/tabfm")
OUT_DIR   = Path("graphs-tabfm")
JSON_DIR  = OUT_DIR / "json"
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


dataset   = TabFMDataset(data_path=DATA_PATH, split="both")
processed = len(list(JSON_DIR.glob("*.json")))
if processed:
    print(f"Resuming: {processed} graphs already in {JSON_DIR}/")

for i, dataset_name, df in dataset.iter_tables():
    try:
        if df.shape[1] < 2:
            continue

        safe_name = dataset_name.replace("/", "_")
        json_path = JSON_DIR / f"{safe_name}.json"
        if json_path.exists():
            continue

        # High-cardinality string columns are reclassified as NUMERIC by
        # build_graph (max_cat_classes=20) and must be numeric for sklearn.
        # Low-cardinality string categoricals are handled fine by _onehot.
        for col in df.select_dtypes(include="object").columns:
            if df[col].nunique() > 20:
                codes, _ = pd.factorize(df[col])
                s = codes.astype(float)
                s[s == -1] = np.nan
                df[col] = s

        G = build_graph(df, threshold=0.7, cardinality_threshold=0.2)

        json_path.write_text(json.dumps(graph_to_dict(G), indent=2))
        print(f"[{processed+1}/{N_TABLES}] {dataset_name}  ({df.shape[0]}×{df.shape[1]})  "
              f"{G.number_of_nodes()} nodes  {G.number_of_edges()} edges  -> {json_path}")

        processed += 1
        if processed >= N_TABLES:
            break
    except Exception as e:
        print(f"Dataset {i} ({dataset_name}) failed: {e}")

print(f"\nDone. {processed} tables written to {JSON_DIR.resolve()}/")
