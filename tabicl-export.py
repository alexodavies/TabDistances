import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import json
import numpy as np
import networkx as nx
import pandas as pd
import torch

from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.graph.builder import build_graph

N_TABLES     = 50
BATCH_SIZE   = 8
MIN_FEATURES = 3
OUT_DIR      = Path("graphs-tabicl")
JSON_DIR     = OUT_DIR / "json"
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


prior = SCMPrior(
    batch_size=BATCH_SIZE,
    batch_size_per_gp=4,
    min_features=MIN_FEATURES,
    max_features=100,
    max_classes=10,
    max_seq_len=1024,
    min_train_size=0.1,
    max_train_size=0.9,
    prior_type="mix_scm",
    n_jobs=1,
    device="cpu",
)

processed = 0
batch_idx  = 0

while processed < N_TABLES:
    X, y, d, seq_lens, train_sizes = prior.get_batch()

    for i in range(X.shape[0]):
        n_features = int(d[i].item())
        seq_len    = int(seq_lens[i].item())

        if n_features < MIN_FEATURES or seq_len == 0:
            continue

        X_i = X[i, :seq_len, :n_features].numpy().astype(np.float64)
        y_i = y[i, :seq_len].numpy().astype(int)

        df = pd.DataFrame(X_i, columns=[f"feature_{j}" for j in range(n_features)])
        df["target"] = y_i

        table_id = f"tabicl_batch{batch_idx:04d}_idx{i:02d}"

        try:
            G = build_graph(df, threshold=0.7, cardinality_threshold=0.2)

            json_path = JSON_DIR / f"{table_id}.json"
            json_path.write_text(json.dumps(graph_to_dict(G), indent=2))
            print(f"[{processed+1}/{N_TABLES}] {table_id}  ({df.shape[0]}×{df.shape[1]})  "
                  f"{G.number_of_nodes()} nodes  {G.number_of_edges()} edges  -> {json_path}")

            processed += 1
        except Exception as e:
            print(f"Table {table_id} failed: {e}")

        if processed >= N_TABLES:
            break

    batch_idx += 1

print(f"\nDone. {processed} tables written to {JSON_DIR.resolve()}/")
