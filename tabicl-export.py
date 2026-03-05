import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd
import torch

from tabdistances.data.priors.tabicl.dataset import SCMPrior

N_TABLES     = 500000
BATCH_SIZE   = 8
MIN_FEATURES = 1
OUT_DIR      = Path("tables-tabicl")
PARQUET_DIR  = OUT_DIR / "parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


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

processed = len(list(PARQUET_DIR.glob("*.parquet")))
if processed:
    print(f"Resuming: {processed} tables already in {PARQUET_DIR}/")
batch_idx = 0

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

        table_id     = f"tabicl_batch{batch_idx:04d}_idx{i:02d}"
        parquet_path = PARQUET_DIR / f"{table_id}.parquet"
        if parquet_path.exists():
            continue

        try:
            df.to_parquet(parquet_path, index=False)
            print(f"[{processed+1}/{N_TABLES}] {table_id}  ({df.shape[0]}×{df.shape[1]})  -> {parquet_path}")

            processed += 1
        except Exception as e:
            print(f"Table {table_id} failed: {e}")

        if processed >= N_TABLES:
            break

    batch_idx += 1

print(f"\nDone. {processed} tables written to {PARQUET_DIR.resolve()}/")
