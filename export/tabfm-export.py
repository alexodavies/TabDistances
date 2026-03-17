import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from tabdistances.data.tabfm.dataset import TabFMDataset

N_TABLES    = 100000
DATA_PATH   = Path("data/tabfm")
OUT_DIR     = Path("tables-tabfm")
PARQUET_DIR = OUT_DIR / "parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


dataset   = TabFMDataset(data_path=DATA_PATH, split="both")
processed = len(list(PARQUET_DIR.glob("*.parquet")))
if processed:
    print(f"Resuming: {processed} tables already in {PARQUET_DIR}/")

for i, dataset_name, df in dataset.iter_tables():
    try:
        if df.shape[1] < 2:
            continue

        safe_name    = dataset_name.replace("/", "_")
        parquet_path = PARQUET_DIR / f"{safe_name}.parquet"
        if parquet_path.exists():
            continue

        # High-cardinality string columns need to be numeric for the model encoder.
        for col in df.select_dtypes(include="object").columns:
            if df[col].nunique() > 20:
                codes, _ = pd.factorize(df[col])
                s = codes.astype(float)
                s[s == -1] = np.nan
                df[col] = s

        df.to_parquet(parquet_path, index=False)
        print(f"[{processed+1}/{N_TABLES}] {dataset_name}  ({df.shape[0]}×{df.shape[1]})  -> {parquet_path}")

        processed += 1
        if processed >= N_TABLES:
            break
    except Exception as e:
        print(f"Dataset {i} ({dataset_name}) failed: {e}")

print(f"\nDone. {processed} tables written to {PARQUET_DIR.resolve()}/")
