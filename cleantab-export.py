import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import random

from tabdistances.data.cleantablib.dataset import CleanTabLibDataset

N_TABLES    = 500000
OUT_DIR     = Path("tables")
PARQUET_DIR = OUT_DIR / "parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


dataset   = CleanTabLibDataset(config="stage4", split="train", deduplicate=True)
processed = len(list(PARQUET_DIR.glob("*.parquet")))
if processed:
    print(f"Resuming: {processed} tables already in {PARQUET_DIR}/")

for i, table_id, df in dataset.iter_tables():
    try:
        if df.shape[1] < 2 or random.random() >= 0.05:
            continue

        safe_id      = table_id.replace("/", "_")
        parquet_path = PARQUET_DIR / f"{safe_id}.parquet"
        if parquet_path.exists():
            continue

        df.to_parquet(parquet_path, index=False)
        print(f"[{processed+1}/{N_TABLES}] {table_id}  ({df.shape[0]}×{df.shape[1]})  -> {parquet_path}")

        processed += 1
        if processed >= N_TABLES:
            break
    except Exception as e:
        print(f"Table {i} failed: {e}")

print(f"\nDone. {processed} tables written to {PARQUET_DIR.resolve()}/")
