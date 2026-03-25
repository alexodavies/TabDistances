import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

from tabdistances.data.t4.dataset import T4Dataset

N_TABLES    = 5000
OUT_DIR     = Path("tables-t4")
PARQUET_DIR = OUT_DIR / "parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# Set your HF token here (or via HF_TOKEN env var) if the repo requires auth.
HF_TOKEN = None

dataset  = T4Dataset(token=HF_TOKEN)
processed = len(list(PARQUET_DIR.glob("*.parquet")))
if processed:
    print(f"Resuming: {processed} tables already in {PARQUET_DIR}/")

for i, content_hash, df in dataset.iter_tables():
    try:
        if df.shape[1] < 2 or random.random() >= 0.5:
            continue

        parquet_path = PARQUET_DIR / f"{content_hash}.parquet"
        if parquet_path.exists():
            continue

        df.to_parquet(parquet_path, index=False)
        print(f"[{processed+1}/{N_TABLES}] {content_hash}  ({df.shape[0]}×{df.shape[1]})  -> {parquet_path}")

        processed += 1
        if processed >= N_TABLES:
            break
    except Exception as e:
        print(f"Table {i} ({content_hash}) failed: {e}")

print(f"\nDone. {processed} tables written to {PARQUET_DIR.resolve()}/")
