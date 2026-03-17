import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
from datasets import load_dataset

N_TABLES = 5

ds = load_dataset("alexodavies/cleantablib", "stage4", streaming=True, verification_mode="no_checks")

for i, example in enumerate(ds["train"]):
    print(f"\n{'='*60}")
    print(f"Table {i}")

    # Show all keys in the example
    print(f"Example keys: {list(example.keys())}")

    # Show non-arrow metadata fields
    for k, v in example.items():
        if k != "arrow_bytes":
            print(f"  {k}: {v!r}")

    # Decode the arrow table and inspect schema + dtypes
    reader = pa.RecordBatchStreamReader(example["arrow_bytes"])
    arrow_table = reader.read_all()

    print(f"\nArrow schema:")
    for field in arrow_table.schema:
        print(f"  {field.name!r}: {field.type}  (metadata: {field.metadata})")

    print(f"\nSchema-level metadata: {arrow_table.schema.metadata}")

    df = arrow_table.to_pandas()
    print(f"\nPandas dtypes:")
    print(df.dtypes.to_string())

    print(f"\nSample data:")
    print(df.head(3).to_string())

    if i + 1 >= N_TABLES:
        break
