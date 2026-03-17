"""Evaluate a discriminator model on the three table sources (1-vs-1).

Loads DataFrames from the Parquet export directories and runs one binary
discriminator per pair of sources, reporting accuracy and per-class metrics
for each comparison.

Usage
-----
Run from the repository root after populating the Parquet directories with the
export scripts::

    python cleantab-export.py
    python tabicl-export.py
    python tabfm-export.py
    python t4-export.py
    python discriminate.py

The MODEL variable below is the only thing to change when swapping in a
concrete discriminator implementation.
"""

import sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from sklearn.metrics import classification_report

from tabdistances.discriminator.dataset import DataFrameDataset
from tabdistances.discriminator.models.tabicl_disc import SmallTabICLDiscriminator


import wandb

# ------------------------------------------------------------------
# Swap in any BaseDiscriminator subclass here.
# ------------------------------------------------------------------
MODEL = SmallTabICLDiscriminator

SOURCES = {
    # "cleantablib": Path("tables/parquet"),
    "tabicl":      Path("tables-tabicl/parquet"),
    "tabfm":       Path("tables-tabfm/parquet"),
    "t4":          Path("tables-t4/parquet"),
}
TEST_SIZE    = 0.2
RANDOM_STATE = 42

print(f"sources: {SOURCES}")


# ------------------------------------------------------------------
# Check all sources are populated
# ------------------------------------------------------------------
missing = [name for name, path in SOURCES.items() if not any(path.glob("*.parquet"))]
if missing:
    print("No tables found for:", ", ".join(missing))
    print("Run the export scripts first:")
    print("  python cleantab-export.py")
    print("  python tabicl-export.py")
    print("  python tabfm-export.py")
    print("  python t4-export.py")
    sys.exit(1)

# ------------------------------------------------------------------
# One binary discriminator per pair
# ------------------------------------------------------------------
for name_a, name_b in combinations(SOURCES, 2):
    print(f"\n{'='*60}")
    print(f"  {name_a}  vs  {name_b}")
    print(f"{'='*60}")

    wandb.init(entity = "hierarchical-diffusion",
           project = "tables-discriminator-tabicl",
           name = f"{name_a}-vs-{name_b}")

    dataset = DataFrameDataset({
        name_a: (SOURCES[name_a], 0),
        name_b: (SOURCES[name_b], 1),
    })
    print(f"Total: {len(dataset)} tables")

    X_train, X_test, y_train, y_test = dataset.train_test_split(
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    model = MODEL()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    wandb.log({"Final Accuracy": accuracy})
    print(f"Accuracy: {accuracy:.3f}\n")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=[name_a, name_b]))
