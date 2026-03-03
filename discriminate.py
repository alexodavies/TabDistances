"""Evaluate a discriminator model on the three graph sources (1-vs-1).

Loads graphs from the JSON export directories and runs one binary
discriminator per pair of sources, reporting accuracy and per-class metrics
for each comparison.

Usage
-----
Run from the repository root after populating the JSON directories with the
export scripts::

    python cleantab-export.py
    python tabicl-export.py
    python tabfm-export.py
    python discriminate.py

The MODEL variable below is the only thing to change when swapping in a
concrete discriminator implementation.
"""

import sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from sklearn.metrics import classification_report

from tabdistances.discriminator.dataset import GraphDataset
from tabdistances.discriminator.models.rf import RFDiscriminator

# ------------------------------------------------------------------
# Swap in any BaseDiscriminator subclass here.
# ------------------------------------------------------------------
MODEL = RFDiscriminator

SOURCES = {
    "cleantablib": Path("graphs/json"),
    "tabicl":      Path("graphs-tabicl/json"),
    "tabfm":       Path("graphs-tabfm/json"),
}
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ------------------------------------------------------------------
# Check all sources are populated
# ------------------------------------------------------------------
missing = [name for name, path in SOURCES.items() if not any(path.glob("*.json"))]
if missing:
    print("No graphs found for:", ", ".join(missing))
    print("Run the export scripts first:")
    print("  python cleantab-export.py")
    print("  python tabicl-export.py")
    print("  python tabfm-export.py")
    sys.exit(1)

# ------------------------------------------------------------------
# One binary discriminator per pair
# ------------------------------------------------------------------
for name_a, name_b in combinations(SOURCES, 2):
    print(f"\n{'='*60}")
    print(f"  {name_a}  vs  {name_b}")
    print(f"{'='*60}")

    dataset = GraphDataset({
        name_a: (SOURCES[name_a], 0),
        name_b: (SOURCES[name_b], 1),
    })
    print(f"Total: {len(dataset)} graphs")

    X_train, X_test, y_train, y_test = dataset.train_test_split(
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    model = MODEL()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.3f}\n")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=[name_a, name_b]))
