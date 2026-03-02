# TabDistances

Represent tabular datasets as column-relationship graphs and measure distances between collections of real and synthetic tables.

Based on Davies (2024), *Optimising Synthetic Generators*.

## Idea

Each table becomes a directed graph where:

- **Nodes** = columns, with features encoding type and value distribution (30-bin cumulative histogram)
- **Edges** = pairwise predictive relationships: sqrt(R²) for numeric targets, sqrt(McFadden pseudo-R²) for categorical targets, thresholded to keep only strong relationships

Distances between dataset collections are then computed over these graph representations.

## Package structure

```
tabdistances/
  columns/
    typing.py         # Infer column types (NUMERIC / CATEGORICAL)
    node_features.py  # Encode column type and histogram as node features
    edge_features.py  # Pairwise predictive weights (R², pseudo-R²)
  graph/
    builder.py        # DataFrame → nx.DiGraph
    visualise.py      # Draw column-relationship graphs
  data/
    cleantablib/
      dataset.py      # Streaming iterator over alexodavies/cleantablib (HuggingFace)
    priors/tabicl/
      dataset.py      # Synthetic dataset generator via SCM priors (TabICL)
      ...
    tabfm/
      dataset.py      # Iterator over TabFM pre-training datasets (Kaggle CSVs)
      downloader.py   # Download TabFM CSVs from Kaggle
      meta_info/      # Bundled dataset schemas (383 JSON files)
      database_config/# Bundled pretrain split lists
      datasets_url/   # Bundled Kaggle URL mapping
```

## Data sources

| Source | Type | Description |
|---|---|---|
| **CleanTabLib** | Real | ~10k real-world tables streamed from `alexodavies/cleantablib` on HuggingFace |
| **TabICL** | Synthetic | Diverse synthetic tables generated via MLP/tree structural causal models |
| **TabFM** | Real | 338 Kaggle datasets from the Industrial Foundation Models pre-training corpus |

## Setup

```bash
pip install -e ".[dev]"
```

For TabFM datasets, also install the Kaggle CLI and add credentials:

```bash
pip install kaggle
# Place API token at ~/.kaggle/kaggle.json (chmod 600)
python download_tabfm.py
```

## Usage

### Build a graph from any DataFrame

```python
import pandas as pd
from tabdistances.graph.builder import build_graph

df = pd.read_csv("my_table.csv")
G = build_graph(df, threshold=0.7)
```

`build_graph` parameters:

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.0` | Minimum edge weight to include |
| `cardinality_threshold` | `None` | Reclassify low-cardinality float64 columns as CATEGORICAL |
| `max_rows` | `2000` | Row cap before fitting edge weights (random sample) |
| `max_cat_classes` | `20` | Reclassify high-cardinality columns as NUMERIC |

### Iterate over datasets

```python
# Real-world (HuggingFace streaming)
from tabdistances.data.cleantablib.dataset import CleanTabLibDataset
for i, table_id, df in CleanTabLibDataset().iter_tables():
    ...

# Synthetic (generated on the fly)
from tabdistances.data.priors.tabicl.dataset import SCMPrior
prior = SCMPrior(batch_size=8, prior_type="mix_scm")
X, y, d, seq_lens, train_sizes = prior.get_batch()

# TabFM pre-training corpus (requires downloaded CSVs)
from tabdistances.data.tabfm.dataset import TabFMDataset
for i, name, df in TabFMDataset(data_path="data/tabfm").iter_tables():
    ...
```

## Scripts

| Script | Output | Description |
|---|---|---|
| `download_tabfm.py` | `data/tabfm/*.csv` | Download TabFM CSVs from Kaggle |
| `cleantab-test.py` | `graphs/vis/` + `graphs/json/` | Graphs from CleanTabLib (vis + JSON) |
| `tabicl-test.py` | `graphs-tabicl/vis/` + `.../json/` | Graphs from TabICL (vis + JSON) |
| `test_tabfm.py` | `graphs-tabfm/vis/` + `.../json/` | Graphs from TabFM (vis + JSON) |
| `cleantab-export.py` | `graphs/json/` | JSON only — no visualisation |
| `tabicl-export.py` | `graphs-tabicl/json/` | JSON only — no visualisation |
| `tabfm-export.py` | `graphs-tabfm/json/` | JSON only — no visualisation |

## Development

```bash
pytest tests/
```
