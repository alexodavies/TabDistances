# TabDistances

Represent tabular datasets as column-relationship graphs and measure distances between collections of real and synthetic tables.

Based on Davies (2024), *Optimising Synthetic Generators*.

## Idea

Each table becomes a directed graph where:

- **Nodes** = columns, with features encoding type and value distribution (30-bin cumulative histogram)
- **Edges** = pairwise predictive relationships: sqrt(R²) for numeric targets, sqrt(McFadden pseudo-R²) for categorical targets, thresholded to keep only strong relationships

Distances between dataset collections are computed over both these graph representations and aggregate 70-dimensional feature vectors that capture distributional shape and inter-column structure.

## Setup

```bash
git clone <repo>
cd TabDistances
pip install -e ".[dev]"
```

To download datasets, run from the repo root:

```bash
python fetch_data.py          # all sources (TabFM, T4, TabICL)
python fetch_data.py --tabfm  # TabFM only (requires Kaggle CLI + ~/.kaggle/kaggle.json)
python fetch_data.py --t4     # T4 only
python fetch_data.py --icl    # TabICL only (generated locally, no download)
```

## Package structure

```
tabdistances/
  columns/
    typing.py         # Infer column types (NUMERIC / CATEGORICAL)
    node_features.py  # Encode column type and histogram as node features
    edge_features.py  # Pairwise predictive weights (R², pseudo-R²)
  graph/
    builder.py        # DataFrame → nx.DiGraph
  data/
    __init__.py       # iter_prior_tables() — SCMPrior → iter_tables() interface
    cleantablib/
      dataset.py      # Streaming iterator over alexodavies/cleantablib (HuggingFace)
    priors/tabicl/
      dataset.py      # Synthetic dataset generator via SCM priors (TabICL)
    tabfm/
      dataset.py      # Iterator over TabFM pre-training datasets (Kaggle CSVs)
  discriminator/
    features.py       # df_to_vec() — 70-D aggregate feature vector; featurize_tables()
    base.py           # BaseDiscriminator ABC
    models/           # RFDiscriminator, AggregateDiscriminator
  analysis/
    distances.py      # compute_nn_distances(), knn_coverage()
    plotting.py       # PAPER_RC, stairs_filled(), grid plot helpers
  config.py           # Shared paths (Paths) and defaults (Defaults, GRID_PARAM_SPACE)
```

## Data sources

| Source | Type | Description |
|---|---|---|
| **CleanTabLib** | Real | ~10k real-world tables streamed from `alexodavies/cleantablib` on HuggingFace |
| **TabICL** | Synthetic | Diverse synthetic tables via MLP/tree structural causal models |
| **TabFM** | Real | 338 Kaggle datasets from the Industrial Foundation Models pre-training corpus |
| **T4** | Real | ~3.1M web-scraped tables streamed from HuggingFace |

## Usage

### Quick imports

```python
import tabdistances as td

# Feature extraction
vec = td.df_to_vec(df)                     # 70-D feature vector
names, feats = td.featurize_tables(        # batch + cache
    source.iter_tables(),
    cache_path=td.config.Paths.CACHE / "my_feats.npz",
)

# Column type inference
types = td.infer_col_types(df)             # dict[str, ColumnType]
```

### Build a graph from any DataFrame

```python
from tabdistances.graph.builder import build_graph

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
# Real-world (HuggingFace streaming — no local data needed)
from tabdistances.data.cleantablib.dataset import CleanTabLibDataset
for i, table_id, df in CleanTabLibDataset().iter_tables():
    ...

# Synthetic (generated on the fly)
from tabdistances.data.priors.tabicl.dataset import SCMPrior
from tabdistances.data import iter_prior_tables

prior = SCMPrior(batch_size=8, prior_type="mix_scm")
for i, name, df in iter_prior_tables(prior, n=500):
    ...                   # same interface as real-data loaders

# TabFM pre-training corpus (requires downloaded CSVs)
from tabdistances.data.tabfm.dataset import TabFMDataset
for i, name, df in TabFMDataset(data_path="data/tabfm").iter_tables():
    ...
```

### Compute coverage metrics

```python
from tabdistances.analysis.distances import compute_nn_distances, knn_coverage

# k-NN distance: how far are TALENT datasets from the TabICL prior?
dists = compute_nn_distances(talent_feats, icl_feats, k=5)

# Coverage: how well does synthetic data cover the real distribution?
recall, precision = knn_coverage(real_feats, synth_feats, k=5, quantile=0.95)
```

### Shared configuration

```python
from tabdistances.config import Paths, Defaults

cache_path = Paths.CACHE / "tabfm_feats.npz"
df_sample  = df.iloc[: Defaults.MAX_ROWS]
```

## Repository layout

```
analysis/        Research scripts: rank_vs_coverage*.py, characterise.py,
                 coverage_analysis.py, correlation_coverage.py, …
optimisation/    Hyperparameter search: genetic.py, bayesian.py,
                 pareto-coverage.py, …
grid/            Large-scale grid search for Isambard HPC (Slurm array jobs)
export/          Data download scripts (TabFM from Kaggle, exports to parquet)
misc/            Exploratory test scripts
benchmark/       TabICL vs Tabula-8B benchmark
tests/           pytest unit tests
notebooks/       Jupyter notebooks (quickstart.ipynb)
```

## Development

```bash
pytest tests/
```
