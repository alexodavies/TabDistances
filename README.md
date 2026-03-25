# TabDistances

Featurise tabular datasets and measure distances between collections of real and synthetic tables.

Based on Davies et. al. (2026), *Mind the Gap? A Comparison of Pre-Training Datasets for Tabular Foundation Models*.

## Idea

Each table is encoded as a fixed-length 70-dimensional feature vector capturing distributional shape and inter-column structure:

- **Columns** are typed (NUMERIC / CATEGORICAL) and encoded as 30-bin cumulative histograms
- **Scalar features** summarise skewness, kurtosis, pairwise correlations, missingness, and column-count
- **Distances** are computed via k-NN over the feature vectors to measure coverage between dataset collections

## Setup

```bash
git clone https://github.com/alexodavies/TabDistances.git
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
    typing.py           # Infer column types (NUMERIC / CATEGORICAL)
    column_features.py  # Encode column type and histogram as feature vectors
  data/
    __init__.py         # iter_prior_tables() — SCMPrior → iter_tables() interface
    priors/tabicl/
      dataset.py        # Synthetic dataset generator via SCM priors (TabICL)
    tabfm/
      dataset.py        # Iterator over TabFM pre-training datasets (Kaggle CSVs)
    t4/
      dataset.py        # Iterator over T4 web-scraped tables (HuggingFace)
  discriminator/
    features.py         # df_to_vec() — 70-D aggregate feature vector; featurize_tables()
    base.py             # BaseDiscriminator ABC
    models/             # RFDiscriminator, AggregateDiscriminator
  analysis/
    distances.py        # compute_nn_distances(), knn_coverage()
    plotting.py         # PAPER_RC, stairs_filled(), grid plot helpers
  config.py             # Shared paths (Paths) and defaults (Defaults, GRID_PARAM_SPACE)
```

## Data sources

| Source | Type | Description |
|---|---|---|
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

### Iterate over datasets

```python
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

# T4 web-scraped tables (HuggingFace streaming — no local data needed)
from tabdistances.data.t4.dataset import T4Dataset
for i, table_id, df in T4Dataset().iter_tables():
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
export/          Data export scripts (TabFM from Kaggle, exports to parquet)
benchmark/       TabICL vs Tabula-8B benchmark
tests/           pytest unit tests
notebooks/       Jupyter notebooks (quickstart.ipynb)
```

## Development

```bash
pytest tests/
```
