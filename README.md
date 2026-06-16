# TabDistances

Featurise tabular datasets and measure distances between collections of real and synthetic tables.

Davies et. al. (2026), *Mind the Gap? A Comparison of Pre-Training Datasets for Tabular Foundation Models*. 

https://arxiv.org/abs/2605.06343

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
    dataset.py          # DataFrame dataset loading/splitting for the discriminator
    loss.py             # Combined quality score for synthetic data evaluation
    models/             # RFDiscriminator, AggregateDiscriminator, SmallTabICLDiscriminator
  model/
    # Tabular transformer components adapted from TabICL/TabDPT, used by
    # SmallTabICLDiscriminator (discriminator/models/tabicl_disc.py)
    embedding.py         # ColEmbedding
    interaction.py       # RowInteraction
    preprocessing.py     # TransformToNumerical
    attention.py, encoders.py, layers.py, rope.py, ssmax.py, kv_cache.py,
    inference.py, inference_config.py, sklearn_utils.py
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
analysis/        Research scripts: pairwise-tables.py, pairwise_col_corr.py,
                 characterise.py, coverage_analysis.py, coverage_ablation.py,
                 correlation_coverage.py, discriminate.py, example-coverage.py
optimisation/    Hyperparameter search: bayesian.py, genetic.py,
                 basic-bayesian.py, basic-genetic.py, pareto-coverage.py,
                 pareto-best-analysis.py, analyse-pareto-coverage.py
grid/            Large-scale grid search for Isambard HPC (Slurm array jobs)
export/          Data export scripts (TabFM from Kaggle, exports to parquet)
benchmark/       TabICL vs Tabula-8B benchmark, plus rank-vs-coverage analysis:
                 TALENT.py, rank_vs_coverage_*.py (tables/columns/correlations/
                 tabicl_emb/combined_{talent,tabarena,pymfe}.py),
                 coverage_correlation_summary.py, partial_correlation_analysis.py
tests/           pytest unit tests
notebooks/       Jupyter notebooks (quickstart.ipynb)
```

## Reproducing the paper

The experiments in *Mind the Gap?* follow a linear pipeline. Each research question (RQ) builds on cached feature vectors produced in the data step.

### 0. Prerequisites

```bash
git clone <repo>
cd TabDistances
pip install -e ".[dev]"
```

Fetch all three corpora (see **Setup** above for per-source flags and credential requirements):

```bash
python fetch_data.py   # TabFM (Kaggle), T4 (HuggingFace), TabICL (generated locally)
```

---

### RQ1 & RQ2 — Pairwise distributional comparisons (Section 4)

*Do synthetic priors cover real tables? Do curated datasets cover web-scraped tables?*

Run the pairwise coverage/discriminator analysis across all three source pairs (ICL vs FM, ICL vs T4, T4 vs FM):

```bash
python analysis/pairwise-tables.py
```

Key flags:

- `--mode full scalar_only hist_only` — feature ablations (default: all three)
- `--match-icl-cols` — resample ICL column counts to match the reference distribution

This logs Bootstrap AUC, k-NN Recall/Precision, PCA projections, and feature importances to W&B and writes figures locally.

To reproduce the interpretable characterisation plots (row/column counts, missingness, skewness, correlation distributions — Appendix Fig 6 / Fig 9):

```bash
python analysis/characterise.py
```

---

### RQ3 — Prior optimisation (Section 5)

*Can the TabICL prior be tuned to cover real tables?*

**Bayesian optimisation** (TPE, 1 000 trials per source, 50 independent restarts):

```bash
python optimisation/bayesian.py
```

**Genetic algorithm** (population 16, 100 generations):

```bash
python optimisation/genetic.py
```

**Grid search — local** (G = 6 resolution, ~86 k configurations; intended for a multi-core machine):

```bash
python grid/basic-grid-search.py
```

**Grid search — Isambard HPC** (Slurm array across 6 nodes, ~23 hrs wall-clock as reported in the paper):

```bash
# Edit CONDA_ENV and WORK_DIR in grid/grid-isambard.sh first, then:
bash grid/grid-submit.sh
```

This submits an array job and automatically queues `grid/grid-combine.sh` to merge per-node CSVs once all tasks finish.

After the grid search, evaluate the top-K configurations at full scale:

```bash
python analysis/coverage_analysis.py --config grid-results-<id>/combined_results_t4.csv
```

To reproduce the Pareto-front analysis over the coverage+AUC loss (Figure 3):

```bash
python optimisation/pareto-coverage.py
python optimisation/analyse-pareto-coverage.py
python optimisation/pareto-best-analysis.py
```

---

### RQ4 — Proximity to prior vs downstream performance (Section 6.2)

*Does a TALENT dataset's distance from the TabICL prior predict TabICL's rank on that dataset?*

**Step 1 — Collect benchmark rankings.** Download `benchmark_dataset.zip` from the link in `TALENT-main/example_datasets/readme.md`, then:

```bash
pip install -r benchmark/requirements_benchmark.txt

# Smoke test on the bundled example datasets
python benchmark/tabicl_vs_tabula8b.py \
    --dataset_path TALENT-main/example_datasets \
    --output benchmark/results.csv \
    --seeds 1 --max_context 32

# Full run
python benchmark/tabicl_vs_tabula8b.py \
    --dataset_path /path/to/benchmark_datasets \
    --output benchmark/results.csv \
    --seeds 3
```

**Step 2 — Compute rank-vs-distance plots** (Figure 4) across the three feature representations:

```bash
python benchmark/rank_vs_coverage_tables.py          # full 70-D feature set
python benchmark/rank_vs_coverage_columns.py         # cumulative column histograms
python benchmark/rank_vs_coverage_correlations.py    # correlation histograms
python benchmark/rank_vs_coverage_combined_talent.py # combined 3-panel figure (TALENT datasets)
```

Each script accepts `--talent-dir`, `--tabicl-dir`, `--mode`, and `--k` flags; run with `--help` for details.

**Step 3 (optional) — Replicate the analysis on TabArena datasets.**

TabArena covers 51 real-world datasets with 16 methods including TabICL.  It does not require running any models — only the evaluation-only install is needed (no AutoGluon, no GPU).

**Prerequisites** (one-time):

```bash
# 1. Generate the TabICL synthetic tables (needed for the ICL prior caches)
python fetch_data.py --icl

# 2. Install the tabarena evaluation package
git clone https://github.com/autogluon/tabarena.git
cd tabarena && uv pip install --prerelease=allow -e "./tabarena" && cd ..

# 3. Install openml for dataset download
pip install openml
```

**Run the analysis:**

```bash
python benchmark/rank_vs_coverage_combined_tabarena.py \
    --download-results \
    --download-datasets \
    --output-dir results/rank-coverage-tabarena
```

On first run the script:

1. Downloads TabArena results via the tabarena package
2. Resolves the 51 datasets via the `tabarena-v0.1` OpenML benchmark suite (exact task IDs, so names match the results CSV)
3. Builds the TabICL prior caches from `tables-tabicl/parquet/` if not already present from the TALENT pipeline
4. Computes coverage and writes figures to `results/rank-coverage-tabarena/`

All intermediate artefacts are cached so subsequent runs are fast.  If you already have a results CSV saved locally, skip `--download-results` and pass `--results-csv /path/to/file.csv` instead.

---

### Expected outputs

| Experiment | Primary outputs |
| --- | --- |
| Pairwise comparisons | Bootstrap AUC table (Table 1), PCA plots (Figs 1–2) |
| Characterisation | Distribution panels (Appendix) |
| Bayesian / genetic search | Convergence curves, Pareto front (Fig 3) |
| Grid search | AUC distribution (Fig 8), top-5 validated configs (Table 1) |
| Rank vs coverage | k-NN histogram + regression plots (Fig 4) |

## Development

```bash
pytest tests/
```
