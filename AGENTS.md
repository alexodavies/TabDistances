# TabDistances — Agent Reference

Dense reference for LLMs and AI agents working with this codebase. Covers API contracts, data structures, conventions, and experiment reproduction.

---

## Repository layout

```
tabdistances/              Core package
  __init__.py              Public API (see below)
  config.py                Paths, Defaults, GRID_PARAM_SPACE
  columns/
    typing.py              infer_col_type(), infer_col_types(), ColumnType enum
    column_features.py     get_histogram(), get_type_int()
  data/
    __init__.py            iter_prior_tables() — SCMPrior → iter_tables() adapter
    priors/tabicl/
      dataset.py           SCMPrior, Prior (ABC), MLPSCM, TreeSCM
    tabfm/dataset.py       TabFMDataset
    t4/dataset.py          T4Dataset
  discriminator/
    features.py            df_to_vec(), featurize_tables(), select_features(), FEATURE_NAMES
    base.py                BaseDiscriminator ABC
    dataset.py             DataFrameDataset, load_dataframes()
    loss.py                combined_score()
    models/
      aggregate.py         AggregateDiscriminator
      rf.py                RFDiscriminator
      tabicl_disc.py       SmallTabICLDiscriminator
  analysis/
    distances.py           compute_nn_distances(), knn_coverage()
    plotting.py            PAPER_RC, stairs_filled(), grid plot helpers

analysis/                  Research scripts (not part of package)
optimisation/              Prior optimisation scripts
grid/                      HPC grid search (Slurm)
benchmark/                 TALENT benchmark runner
fetch_data.py              Data download entry point
```

---

## Public API (`import tabdistances as td`)

```python
td.ColumnType                    # enum: NUMERIC | CATEGORICAL
td.infer_col_types(df)           # dict[str, ColumnType]
td.df_to_vec(df)                 # np.ndarray, shape (70,), float64
td.featurize_tables(tables, cache_path=None, max_rows=512, desc="...")
                                 # -> (list[str], np.ndarray shape (N, 70))
td.FEATURE_NAMES                 # list[str], length 70
td.FEATURE_DIM                   # 70
```

---

## Configuration (`tabdistances/config.py`)

### `Paths` — all are `pathlib.Path`

| Attribute | Value |
| --- | --- |
| `Paths.ROOT` | repo root |
| `Paths.DATA` | `<root>/data` (TALENT benchmark) |
| `Paths.CACHE` | `<root>/cache` (NPZ feature caches) |
| `Paths.RESULTS` | `<root>/results` |
| `Paths.TABLES_ICL` | `<root>/tables-tabicl/parquet` |
| `Paths.TABLES_FM` | `<root>/tables-tabfm/parquet` |
| `Paths.TABLES_T4` | `<root>/tables-t4/parquet` |

### `Defaults`

| Attribute | Value | Meaning |
| --- | --- | --- |
| `MAX_ROWS` | 512 | Row cap before featurisation |
| `KNN_K` | 5 | Neighbours for coverage |
| `RANDOM_STATE` | 42 | Global seed |
| `CARDINALITY_THRESHOLD` | 0.2 | Unique-value ratio for CATEGORICAL inference |
| `COVERAGE_QUANTILE` | 0.95 | Percentile for adaptive k-NN radius |
| `HIGH_CORR_THRESHOLD` | 0.7 | Absolute Pearson |r| cutoff |
| `N_REAL_CACHE` | 5000 | Real tables cached for discriminator training |

### `GRID_PARAM_SPACE`

Dict of TabICL prior parameters to sweep. Each entry is `{"type": "int"|"float"|"cat", "lo": ..., "hi": ..., "values": [...]}`. Used by `grid/` and `optimisation/` scripts.

---

## Feature vectors (`tabdistances/discriminator/features.py`)

### `df_to_vec(df: pd.DataFrame) -> np.ndarray`

Returns shape **(70,), float64**. Invariant to column/row order.

| Dims | Feature |
| --- | --- |
| 0 | `n_cols` |
| 1 | `type_ratio` — fraction categorical |
| 2 | `missing_rate` — fraction NaN across all cells |
| 3–4 | mean/std of per-column skewness (numeric only) |
| 5–6 | mean/std of per-column kurtosis (numeric only) |
| 7–8 | mean/std of pairwise Pearson |r| (numeric pairs) |
| 9 | `frac_high_corr` — fraction of pairs with |r| > 0.7 |
| 10–39 | Mean of 30-bin cumulative column histograms |
| 40–69 | Std of 30-bin cumulative column histograms |

Edge cases: empty DataFrame → `zeros(70)`. No numeric columns → histogram dims = 0.

### `featurize_tables(tables, cache_path=None, max_rows=512, desc="Featurising tables")`

- `tables`: iterable of `(index: int, name: str, df: pd.DataFrame)`
- Returns `(names: list[str], feats: np.ndarray shape (N, 70))`
- If `cache_path` is set: reads existing NPZ, computes missing rows, writes back.

### `select_features(feats, mode) -> (filtered_feats, feature_names)`

| Mode | Dims | Indices |
| --- | --- | --- |
| `"full"` | 69 | 1–69 (n_cols excluded) |
| `"scalar_only"` | 9 | 1–9 |
| `"hist_only"` | 60 | 10–69 |

### NPZ cache format

```python
data = np.load("path.npz", allow_pickle=True)
names = data["names"].tolist()  # list[str], length N
feats = data["feats"]           # float64, shape (N, 70)
```

---

## Data iterators

All three sources share the same protocol:

```python
def iter_tables() -> Iterator[tuple[int, str, pd.DataFrame]]:
    # index : int  — sequential from 0
    # name  : str  — dataset identifier
    # df    : pd.DataFrame
```

### `TabFMDataset(data_path, split="both")`

- `split`: `"cls"` | `"reg"` | `"both"`
- `data_path`: directory of `<dataset>.csv` files (populated by `fetch_data.py --tabfm`)
- Filters to `num_features + cat_features` columns from bundled `meta_info/` JSON; drops `other_features`.
- Skips missing/corrupt files silently.

### `T4Dataset(split="train", token=None)`

- Streams from HuggingFace `mlfoundations/t4-full` (no local copy needed).
- Table name is Parquet file stem (content hash).
- `max_chunks` kwarg on `iter_tables()` limits how many zip shards are processed.

### `iter_prior_tables(prior: SCMPrior, n: int)`

- Wraps `prior.get_batch()` into the `iter_tables()` protocol.
- Names: `"tabicl_{index:06d}"`.
- Skips zero-length or zero-feature datasets.

### `SCMPrior` key parameters

```python
SCMPrior(
    batch_size=256,
    prior_type="mix_scm",   # "mlp_scm" | "tree_scm" | "mix_scm"
    max_classes=10,
    max_seq_len=1024,
    replay_small=False,     # oversample short sequences
    n_jobs=-1,              # parallel workers
)
```

`prior.get_batch()` returns `(X, y, d, seq_lens, train_sizes)`:
- `X`: Tensor `(B, T, max_features)` or NestedTensor
- `y`: Tensor `(B, T)`
- `d`: Tensor `(B,)` — active feature count per dataset
- `seq_lens`, `train_sizes`: Tensor `(B,)`.

---

## Distance and coverage (`tabdistances/analysis/distances.py`)

### `compute_nn_distances(query_feats, ref_feats, k=5, scale=True) -> np.ndarray`

- NaN/Inf → 0 before computation.
- Min-Max scales to [0, 1] using `ref_feats` as the fitting set.
- Returns shape `(n_query,)` — mean distance to k nearest ref points.

### `knn_coverage(real_feats, synth_feats, k=5, quantile=0.95) -> (recall, precision)`

- `recall`: fraction of real tables covered by synth (A∈B in paper notation).
- `precision`: fraction of synth tables covered by real (B∈A).
- Threshold = `quantile`-th percentile of synth's internal k-NN distances.
- Both values in [0, 1].

---

## Discriminator loss (`tabdistances/discriminator/loss.py`)

### `combined_score(recall, precision, discriminator_auc) -> float`

```
auc_quality = (1 - auc) * 2        # maps [0.5, 1.0] → [1.0, 0.0]
score = sqrt(recall² + precision² + auc_quality²) / sqrt(3)
```

Maximise toward 1.0. Used as objective in Bayesian/genetic optimisation.

---

## Column typing (`tabdistances/columns/typing.py`)

### `infer_col_type(col, cardinality_threshold=0.2, max_cat_classes=20) -> ColumnType`

1. If dtype == float64 and unique_ratio < 0.2 → CATEGORICAL (catches standard-scaled integer codes)
2. If dtype == float64 and unique_ratio >= 0.2 → NUMERIC
3. Otherwise: nunique > max_cat_classes → NUMERIC, else CATEGORICAL

---

## Discriminator models

All subclass `BaseDiscriminator` with `fit(X: list[pd.DataFrame], y: np.ndarray)` and `predict(X) -> np.ndarray`.

### `AggregateDiscriminator(clf)`

Wraps any sklearn estimator. Converts each DataFrame to a 70-D feature vector, stacks, and calls `clf.fit/predict`.

### `RFDiscriminator(n_estimators=300, random_state=42)`

`AggregateDiscriminator` backed by `RandomForestClassifier`.

### `SmallTabICLDiscriminator`

Neural discriminator using TabICL's ColEmbedding + RowInteraction encoder with a linear head.

Key constructor args:

```python
SmallTabICLDiscriminator(
    embed_dim=32,
    col_num_blocks=1, col_nhead=4,
    row_num_blocks=2, row_nhead=4, row_num_cls=2,
    lr=1e-3, epochs=30, batch_size=128,
    max_rows=512, val_frac=0.15,
    device=None,  # auto-detects CUDA
)
```

- `fit()` logs `train/loss`, `val/loss`, `val/roc_auc` to W&B (silent if W&B absent).
- `predict_proba()` returns shape `(n, num_classes)`.
- Internal pre-processing: TransformToNumerical → NaN/Inf → 0 → per-column standardisation (mean=0, std=1, clipped ±5).

---

## DataFrameDataset (`tabdistances/discriminator/dataset.py`)

```python
DataFrameDataset(
    sources: dict[str, tuple[Path, int]],  # {"tabicl": (path, 0), "tabfm": (path, 1)}
    max_rows: int = 1000,
)
# .tables: list[pd.DataFrame]
# .labels: np.ndarray shape (N,)
# .train_test_split(test_size=0.2, random_state=42) -> (X_train, X_test, y_train, y_test)
```

`load_dataframes(parquet_dir, label, max_rows, max_tables, random_state, max_workers=8)` — parallelised Parquet loader, returns `list[tuple[pd.DataFrame, int]]`.

---

## Experiment scripts

| Script | Experiment | Key flags |
| --- | --- | --- |
| `analysis/pairwise-tabes.py` | RQ1 & RQ2: pairwise AUC + coverage | `--mode`, `--match-icl-cols` |
| `analysis/characterise.py` | Structural characterisation plots | — |
| `analysis/coverage_analysis.py` | Top-K grid configs, full-scale validation | `--config <csv>` |
| `analysis/rank_vs_coverage_tables.py` | RQ4: rank vs 70-D distance | `--talent-dir`, `--tabicl-dir`, `--k` |
| `analysis/rank_vs_coverage_columns.py` | RQ4: rank vs col-hist distance | same |
| `analysis/rank_vs_coverage_correlations.py` | RQ4: rank vs corr-hist distance | same |
| `analysis/rank_vs_coverage_combined.py` | Combined 3-panel Fig 4 | same |
| `optimisation/bayesian.py` | RQ3: TPE (1000 trials × 50 restarts) | constants in CONFIG block |
| `optimisation/genetic.py` | RQ3: GA (pop=16, gen=100) | constants in CONFIG block |
| `optimisation/pareto-coverage.py` | RQ3: Bayesian over coverage+AUC loss | — |
| `grid/basic-grid-search.py` | RQ3: local grid (~86k configs) | constants in CONFIG block |
| `grid/grid-submit.sh` | RQ3: Isambard HPC array job | edit `CONDA_ENV`, `WORK_DIR` |
| `benchmark/tabicl_vs_tabula8b.py` | RQ4: TALENT benchmark rankings | `--dataset_path`, `--output`, `--seeds` |

All scripts are run from the **repo root**. Configuration is via constants at the top of each file (no CLI flags unless documented in the script's docstring).

---

## Conventions

- **Caching**: check NPZ → compute missing → write back. `names` and `feats` are always index-aligned.
- **Error handling**: data loaders print and skip on failure; W&B calls are wrapped in `try/except`.
- **Feature scaling**: always Min-Max to [0, 1] using the *reference* set as the fitting set.
- **Seeds**: `Defaults.RANDOM_STATE = 42` used everywhere via `random_state=` kwargs.
- **W&B**: optional. Scripts log to W&B project names derived from the comparison being run; no crash if unavailable.
- **Row cap**: `Defaults.MAX_ROWS = 512` applied before `df_to_vec` in batch featurisation.
- **Feature mode ablation**: always run `"full"`, `"scalar_only"`, `"hist_only"` to validate findings are not feature-set-specific.
