# Test suite reference

49 tests across 5 files. Run with `pytest tests/`.

---

## test_typing.py — basic column type inference

| Test | What it checks |
|---|---|
| `test_float64_is_numeric` | `float64` column → `NUMERIC` |
| `test_int32_is_categorical` | `int32` column → `CATEGORICAL` |
| `test_int64_is_categorical` | `int64` column → `CATEGORICAL` |
| `test_infer_col_types_returns_dict` | `infer_col_types` returns correct types for a mixed DataFrame |

---

## test_typing_edge_cases.py — reclassification heuristics

| Test | What it checks |
|---|---|
| `test_low_cardinality_float_reclassified_as_categorical` | `float64` with few unique values (ratio < 0.2) → `CATEGORICAL` |
| `test_high_cardinality_float_stays_numeric` | `float64` with all-unique values stays `NUMERIC` |
| `test_cardinality_threshold_none_disables_reclassification` | `cardinality_threshold=None` forces `float64` → `NUMERIC` regardless |
| `test_high_cardinality_int_reclassified_as_numeric` | Integer column with > 20 unique values → `NUMERIC` |
| `test_low_cardinality_int_stays_categorical` | Integer column with few unique values stays `CATEGORICAL` |
| `test_max_cat_classes_none_disables_reclassification` | `max_cat_classes=None` keeps high-cardinality integers as `CATEGORICAL` |
| `test_string_column_is_categorical` | String column → `CATEGORICAL` |
| `test_high_cardinality_string_reclassified_as_numeric` | String column with > 20 unique values → `NUMERIC` |
| `test_infer_col_types_all_columns_present` | All column names appear in the output dict |
| `test_infer_col_types_empty_dataframe` | Empty DataFrame → empty dict (no error) |

---

## test_column_features.py — column encoding

| Test | What it checks |
|---|---|
| `test_numeric_type_int` | `get_type_int(NUMERIC)` → `[0.0]` |
| `test_categorical_type_int` | `get_type_int(CATEGORICAL)` → `[1.0]` |
| `test_type_int_same_dimension` | Both type encodings have the same shape |
| `test_type_int_returns_copy` | Mutating the returned array does not affect subsequent calls |
| `test_histogram_length` | `get_histogram` returns a 30-element array |
| `test_histogram_is_cumulative` | Output is a CDF: last bin = 1.0, all bins non-decreasing |
| `test_histogram_empty_column` | All-NaN column → 30 zeros (no error) |

---

## test_features.py — feature extraction and caching

### `df_to_vec`

| Test | What it checks |
|---|---|
| `test_df_to_vec_output_shape` | Returns a vector of length `FEATURE_DIM` (70) |
| `test_df_to_vec_empty_df` | Empty DataFrame → 70 zeros |
| `test_df_to_vec_no_numeric_columns` | All-categorical DataFrame → valid vector, no NaNs |
| `test_df_to_vec_type_ratio` | Feature index 1 matches the known fraction of categorical columns |
| `test_df_to_vec_missing_rate` | Feature index 2 matches the known fraction of missing cells |
| `test_df_to_vec_no_nans_in_output` | Output contains no NaN values for a clean numeric DataFrame |
| `test_df_to_vec_deterministic` | Same DataFrame produces identical output on repeated calls |

### `FEATURE_NAMES` / `FEATURE_MODES`

| Test | What it checks |
|---|---|
| `test_feature_names_length` | `FEATURE_NAMES` has exactly `FEATURE_DIM` entries |
| `test_select_features_full` | `"full"` mode excludes `n_cols` (index 0), returns `FEATURE_DIM - 1` features |
| `test_select_features_scalar_only` | `"scalar_only"` mode excludes `n_cols`, all names are non-histogram |
| `test_select_features_hist_only` | `"hist_only"` mode returns only `hist_*` named features |
| `test_select_features_invalid_mode` | Unknown mode raises `ValueError` |

### `featurize_tables`

| Test | What it checks |
|---|---|
| `test_featurize_tables_shape` | Returns `(n_tables,)` names and `(n_tables, FEATURE_DIM)` array |
| `test_featurize_tables_names_match_rows` | Length of names equals number of feature rows |
| `test_featurize_tables_cache_write_and_load` | Cache file is written; reloading with an empty iterator returns the same results |
| `test_featurize_tables_cache_no_recompute` | Re-feeding the same tables does not grow the cache beyond the original count |
| `test_featurize_tables_empty_source` | Empty iterator with no cache → empty names list and `(0, FEATURE_DIM)` array |

---

## test_analysis.py — k-NN distances and coverage

### `compute_nn_distances`

| Test | What it checks |
|---|---|
| `test_compute_nn_distances_shape` | Returns shape `(N,)` for N query points |
| `test_compute_nn_distances_non_negative` | All distances ≥ 0 |
| `test_compute_nn_distances_closer_when_same_distribution` | Mean distance to matched distribution < mean distance to shifted distribution |
| `test_compute_nn_distances_nan_input` | NaN query features produce finite (not NaN) distances |
| `test_compute_nn_distances_no_scale` | `scale=False` path runs without error and returns valid distances |
| `test_compute_nn_distances_k_clipped_to_ref_size` | k larger than the reference set does not raise |

### `knn_coverage`

| Test | What it checks |
|---|---|
| `test_knn_coverage_returns_two_floats` | Returns a `(float, float)` tuple |
| `test_knn_coverage_range` | Both recall and precision are in `[0, 1]` |
| `test_knn_coverage_identical_distributions` | Same-distribution real/synthetic → both metrics > 0.5 |
| `test_knn_coverage_disjoint_distributions` | Synthetic 100 units away from real → recall < 0.1 |
| `test_knn_coverage_perfect_overlap` | Identical real and synthetic sets → recall and precision both > 0.9 |
