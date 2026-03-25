"""Shared paths and default hyperparameters for the TabDistances project.

All analysis, optimisation, and grid scripts should import from here rather
than defining their own CONFIG blocks, ensuring a single source of truth.

Usage
-----
>>> from tabdistances.config import Paths, Defaults
>>> feats_cache = Paths.CACHE / "my_feats.npz"
>>> df_sample = df.iloc[: Defaults.MAX_ROWS]

HPC / CI overrides
------------------
``grid/grid-isambard.py`` supports environment-variable overrides (TAB_DATA_ROOT,
TAB_OUT_DIR, etc.) for Slurm array jobs.  Those overrides take precedence over
the path constants here — that is intentional and should be preserved.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


class Paths:
    """Repository-relative paths to data, cache, and output directories."""

    ROOT    = _REPO_ROOT
    DATA    = _REPO_ROOT / "data"        # TALENT benchmark splits (N_train.npy etc.)
    CACHE   = _REPO_ROOT / "cache"       # pre-computed feature vector NPZ files
    RESULTS = _REPO_ROOT / "results"     # output figures and CSVs

    # Pre-exported parquet stores (one file per table)
    TABLES_ICL = _REPO_ROOT / "tables-tabicl" / "parquet"
    TABLES_FM  = _REPO_ROOT / "tables-tabfm"  / "parquet"
    TABLES_T4  = _REPO_ROOT / "tables-t4"     / "parquet"


class Defaults:
    """Default hyperparameter values shared across scripts.

    Change a value here and it propagates to every script that imports it.
    """

    #: Row cap applied before featurising a single table.
    MAX_ROWS: int = 512

    #: Number of nearest neighbours used in coverage and distance metrics.
    KNN_K: int = 5

    #: Global random seed for reproducibility.
    RANDOM_STATE: int = 42

    #: Threshold for classifying a column pair as "highly correlated" (|r| > this).
    HIGH_CORR_THRESHOLD: float = 0.7

    #: Fraction-of-unique-values threshold below which a float64 column is
    #: reclassified as CATEGORICAL in :func:`tabdistances.columns.typing.infer_col_types`.
    CARDINALITY_THRESHOLD: float = 0.2

    #: Percentile used to set the k-NN radius in coverage recall/precision.
    COVERAGE_QUANTILE: float = 0.95

    #: Number of real tables to load and cache for discriminator training.
    N_REAL_CACHE: int = 5_000


# ---------------------------------------------------------------------------
# Grid-search parameter space
# (imported by grid/grid-isambard.py and grid/grid-combine.py)
# ---------------------------------------------------------------------------

#: TabICL prior parameter space used for the Isambard HPC grid search.
#:
#: Format: ``{param_name: (type, *spec)}`` where *type* is one of:
#:
#: * ``"int"``   — integer range: spec is ``(lo, hi)`` inclusive
#: * ``"float"`` — float range:   spec is ``(lo, hi)`` inclusive
#: * ``"cat"``   — categorical:   spec is ``([values],)``
#:
#: **Both** ``grid/grid-isambard.py`` and ``grid/grid-combine.py`` import
#: this constant.  Do not define it in both files — edit it once here.
GRID_PARAM_SPACE: dict = {
    # Shared: active for all prior types
    "max_classes":              ("int",   2,    20),
    "cat_prob":                 ("float", 0.0,  0.6),
    "multiclass_ordered_prob":  ("float", 0.0,  1.0),
    "balanced":                 ("cat",   [False, True]),
    "replay_small":             ("cat",   [False, True]),
    # Conditional: prior_type always varies
    "prior_type":               ("cat",   ["mlp_scm", "tree_scm", "mix_scm"]),
    # Conditional: tree_model active for tree_scm and mix_scm only
    "tree_model":               ("cat",   ["decision_tree", "extra_trees",
                                           "random_forest", "xgboost"]),
    # Conditional: mix_prob_mlp active for mix_scm only
    "mix_prob_mlp":             ("float", 0.0,  1.0),
    # Conditional: active for tree_scm and mix_scm only
    "tree_depth_lambda":        ("float", 0.3,  1.0),
    "tree_n_estimators_lambda": ("float", 0.3,  1.0),
}
