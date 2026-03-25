"""tabdistances — represent tabular datasets as column-relationship graphs
and measure distances between collections of real and synthetic tables.

Quick start
-----------
>>> import tabdistances as td
>>> names, feats = td.featurize_tables(source.iter_tables())
>>> print(feats.shape)  # (n_tables, 70)

Data loaders (require optional dependencies)
--------------------------------------------
These are available only when the relevant packages are installed.
The core feature-extraction functions (df_to_vec, featurize_tables) always work.

    from tabdistances.data.cleantablib.dataset import CleanTabLibDataset  # needs: datasets
    from tabdistances.data.tabfm.dataset import TabFMDataset               # needs: kaggle
    from tabdistances.data.t4.dataset import T4Dataset                     # needs: datasets
"""

from tabdistances.columns.typing import ColumnType, infer_col_types
from tabdistances.discriminator.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    df_to_vec,
    featurize_tables,
)

__all__ = [
    # Column utilities
    "ColumnType",
    "infer_col_types",
    # Feature extraction
    "df_to_vec",
    "featurize_tables",
    "FEATURE_NAMES",
    "FEATURE_DIM",
]

# Data loaders — imported conditionally because they need optional dependencies.
# They are added to __all__ only when available so IDEs see them when installed.
try:
    from tabdistances.data.cleantablib.dataset import CleanTabLibDataset  # noqa: E402
    __all__ += ["CleanTabLibDataset"]
except ImportError:
    pass

try:
    from tabdistances.data.tabfm.dataset import TabFMDataset  # noqa: E402
    __all__ += ["TabFMDataset"]
except ImportError:
    pass

try:
    from tabdistances.data.t4.dataset import T4Dataset  # noqa: E402
    __all__ += ["T4Dataset"]
except ImportError:
    pass
