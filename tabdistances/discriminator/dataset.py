"""DataFrame dataset loading and splitting for the source discriminator."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _sklearn_split


def load_dataframes(
    parquet_dir: Union[str, Path],
    label: int,
    max_rows: int = 1000,
) -> list[tuple[pd.DataFrame, int]]:
    """Load DataFrames from a Parquet export directory and assign a class label.

    Parameters
    ----------
    parquet_dir : str or Path
        Directory containing ``<name>.parquet`` files produced by the export
        scripts (``cleantab-export.py``, ``tabicl-export.py``, ``tabfm-export.py``).
    label : int
        Class label to assign to every table loaded from this directory.
    max_rows : int, default=1000
        Cap the number of rows per table (random sample without replacement).

    Returns
    -------
    list of (pd.DataFrame, int)
        One pair per Parquet file successfully loaded.
    """
    parquet_dir = Path(parquet_dir)
    pairs: list[tuple[pd.DataFrame, int]] = []

    for path in sorted(parquet_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(path)
            if max_rows and len(df) > max_rows:
                df = df.sample(max_rows, random_state=42)
            pairs.append((df, label))
        except Exception as exc:
            print(f"[LOAD ERROR] {path.name}: {exc}")

    return pairs


class DataFrameDataset:
    """Labelled collection of DataFrames from one or more source directories.

    Parameters
    ----------
    sources : dict[str, tuple[Path, int]]
        Mapping of source name to ``(parquet_dir, label)`` pairs.  Example::

            DataFrameDataset({
                "cleantablib": (Path("tables/parquet"),        0),
                "tabicl":      (Path("tables-tabicl/parquet"), 1),
                "tabfm":       (Path("tables-tabfm/parquet"),  2),
            })
    max_rows : int, default=1000
        Passed to :func:`load_dataframes`.
    """

    def __init__(
        self,
        sources: dict[str, tuple[Union[str, Path], int]],
        max_rows: int = 1000,
    ) -> None:
        self._dfs: list[pd.DataFrame] = []
        self._labels: list[int] = []
        self._names: list[str] = []

        for source_name, (parquet_dir, label) in sources.items():
            pairs = load_dataframes(parquet_dir, label, max_rows=max_rows)
            for df, lbl in pairs:
                self._dfs.append(df)
                self._labels.append(lbl)
                self._names.append(source_name)

            print(f"  {source_name}: {len(pairs)} tables (label={label})")

    @property
    def tables(self) -> list[pd.DataFrame]:
        return list(self._dfs)

    @property
    def labels(self) -> np.ndarray:
        return np.array(self._labels)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def __len__(self) -> int:
        return len(self._dfs)

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], np.ndarray, np.ndarray]:
        """Stratified train/test split.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        indices = np.arange(len(self))
        y = self.labels

        train_idx, test_idx = _sklearn_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        X_train = [self._dfs[i] for i in train_idx]
        X_test  = [self._dfs[i] for i in test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        return X_train, X_test, y_train, y_test
