"""Dataset wrapper for the Industrial Foundation Models pre-training corpus.

Provides an iterator over the real-world tabular datasets used to pre-train
TabFM, using metadata bundled with this package (``meta_info/``,
``database_config/``).

Because the raw CSV files must be downloaded separately (they are hosted on
Kaggle), callers must supply a ``data_path`` pointing to the directory that
contains ``<dataset-name>.csv`` files.  Use :class:`TabFMDownloader` to fetch
them.  Datasets whose CSV is absent are skipped with a printed warning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Literal, Tuple, Union

import pandas as pd

# Metadata bundled with this package.
_HERE = Path(__file__).parent
_META_INFO_DIR = _HERE / "meta_info"
_DATABASE_CONFIG_DIR = _HERE / "database_config"

_Split = Literal["cls", "reg", "both"]

_CONFIG_FILES: dict[str, str] = {
    "cls": "pretrain_cls",
    "reg": "pretrain_reg",
}


class TabFMDataset:
    """Iterator over the TabFM pre-training datasets.

    Reads dataset names from the bundled database-config files, looks up each
    dataset's schema via its bundled meta_info JSON, and loads the
    corresponding CSV from ``data_path``.  Yields ``(index, dataset_name,
    DataFrame)`` triples.

    Parameters
    ----------
    data_path : str or Path
        Directory containing ``<dataset-name>.csv`` files.  Populate it with
        :class:`TabFMDownloader`.
    split : {"cls", "reg", "both"}, default="both"
        Which pre-training split to iterate over.  ``"cls"`` uses
        ``pretrain_cls``, ``"reg"`` uses ``pretrain_reg``, and ``"both"``
        chains them in that order.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: _Split = "both",
    ) -> None:
        self.data_path = Path(data_path)
        self.split = split

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _config_files(self) -> list[Path]:
        keys = ["cls", "reg"] if self.split == "both" else [self.split]
        return [_DATABASE_CONFIG_DIR / _CONFIG_FILES[k] for k in keys]

    def _iter_dataset_names(self) -> Iterator[str]:
        seen: set[str] = set()
        for config_path in self._config_files():
            with config_path.open() as fh:
                for line in fh:
                    name = line.strip()
                    if name and name not in seen:
                        seen.add(name)
                        yield name

    @staticmethod
    def _load_meta_info(meta_path: Path) -> dict:
        with meta_path.open() as fh:
            return json.load(fh)

    @staticmethod
    def _build_dataframe(df_raw: pd.DataFrame, meta_info: dict) -> pd.DataFrame:
        """Select and cast columns according to the meta_info schema.

        Only ``num_features`` and ``cat_features`` columns are retained
        (``other_features`` such as free-text or date fields are dropped as
        they are not meaningful for graph construction).  Numerical columns are
        cast to ``float64``; categorical columns are kept as ``object`` (str).
        """
        basic = meta_info["basic_info"]
        num_cols: list[str] = basic.get("num_features", [])
        cat_cols: list[str] = basic.get("cat_features", [])

        num_cols = [c for c in num_cols if c in df_raw.columns]
        cat_cols = [c for c in cat_cols if c in df_raw.columns]

        if not num_cols and not cat_cols:
            raise ValueError("No usable columns found after filtering.")

        parts: list[pd.DataFrame] = []
        if num_cols:
            num_df = df_raw[num_cols].copy()
            for col in num_cols:
                num_df[col] = pd.to_numeric(num_df[col], errors="coerce")
            parts.append(num_df)
        if cat_cols:
            cat_df = df_raw[cat_cols].astype(str).copy()
            parts.append(cat_df)

        return pd.concat(parts, axis=1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def iter_tables(self) -> Iterator[Tuple[int, str, pd.DataFrame]]:
        """Yield ``(index, dataset_name, DataFrame)`` triples.

        Datasets whose CSV or meta_info file is missing are skipped with a
        printed warning.  Any other error during loading or parsing is also
        caught and reported.

        Yields
        ------
        index : int
            Zero-based position of the dataset in the iteration order.
        dataset_name : str
            Name of the dataset (matches the filename stem of its CSV).
        df : pd.DataFrame
            Tabular data with numerical columns cast to ``float64`` and
            categorical columns kept as ``str``.
        """
        for index, name in enumerate(self._iter_dataset_names()):
            csv_path = self.data_path / f"{name}.csv"
            meta_path = _META_INFO_DIR / f"{name}.json"

            if not csv_path.exists():
                print(f"[MISSING CSV]  {name}  ({csv_path})")
                continue
            if not meta_path.exists():
                print(f"[MISSING META] {name}  ({meta_path})")
                continue

            try:
                meta_info = self._load_meta_info(meta_path)
                df_raw = pd.read_csv(csv_path, low_memory=False)
                df = self._build_dataframe(df_raw, meta_info)
                yield index, name, df

            except Exception as exc:
                print(f"[ERROR] {name}: {exc}")
