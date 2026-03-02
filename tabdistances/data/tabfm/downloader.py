"""Kaggle dataset downloader for the TabFM pre-training corpus.

Downloads the raw CSV files for datasets listed in the bundled database-config
files, using the Kaggle CLI.  Requires the ``kaggle`` package and a valid
``~/.kaggle/kaggle.json`` credentials file.

Usage
-----
Instantiate :class:`TabFMDownloader` and call :meth:`download`::

    from tabdistances.data.tabfm.downloader import TabFMDownloader

    dl = TabFMDownloader(data_path="data/tabfm", split="both")
    dl.download(skip_existing=True)
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Literal, Union

import pandas as pd

# Metadata bundled with this package.
_HERE = Path(__file__).parent
_META_INFO_DIR = _HERE / "meta_info"
_DATABASE_CONFIG_DIR = _HERE / "database_config"
_URL_FILE = _HERE / "datasets_url" / "all_kaggle_datasets"

_Split = Literal["cls", "reg", "both"]

_CONFIG_FILES: dict[str, str] = {
    "cls": "pretrain_cls",
    "reg": "pretrain_reg",
}


class TabFMDownloader:
    """Download raw CSV files for the TabFM pre-training datasets from Kaggle.

    Uses the bundled dataset-name-to-URL mapping to resolve each dataset's
    Kaggle handle, then downloads and extracts it via the ``kaggle`` CLI.

    When a dataset archive contains multiple CSV files, the one whose
    ``(row_count, col_count)`` best matches the bundled meta_info schema is
    selected.  If no CSV is found after extraction the dataset is skipped.

    Parameters
    ----------
    data_path : str or Path
        Directory where ``<dataset-name>.csv`` files will be written.  Created
        if it does not already exist.
    split : {"cls", "reg", "both"}, default="both"
        Which pre-training split(s) to download.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: _Split = "both",
    ) -> None:
        self.data_path = Path(data_path)
        self.split = split
        self._url_lookup: dict[str, str] = {}  # slug -> "owner/slug"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_url_lookup(self) -> dict[str, str]:
        """Parse the bundled URL file into a ``{slug: "owner/slug"}`` dict."""
        lookup: dict[str, str] = {}
        with _URL_FILE.open() as fh:
            for line in fh:
                url = line.strip()
                if not url:
                    continue
                # URL format: https://www.kaggle.com/datasets/<owner>/<slug>
                parts = url.rstrip("/").split("/")
                if len(parts) >= 2:
                    owner, slug = parts[-2], parts[-1]
                    lookup[slug] = f"{owner}/{slug}"
        return lookup

    def _iter_dataset_names(self):
        keys = ["cls", "reg"] if self.split == "both" else [self.split]
        seen: set[str] = set()
        for key in keys:
            path = _DATABASE_CONFIG_DIR / _CONFIG_FILES[key]
            with path.open() as fh:
                for line in fh:
                    name = line.strip()
                    if name and name not in seen:
                        seen.add(name)
                        yield name

    @staticmethod
    def _load_meta_info(path: Path) -> dict | None:
        try:
            with path.open() as fh:
                return json.load(fh)
        except Exception:
            return None

    @staticmethod
    def _find_csv(download_dir: Path, meta_info: dict | None) -> Path | None:
        """Return the best-matching CSV file from an extracted download."""
        csvs = list(download_dir.rglob("*.csv"))
        if not csvs:
            return None
        if len(csvs) == 1:
            return csvs[0]

        # Multiple CSVs — use meta_info row/col counts to pick the best match.
        if meta_info is not None:
            target_rows = meta_info["basic_info"].get("row_num", 0)
            target_cols = meta_info["basic_info"].get("col_num", 0)

            def _score(p: Path) -> float:
                try:
                    df = pd.read_csv(p, nrows=0)
                    ncols = len(df.columns)
                    nrows = sum(1 for _ in p.open()) - 1  # subtract header
                    row_err = abs(nrows - target_rows) / max(target_rows, 1)
                    col_err = abs(ncols - target_cols) / max(target_cols, 1)
                    return row_err + col_err
                except Exception:
                    return float("inf")

            return min(csvs, key=_score)

        # Fall back to the largest file.
        return max(csvs, key=lambda p: p.stat().st_size)

    @staticmethod
    def _kaggle_cmd() -> str:
        """Return the path to the kaggle CLI, raising clearly if absent."""
        cmd = shutil.which("kaggle")
        if cmd is None:
            raise RuntimeError(
                "The 'kaggle' CLI is required. Install it with:\n"
                "  pip install kaggle\n"
                "Then place your API token at ~/.kaggle/kaggle.json."
            )
        return cmd

    def _download_one(
        self,
        name: str,
        handle: str,
        meta_info: dict | None,
        verbose: bool,
    ) -> bool:
        """Download a single dataset via the kaggle CLI. Returns True on success."""
        import subprocess

        cmd = self._kaggle_cmd()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = subprocess.run(
                [cmd, "datasets", "download", "-d", handle, "-p", str(tmp_path), "--unzip"],
                capture_output=not verbose,
                text=True,
            )
            if result.returncode != 0:
                err = result.stderr.strip() if not verbose else ""
                print(f"  [DOWNLOAD FAILED] {handle}{': ' + err if err else ''}")
                return False

            csv_src = self._find_csv(tmp_path, meta_info)
            if csv_src is None:
                print(f"  [NO CSV FOUND] {name} (handle={handle})")
                return False

            dest = self.data_path / f"{name}.csv"
            shutil.copy2(csv_src, dest)
            if verbose:
                rows = meta_info["basic_info"].get("row_num", "?") if meta_info else "?"
                print(f"  -> {dest}  ({rows} rows expected, source: {csv_src.name})")
            return True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download(self, skip_existing: bool = True, verbose: bool = True) -> dict[str, bool]:
        """Download all datasets for the configured split.

        Parameters
        ----------
        skip_existing : bool, default=True
            If True, datasets whose CSV already exists in ``data_path`` are
            skipped without re-downloading.
        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        dict[str, bool]
            Mapping of dataset name to success flag.
        """
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._url_lookup = self._load_url_lookup()

        results: dict[str, bool] = {}
        names = list(self._iter_dataset_names())
        n = len(names)

        for i, name in enumerate(names, 1):
            dest = self.data_path / f"{name}.csv"
            prefix = f"[{i:>3}/{n}] {name}"

            if skip_existing and dest.exists():
                if verbose:
                    print(f"{prefix}  (exists, skipping)")
                results[name] = True
                continue

            handle = self._url_lookup.get(name)
            if handle is None:
                print(f"{prefix}  [NO URL] not in all_kaggle_datasets")
                results[name] = False
                continue

            meta_info = self._load_meta_info(_META_INFO_DIR / f"{name}.json")

            if verbose:
                print(f"{prefix}  downloading {handle} ...")

            results[name] = self._download_one(name, handle, meta_info, verbose)

        ok = sum(results.values())
        print(f"\nDone. {ok}/{n} datasets available in {self.data_path.resolve()}/")
        return results
