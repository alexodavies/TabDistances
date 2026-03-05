"""
Dataset wrapper for the T4 (Tremendous TabLib Trawl) HuggingFace dataset.

T4 (~3.1M tables) is stored as chunk zip files, each containing one Parquet
file per table.  This class streams through them without bulk-extracting to
disk: each chunk zip is downloaded once to the HF cache, then Parquet files
are read in-memory via zipfile + BytesIO.

Reference: https://huggingface.co/datasets/mlfoundations/t4-full
"""

from __future__ import annotations

import io
import zipfile
from typing import Iterator, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


class T4Dataset:
    """Streaming iterator over the T4 dataset.

    Parameters
    ----------
    split : str, default="train"
        Not used (T4 has no standard splits), kept for API symmetry.
    token : str or None, default=None
        HuggingFace access token.  Required if your account must agree to the
        TabLib licence before accessing the repo.
    """

    HF_REPO = "mlfoundations/t4-full"

    def __init__(
        self,
        split: str = "train",
        token: str | None = None,
    ) -> None:
        self.split = split
        self.token = token

    def _chunk_zip_paths(self) -> list[str]:
        """Return sorted list of chunk zip file paths in the HF repo."""
        all_files = list(list_repo_files(
            self.HF_REPO,
            repo_type="dataset",
            token=self.token,
        ))
        zips = sorted(f for f in all_files if f.endswith(".zip"))
        return zips

    def iter_tables(
        self,
        max_chunks: int | None = None,
    ) -> Iterator[Tuple[int, str, pd.DataFrame]]:
        """Yield ``(global_index, content_hash, DataFrame)`` triples.

        Downloads each chunk zip from the HF cache on first access.
        Parquet files inside the zip are read in-memory without extraction.

        Parameters
        ----------
        max_chunks : int or None
            Stop after this many chunk zips (useful for quick tests).

        Yields
        ------
        global_index : int
            Monotonically increasing counter across all chunks.
        content_hash : str
            The content hash (filename stem) used as the table identifier.
        df : pd.DataFrame
            The deserialized tabular data.
        """
        chunk_paths = self._chunk_zip_paths()
        if max_chunks is not None:
            chunk_paths = chunk_paths[:max_chunks]

        global_idx = 0

        for chunk_path in chunk_paths:
            try:
                local_zip = hf_hub_download(
                    repo_id=self.HF_REPO,
                    filename=chunk_path,
                    repo_type="dataset",
                    token=self.token,
                )
            except Exception as exc:
                print(f"[DOWNLOAD ERROR] {chunk_path}: {exc}")
                continue

            try:
                with zipfile.ZipFile(local_zip, "r") as zf:
                    parquet_names = sorted(
                        n for n in zf.namelist() if n.endswith(".parquet")
                    )
                    for pq_name in parquet_names:
                        content_hash = pq_name.rsplit("/", 1)[-1].replace(".parquet", "")
                        try:
                            data = zf.read(pq_name)
                            df = pd.read_parquet(io.BytesIO(data))
                            yield global_idx, content_hash, df
                            global_idx += 1
                        except Exception as exc:
                            print(f"[READ ERROR] {pq_name}: {exc}")
            except Exception as exc:
                print(f"[ZIP ERROR] {local_zip}: {exc}")
