"""
Dataset wrapper for the CleanTabLib HuggingFace dataset.

Provides a streaming iterator over real-world tabular datasets from
``alexodavies/cleantablib``, yielding (index, DataFrame) pairs with
robust error recovery for malformed stream entries.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import pyarrow as pa
import pandas as pd
from datasets import load_dataset


class CleanTabLibDataset:
    """Streaming iterator over the CleanTabLib dataset.

    Wraps ``alexodavies/cleantablib`` on HuggingFace Hub, handling Arrow
    deserialization and skipping corrupt stream entries automatically.

    Parameters
    ----------
    config : str, default="stage4"
        Dataset configuration/subset name (e.g. ``"stage4"``).
    split : str, default="train"
        Dataset split to load.
    deduplicate : bool, default=True
        If True, skip examples whose ``table_id`` has already been yielded.
    """

    HF_REPO = "alexodavies/cleantablib"

    def __init__(
        self,
        config: str = "stage4",
        split: str = "train",
        deduplicate: bool = True,
    ) -> None:
        self.config = config
        self.split = split
        self.deduplicate = deduplicate

    def _raw_stream(self):
        """Load and return the raw HuggingFace streaming dataset split."""
        ds = load_dataset(
            self.HF_REPO,
            self.config,
            streaming=True,
            verification_mode="no_checks",
        )
        return ds[self.split]

    @staticmethod
    def _safe_iter(dataset) -> Iterator[Tuple[int, dict]]:
        """Iterate robustly, skipping past corrupt stream entries.

        Yields ``(global_index, example)`` pairs.
        """
        skipped = 0
        while True:
            ds_from = dataset.skip(skipped) if skipped > 0 else dataset
            pos = skipped
            try:
                for example in ds_from:
                    yield pos, example
                    pos += 1
                break
            except Exception as e:
                print(f"Iterator error at position {pos}: {e}. Skipping past it.")
                skipped = pos + 1

    def iter_tables(self) -> Iterator[Tuple[int, str, pd.DataFrame]]:
        """Yield ``(index, table_id, DataFrame)`` triples from the dataset stream.

        Skips examples that fail to deserialise. When ``deduplicate=True``
        (the default), tables with a previously seen ``table_id`` are also
        skipped.

        Yields
        ------
        index : int
            Position of the example in the raw stream.
        table_id : str
            Unique identifier for the table as provided by the dataset.
        df : pd.DataFrame
            The deserialized tabular data.
        """
        seen_ids: dict[str, int] = {}
        raw = self._raw_stream()

        for i, example in self._safe_iter(raw):
            try:
                table_id = example["table_id"]

                if self.deduplicate:
                    if table_id in seen_ids:
                        print(
                            f"[DUPLICATE] table_id={table_id!r} at i={i}, "
                            f"first seen at i={seen_ids[table_id]}"
                        )
                        continue
                    seen_ids[table_id] = i

                reader = pa.RecordBatchStreamReader(example["arrow_bytes"])
                df = reader.read_all().to_pandas()
                yield i, table_id, df

            except Exception as e:
                print(f"Example {i} failed to deserialise: {e}")
