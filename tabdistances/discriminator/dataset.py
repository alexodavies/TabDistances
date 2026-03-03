"""Graph dataset loading and splitting for the source discriminator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split as _sklearn_split

from tabdistances.columns.typing import ColumnType


def load_graphs(
    json_dir: Union[str, Path],
    label: int,
) -> list[tuple[nx.DiGraph, int]]:
    """Load graphs from a JSON export directory and assign a class label.

    Reads every ``.json`` file produced by the export scripts
    (``cleantab-export.py``, ``tabicl-export.py``, ``tabfm-export.py``) and
    reconstructs ``nx.DiGraph`` objects with the same node and edge attributes
    as :func:`tabdistances.graph.builder.build_graph` produces.

    Parameters
    ----------
    json_dir : str or Path
        Directory containing ``<name>.json`` graph files.
    label : int
        Class label to assign to every graph loaded from this directory.

    Returns
    -------
    list of (nx.DiGraph, int)
        One pair per JSON file successfully loaded.
    """
    json_dir = Path(json_dir)
    pairs: list[tuple[nx.DiGraph, int]] = []

    for path in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            G = nx.DiGraph()

            for node in data["nodes"]:
                G.add_node(
                    node["id"],
                    col_type=ColumnType(node["col_type"]),
                    type_int=np.array(node["type_int"]),
                    histogram=np.array(node["histogram"]),
                )

            for edge in data["edges"]:
                G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

            pairs.append((G, label))

        except Exception as exc:
            print(f"[LOAD ERROR] {path.name}: {exc}")

    return pairs


class GraphDataset:
    """Labelled collection of graphs from one or more source directories.

    Parameters
    ----------
    sources : dict[str, tuple[Path, int]]
        Mapping of source name to ``(json_dir, label)`` pairs.  Example::

            GraphDataset({
                "cleantablib": (Path("graphs/json"),        0),
                "tabicl":      (Path("graphs-tabicl/json"), 1),
                "tabfm":       (Path("graphs-tabfm/json"),  2),
            })
    """

    def __init__(self, sources: dict[str, tuple[Union[str, Path], int]]) -> None:
        self._graphs: list[nx.DiGraph] = []
        self._labels: list[int] = []
        self._names: list[str] = []

        for source_name, (json_dir, label) in sources.items():
            pairs = load_graphs(json_dir, label)
            for g, lbl in pairs:
                self._graphs.append(g)
                self._labels.append(lbl)
                self._names.append(source_name)

            print(f"  {source_name}: {len(pairs)} graphs (label={label})")

    @property
    def graphs(self) -> list[nx.DiGraph]:
        return list(self._graphs)

    @property
    def labels(self) -> np.ndarray:
        return np.array(self._labels)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def __len__(self) -> int:
        return len(self._graphs)

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[list[nx.DiGraph], list[nx.DiGraph], np.ndarray, np.ndarray]:
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

        X_train = [self._graphs[i] for i in train_idx]
        X_test  = [self._graphs[i] for i in test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        return X_train, X_test, y_train, y_test
