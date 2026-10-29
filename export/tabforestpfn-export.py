"""Export tables from the TabForestPFN synthetic priors (mix / forest / tabpfn).

Usage
-----
    python export/tabforestpfn-export.py --component forest -n 25000
    python export/tabforestpfn-export.py --component mix    -n 25000   # needs tabpfn==0.1.9
    python export/tabforestpfn-export.py --component tabpfn -n 5000    # needs tabpfn==0.1.9, slow

"mix" is the exact prior used to pretrain TabForestPFN (50/50 forest / tabpfn
per table, see TabForestPFN/config/pretrain.yaml). "forest" and "tabpfn" run
each primitive standalone so a distributional mismatch found downstream can
be attributed to whichever component drives it.

Row count: TabForestPFN's raw generators take a single ``n_samples`` kwarg,
not a support/query split (that split only exists in the higher-level
``SyntheticDataset`` wrapper, which we bypass here). We define a table as
support + query, sampling the support size the same way pretraining does
(Uniform[min_samples_support, max_samples_support)) and adding the fixed
query size, per TabForestPFN/config/pretrain.yaml. This is sampled fresh
per table.

Column count is left at the paper's own range ([3, 100]) -- matching it to a
real corpus's n_cols distribution is done downstream (analysis/pairwise-tables.py
--match-tabforestpfn-cols), not by restricting the generator here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TabForestPFN"))

import numpy as np
import pandas as pd

# Config values from TabForestPFN/config/pretrain.yaml
MIN_FEATURES        = 3
MAX_FEATURES        = 100
MAX_CLASSES         = 10
MIN_SAMPLES_SUPPORT  = 128
MAX_SAMPLES_SUPPORT  = 1024
N_SAMPLES_QUERY      = 256
FOREST_KWARGS = dict(min_depth=1, max_depth=25, base_size=1024, categorical_x=True)


def _build_row_sampler(rng: np.random.Generator):
    # table = support + query (see module docstring)
    def sample_n_samples() -> int:
        n_support = int(rng.integers(MIN_SAMPLES_SUPPORT, MAX_SAMPLES_SUPPORT))
        return n_support + N_SAMPLES_QUERY
    return sample_n_samples


def _make_table(component: str, n_samples: int):
    if component == "forest":
        from tabularbench.data.synthetic_generator_forest import synthetic_dataset_function_forest
        return synthetic_dataset_function_forest(
            min_features=MIN_FEATURES, max_features=MAX_FEATURES,
            n_samples=n_samples, max_classes=MAX_CLASSES, **FOREST_KWARGS,
        )
    elif component == "tabpfn":
        from tabularbench.data.synthetic_generator_tabpfn import synthetic_dataset_generator_tabpfn
        gen = synthetic_dataset_generator_tabpfn(
            n_samples=n_samples, min_features=MIN_FEATURES,
            max_features=MAX_FEATURES, max_classes=MAX_CLASSES,
        )
        return next(gen)
    elif component == "mix":
        from tabularbench.data.synthetic_generator_mix import synthetic_dataset_generator_mix
        gen = synthetic_dataset_generator_mix(
            min_features=MIN_FEATURES, max_features=MAX_FEATURES,
            n_samples=n_samples, max_classes=MAX_CLASSES, **FOREST_KWARGS,
        )
        return next(gen)
    else:
        raise ValueError(f"unknown component: {component}")


def main(component: str, n_tables: int) -> None:
    out_dir     = Path(f"tables-tabforestpfn-{component}")
    parquet_dir = out_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    rng          = np.random.default_rng(0)
    sample_n     = _build_row_sampler(rng)

    processed = len(list(parquet_dir.glob("*.parquet")))
    if processed:
        print(f"Resuming: {processed} tables already in {parquet_dir}/")

    idx = processed
    while processed < n_tables:
        table_id     = f"tabforestpfn_{component}_{idx:05d}"
        parquet_path = parquet_dir / f"{table_id}.parquet"
        idx += 1
        if parquet_path.exists():
            continue

        n_samples = sample_n()
        try:
            x, y = _make_table(component, n_samples)
        except Exception as e:
            print(f"Table {table_id} failed to generate: {e}")
            continue

        n_features = x.shape[1]
        df = pd.DataFrame(x, columns=[f"feature_{j}" for j in range(n_features)])
        df["target"] = y

        try:
            df.to_parquet(parquet_path, index=False)
            processed += 1
            print(f"[{processed}/{n_tables}] {table_id}  ({df.shape[0]}x{df.shape[1]})  -> {parquet_path}")
        except Exception as e:
            print(f"Table {table_id} failed to write: {e}")

    print(f"\nDone. {processed} tables written to {parquet_dir.resolve()}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["mix", "forest", "tabpfn"], default="forest",
                        help="Which TabForestPFN generator to sample from (default: %(default)s). "
                             "mix and tabpfn require tabpfn==0.1.9 to be installed.")
    parser.add_argument("-n", "--n-tables", type=int, default=25000,
                        help="Number of tables to generate (default: %(default)s)")
    args = parser.parse_args()
    main(args.component, args.n_tables)
