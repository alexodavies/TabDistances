#!/usr/bin/env python3
"""Download and export all dataset sources used by TabDistances.

Run from the repository root:

    python fetch_data.py [--tabfm] [--t4] [--icl]
                         [--tabfm-tables N] [--t4-tables N] [--icl-tables N]

By default, all three sources are fetched.  Pass individual flags to fetch only
specific ones.

What this script does
---------------------
TabFM
  1. Downloads raw CSVs from Kaggle into ``data/tabfm/``  (requires kaggle CLI
     and ``~/.kaggle/kaggle.json``).
  2. Converts them to parquet in ``tables-tabfm/parquet/``.
  Skipped if ``tables-tabfm/parquet/`` already contains >= --tabfm-tables files.

T4
  Downloads tables from HuggingFace (``mlfoundations/t4-full``) and saves them
  as parquet in ``tables-t4/parquet/``.
  Skipped if ``tables-t4/parquet/`` already contains >= --t4-tables files.

TabICL
  Generates synthetic tables via SCMPrior and saves them as parquet in
  ``tables-tabicl/parquet/``.
  Skipped if ``tables-tabicl/parquet/`` already contains >= --icl-tables files.

"""

import argparse
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MIN_TABFM = 300      # skip TabFM export if this many parquets already exist
MIN_T4    = 1_000    # skip T4 export if this many parquets already exist
MIN_ICL   = 10_000   # skip TabICL export if this many parquets already exist

ROOT = Path(__file__).resolve().parent


def _count(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob("*.parquet"))


# ---------------------------------------------------------------------------
# TabFM
# ---------------------------------------------------------------------------

def fetch_tabfm(min_tables: int) -> None:
    parquet_dir = ROOT / "tables-tabfm" / "parquet"
    n = _count(parquet_dir)
    if n >= min_tables:
        print(f"[TabFM] {n} parquet files already present — skipping.")
        return

    print(f"[TabFM] {n} parquet files found, need {min_tables}.")

    # Step 1: download CSVs from Kaggle
    csv_dir = ROOT / "data" / "tabfm"
    if not csv_dir.exists() or not any(csv_dir.glob("*.csv")):
        print("[TabFM] Downloading CSVs from Kaggle …")
        try:
            from tabdistances.data.tabfm.downloader import TabFMDownloader
            TabFMDownloader(data_path=csv_dir, split="both").download(
                skip_existing=True, verbose=True
            )
        except Exception as exc:
            print(f"[TabFM] CSV download failed: {exc}")
            print("  Make sure kaggle CLI is configured (~/.kaggle/kaggle.json).")
            return
    else:
        print(f"[TabFM] CSVs already present at {csv_dir}")

    # Step 2: convert to parquet
    print("[TabFM] Converting CSVs to parquet …")
    import numpy as np
    import pandas as pd
    from tabdistances.data.tabfm.dataset import TabFMDataset

    parquet_dir.mkdir(parents=True, exist_ok=True)
    dataset = TabFMDataset(data_path=csv_dir, split="both")
    processed = _count(parquet_dir)

    for i, name, df in dataset.iter_tables():
        try:
            if df.shape[1] < 2:
                continue
            out = parquet_dir / f"{name.replace('/', '_')}.parquet"
            if out.exists():
                continue
            for col in df.select_dtypes(include="object").columns:
                if df[col].nunique() > 20:
                    codes, _ = pd.factorize(df[col])
                    s = codes.astype(float)
                    s[s == -1] = np.nan
                    df[col] = s
            df.to_parquet(out, index=False)
            processed += 1
            print(f"  [{processed}] {name}  ({df.shape[0]}×{df.shape[1]})")
        except Exception as exc:
            print(f"  [SKIP] {name}: {exc}")

    print(f"[TabFM] Done — {_count(parquet_dir)} parquet files in {parquet_dir}")


# ---------------------------------------------------------------------------
# T4
# ---------------------------------------------------------------------------

def fetch_t4(min_tables: int) -> None:
    parquet_dir = ROOT / "tables-t4" / "parquet"
    n = _count(parquet_dir)
    if n >= min_tables:
        print(f"[T4] {n} parquet files already present — skipping.")
        return

    print(f"[T4] {n} parquet files found, need {min_tables}. Downloading …")

    from tabdistances.data.t4.dataset import T4Dataset

    parquet_dir.mkdir(parents=True, exist_ok=True)
    dataset = T4Dataset()
    processed = _count(parquet_dir)
    existing = {p.stem for p in parquet_dir.glob("*.parquet")}

    for i, content_hash, df in dataset.iter_tables():
        try:
            if df.shape[1] < 2 or random.random() >= 0.5:
                continue
            if content_hash in existing:
                continue
            out = parquet_dir / f"{content_hash}.parquet"
            df.to_parquet(out, index=False)
            existing.add(content_hash)
            processed += 1
            print(f"  [{processed}/{min_tables}] {content_hash}  ({df.shape[0]}×{df.shape[1]})")
            if processed >= min_tables:
                break
        except Exception as exc:
            print(f"  [SKIP] {content_hash}: {exc}")

    print(f"[T4] Done — {_count(parquet_dir)} parquet files in {parquet_dir}")


# ---------------------------------------------------------------------------
# TabICL
# ---------------------------------------------------------------------------

def fetch_icl(min_tables: int) -> None:
    parquet_dir = ROOT / "tables-tabicl" / "parquet"
    n = _count(parquet_dir)
    if n >= min_tables:
        print(f"[TabICL] {n} parquet files already present — skipping.")
        return

    print(f"[TabICL] {n} parquet files found, need {min_tables}. Generating …")

    import numpy as np
    import pandas as pd
    from tabdistances.data.priors.tabicl.dataset import SCMPrior

    parquet_dir.mkdir(parents=True, exist_ok=True)
    prior = SCMPrior(
        batch_size=8,
        batch_size_per_gp=4,
        min_features=2,
        max_features=100,
        max_classes=10,
        max_seq_len=1024,
        min_train_size=0.1,
        max_train_size=0.9,
        prior_type="mix_scm",
        n_jobs=6,
        device="cpu",
    )

    processed = _count(parquet_dir)
    batch_idx = 0

    while processed < min_tables:
        X, y, d, seq_lens, _ = prior.get_batch()
        for i in range(X.shape[0]):
            n_feat  = int(d[i].item())
            seq_len = int(seq_lens[i].item())
            if n_feat < 2 or seq_len == 0:
                continue
            table_id = f"tabicl_batch{batch_idx:04d}_idx{i:02d}"
            out = parquet_dir / f"{table_id}.parquet"
            if out.exists():
                continue
            try:
                X_i = X[i, :seq_len, :n_feat].numpy().astype(np.float64)
                y_i = y[i, :seq_len].numpy().astype(int)
                df  = pd.DataFrame(X_i, columns=[f"feature_{j}" for j in range(n_feat)])
                df["target"] = y_i
                df.to_parquet(out, index=False)
                processed += 1
                print(f"  [{processed}/{min_tables}] {table_id}  ({df.shape[0]}×{df.shape[1]})")
            except Exception as exc:
                print(f"  [SKIP] {table_id}: {exc}")
            if processed >= min_tables:
                break
        batch_idx += 1

    print(f"[TabICL] Done — {_count(parquet_dir)} parquet files in {parquet_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tabfm",        action="store_true", default=False)
    parser.add_argument("--t4",           action="store_true", default=False)
    parser.add_argument("--icl",          action="store_true", default=False)
    parser.add_argument("--tabfm-tables", type=int, default=MIN_TABFM,
                        help=f"Minimum TabFM parquet files before skipping (default {MIN_TABFM})")
    parser.add_argument("--t4-tables",    type=int, default=MIN_T4,
                        help=f"Minimum T4 parquet files before skipping (default {MIN_T4})")
    parser.add_argument("--icl-tables",   type=int, default=MIN_ICL,
                        help=f"Minimum TabICL parquet files before skipping (default {MIN_ICL})")
    args = parser.parse_args()

    # If no flags given, fetch all.
    fetch_all = not args.tabfm and not args.t4 and not args.icl

    if args.tabfm or fetch_all:
        fetch_tabfm(args.tabfm_tables)

    if args.t4 or fetch_all:
        fetch_t4(args.t4_tables)

    if args.icl or fetch_all:
        fetch_icl(args.icl_tables)


if __name__ == "__main__":
    main()
