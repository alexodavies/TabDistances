#!/usr/bin/env python3
"""
Benchmark TabICL (via TALENT) vs AutoGluon and classical models on classification datasets.

Datasets must be in TALENT format:
  <dataset_path>/<dataset_name>/N_train.npy, C_train.npy, y_train.npy, info.json, ...

To download the full TALENT benchmark suite, get benchmark_dataset.zip from the
Google Drive link in TALENT-main/example_datasets/readme.md.

Install extra deps first:
  pip install -r benchmark/requirements_benchmark.txt

Usage:
  # Smoke test on the included example datasets
  python benchmark/tabicl_vs_tabula8b.py \
      --dataset_path TALENT-main/example_datasets \
      --output benchmark/results.csv \
      --seeds 1 --max_context 32

  # Full run
  python benchmark/tabicl_vs_tabula8b.py \
      --dataset_path /path/to/benchmark_datasets \
      --output benchmark/results.csv \
      --seeds 3

  # One model only
  python benchmark/tabicl_vs_tabula8b.py \
      --dataset_path TALENT-main/example_datasets \
      --models tabicl \
      --output benchmark/results_tabicl.csv
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Make TALENT importable ─────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "TALENT-main"))

from TALENT.model.utils import get_method, set_seeds
from TALENT.model.lib.data import get_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _talent_to_df(N, C, split: str) -> pd.DataFrame:
    """Convert TALENT numpy arrays for one split to a pandas DataFrame."""
    parts = {}
    if N is not None and split in N:
        arr = np.array(N[split], dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        for i in range(arr.shape[1]):
            parts[f"num_{i}"] = arr[:, i]
    if C is not None and split in C:
        arr = C[split]
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        for i in range(arr.shape[1]):
            parts[f"cat_{i}"] = arr[:, i].astype(str)
    return pd.DataFrame(parts)


# ─────────────────────────────────────────────────────────────────────────────
# TabICL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_tabicl_args(seed: int, dataset: str) -> argparse.Namespace:
    """Build a minimal args namespace accepted by TabICLMethod."""
    import importlib.resources as pkg_resources
    import TALENT

    with pkg_resources.files(TALENT).joinpath("configs/default/tabicl.json").open("r") as f:
        default_para = json.load(f)

    config = default_para["tabicl"]
    # base.py data_format references config['training']['n_bins']; not used by
    # TabICL's overridden data_format, but set it defensively.
    config.setdefault("training", {})["n_bins"] = 2

    save_dir = str(_ROOT / f"results_model/{dataset}-tabicl-seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    return argparse.Namespace(
        model_type="tabicl",
        dataset=dataset,
        dataset_path="",
        seed=seed,
        # TabICLMethod.__init__ asserts these exact values:
        normalization="none",
        cat_policy="indices",
        num_policy="none",
        tune=False,
        # General preprocessing
        num_nan_policy="mean",
        cat_nan_policy="new",
        n_bins=2,
        cat_min_frequency=0.0,
        # Training loop (unused by TabICL, but Method base class reads them)
        max_epoch=0,
        batch_size=1024,
        workers=0,
        use_float=False,
        retune=False,
        evaluate_option="best-val",
        model_path=str(_ROOT / "results_model"),
        save_path=save_dir,
        config=config,
    )


def _cap_train_split(train_val_data, max_context, seed):
    """Return a copy of train_val_data with the 'train' split capped to
    max_context examples, sampled in a balanced-per-class fashion."""
    N, C, y = train_val_data
    y_train = np.array(y["train"])
    unique_labels = np.unique(y_train)
    per_class = max(1, max_context // len(unique_labels))

    rng = np.random.default_rng(seed)
    chosen = []
    for lbl in unique_labels:
        idx = np.where(y_train == lbl)[0]
        n = min(per_class, len(idx))
        chosen.append(rng.choice(idx, size=n, replace=False))
    sel = np.concatenate(chosen)
    rng.shuffle(sel)

    def _slice(arr_dict):
        if arr_dict is None:
            return None
        out = dict(arr_dict)
        if "train" in out and out["train"] is not None:
            out["train"] = np.array(out["train"])[sel]
        return out

    return _slice(N), _slice(C), _slice(y)


def run_tabicl(dataset_name, train_val_data, test_data, info, n_seeds, max_context=None):
    """Run TabICL for n_seeds seeds; return list of per-seed metric dicts.

    If max_context is given the training split is capped to that many examples
    (balanced per class).
    """
    results = []
    for seed in range(n_seeds):
        set_seeds(seed)
        args = _make_tabicl_args(seed, dataset_name)
        method = get_method("tabicl")(args, is_regression=False)

        fit_data = (
            _cap_train_split(train_val_data, max_context, seed)
            if max_context is not None
            else train_val_data
        )

        t0 = time.time()
        method.fit(fit_data, info)
        fit_time = time.time() - t0

        t1 = time.time()
        _vl, vres, metric_name, _ = method.predict(
            test_data, info, model_name="best-val"
        )
        inference_time = time.time() - t1

        row = {"seed": seed, "fit_time": fit_time, "inference_time": inference_time}
        for name, val in zip(metric_name, vres):
            row[name] = float(val)
        results.append(row)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Classical model helpers
# ─────────────────────────────────────────────────────────────────────────────

_CLASSICAL_MODELS = ["xgboost", "catboost", "lightgbm", "RandomForest", "LogReg", "svm", "knn"]


def _make_classical_args(model_type: str, seed: int, dataset: str) -> argparse.Namespace:
    """Build a minimal args namespace accepted by any TALENT classical method."""
    import importlib.resources as pkg_resources
    import TALENT

    with pkg_resources.files(TALENT).joinpath(f"configs/default/{model_type}.json").open("r") as f:
        default_para = json.load(f)

    config = default_para[model_type]
    config.setdefault("fit", {})["n_bins"] = 2

    save_dir = str(_ROOT / f"results_model/{dataset}-{model_type}-seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    return argparse.Namespace(
        model_type=model_type,
        dataset=dataset,
        dataset_path="",
        seed=seed,
        normalization="quantile",
        num_nan_policy="mean",
        cat_nan_policy="new",
        cat_policy="ordinal",   # all classical_methods assert cat_policy != 'indices'
        num_policy="none",
        n_bins=2,
        cat_min_frequency=0.0,
        tune=False,
        retune=False,
        evaluate_option="best-val",
        model_path=str(_ROOT / "results_model"),
        save_path=save_dir,
        config=config,
    )


def run_classical(model_type, dataset_name, train_val_data, test_data, info, n_seeds):
    """Run any TALENT classical method for n_seeds seeds; return per-seed metric dicts."""
    results = []
    for seed in range(n_seeds):
        set_seeds(seed)
        args = _make_classical_args(model_type, seed, dataset_name)
        method = get_method(model_type)(args, is_regression=False)

        t0 = time.time()
        method.fit(train_val_data, info)
        fit_time = time.time() - t0

        t1 = time.time()
        vres, metric_name, _ = method.predict(test_data, info, model_name="best-val")
        inference_time = time.time() - t1

        row = {"seed": seed, "fit_time": fit_time, "inference_time": inference_time}
        for name, val in zip(metric_name, vres):
            row[name] = float(val)
        results.append(row)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# AutoGluon helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_autogluon(dataset_name, train_val_data, test_data, info, n_seeds):
    """Run AutoGluon TabularPredictor for n_seeds seeds; return per-seed metric dicts."""
    import sklearn.metrics as skm
    from autogluon.tabular import TabularPredictor

    N_tv, C_tv, y_tv = train_val_data
    N_te, C_te, y_te = test_data

    train_df = _talent_to_df(N_tv, C_tv, "train")
    test_df = _talent_to_df(N_te, C_te, "test")

    y_train = np.array(y_tv["train"])
    y_test = np.array(y_te["test"])

    train_df = train_df.copy()
    train_df["label"] = y_train

    results = []
    for seed in range(n_seeds):
        save_path = str(_ROOT / f"results_model/{dataset_name}-autogluon-seed{seed}")

        t0 = time.time()
        predictor = TabularPredictor(label="label", path=save_path, verbosity=0).fit(
            train_df, ag_args_fit={"random_state": seed}
        )
        fit_time = time.time() - t0

        t1 = time.time()
        predictions = predictor.predict(test_df).to_numpy()
        inference_time = time.time() - t1

        accuracy = skm.accuracy_score(y_test, predictions)
        avg_recall = skm.balanced_accuracy_score(y_test, predictions)
        avg_precision = skm.precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        f1 = skm.f1_score(y_test, predictions, average="macro", zero_division=0)

        row = {
            "seed": seed,
            "fit_time": fit_time,
            "inference_time": inference_time,
            "Accuracy": accuracy,
            "Avg_Recall": avg_recall,
            "Avg_Precision": avg_precision,
            "F1": f1,
        }
        results.append(row)
        print(
            f"    seed {seed}: Accuracy={accuracy:.4f}, F1={f1:.4f}, "
            f"fit={fit_time:.1f}s, inference={inference_time:.1f}s"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_KEYS = {"seed", "fit_time", "inference_time"}


def _write_results(writer, out_file, dataset, model_name, seed_results, wb_table=None):
    for r in seed_results:
        for metric, value in r.items():
            if metric in _SKIP_KEYS:
                continue
            row = {
                "dataset": dataset,
                "model": model_name,
                "seed": r["seed"],
                "metric": metric,
                "value": value,
                "fit_time": r.get("fit_time", float("nan")),
                "inference_time": r.get("inference_time", float("nan")),
            }
            writer.writerow(row)
            if wb_table is not None:
                wb_table.add_data(*[row[k] for k in wb_table.columns])
    out_file.flush()


def _log_wandb_summary(run, dataset, model_name, seed_results):
    """Log per-dataset mean metrics as wandb summary scalars."""
    import wandb

    metric_keys = [k for k in seed_results[0] if k not in _SKIP_KEYS and k != "seed"]
    for metric in metric_keys:
        vals = [r[metric] for r in seed_results if metric in r]
        run.log({
            f"{model_name}/{dataset}/{metric}": float(np.mean(vals)),
            f"{model_name}/{dataset}/{metric}_std": float(np.std(vals)),
        })


def _compute_and_write_rankings(
    writer, out_file, dataset, model_scores, rank_metric, wb_run, wb_ranking_rows, wb_table
):
    """Rank all models by mean rank_metric (desc) and write ranking rows to CSV/wandb."""
    ranked = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (model_name, score) in enumerate(ranked, start=1):
        row = {
            "dataset": dataset,
            "model": model_name,
            "seed": -1,
            "metric": "ranking",
            "value": rank,
            "fit_time": float("nan"),
            "inference_time": float("nan"),
        }
        writer.writerow(row)
        if wb_table is not None:
            wb_table.add_data(*[row[k] for k in wb_table.columns])
        if wb_ranking_rows is not None:
            wb_ranking_rows.append((dataset, model_name, rank, round(score, 4)))
    out_file.flush()

    # Log just the tabicl rank live to wandb
    if wb_run is not None and "tabicl" in model_scores:
        tabicl_rank = next(r for r, (m, _) in enumerate(ranked, 1) if m == "tabicl")
        wb_run.log({"ranking/tabicl": tabicl_rank})

    # Print the ranking table
    print(f"  Rankings on {dataset} (by {rank_metric}):")
    for rank, (model_name, score) in enumerate(ranked, start=1):
        print(f"    {rank}. {model_name:20s} {rank_metric}={score:.4f}")


def _discover_datasets(dataset_path: str):
    root = Path(dataset_path)
    return sorted(
        d.name for d in root.iterdir() if d.is_dir() and (d / "info.json").exists()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TabICL vs AutoGluon and classical models on TALENT classification datasets"
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Directory containing TALENT-format dataset subfolders",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names to run (default: all subdirs with info.json)",
    )
    parser.add_argument(
        "--output",
        default="benchmark/results.csv",
        help="Output CSV path (default: benchmark/results.csv)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds per model (default: 1)",
    )
    parser.add_argument(
        "--max_context",
        type=int,
        default=512,
        help="Max labeled training examples in TabICL context (default: 512)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tabicl", "autogluon"] + _CLASSICAL_MODELS,
        choices=["tabicl", "autogluon"] + _CLASSICAL_MODELS,
        help="Which models to benchmark (default: tabicl + autogluon + classical)",
    )
    parser.add_argument(
        "--rank_metric",
        default="F1",
        choices=["Accuracy", "F1", "AUC", "Avg_Recall", "Avg_Precision"],
        help="Metric to use for inter-model ranking (default: F1)",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        help="W&B project name to log results (omit to disable W&B)",
    )
    parser.add_argument(
        "--wandb_run_name",
        default=None,
        help="W&B run name (default: auto-generated by W&B)",
    )
    args = parser.parse_args()

    dataset_names = args.datasets or _discover_datasets(args.dataset_path)
    print(f"Datasets ({len(dataset_names)}): {dataset_names}")

    # ── W&B setup ────────────────────────────────────────────────────────────
    wb_run = None
    wb_table = None
    wb_ranking_rows = None
    if args.wandb_project:
        import wandb
        wb_run = wandb.init(
            project="ICL-Benchmarking",
            name=args.wandb_run_name,
            config={
                "datasets": dataset_names,
                "seeds": args.seeds,
                "max_context": args.max_context,
                "models": args.models,
            },
        )
        wb_table = wandb.Table(columns=[
            "dataset", "model", "seed", "metric", "value", "fit_time", "inference_time"
        ])
        wb_ranking_rows = []

    # Prepare output CSV
    os.makedirs(Path(args.output).parent, exist_ok=True)
    fieldnames = [
        "dataset", "model", "seed", "metric", "value", "fit_time", "inference_time"
    ]
    out_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    writer.writeheader()

    for dataset_name in dataset_names:
        print(f"\n{'=' * 60}\nDataset: {dataset_name}")
        try:
            train_val_data, test_data, info = get_dataset(
                dataset_name, args.dataset_path
            )
        except Exception as e:
            print(f"  [SKIP] Could not load: {e}")
            continue

        task_type = info.get("task_type", "")
        if task_type not in ("binclass", "multiclass"):
            print(f"  [SKIP] task_type={task_type!r} — only classification supported")
            continue

        n_test = len(test_data[2]["test"])
        print(f"  task={task_type}, test_size={n_test}")

        dataset_scores = {}  # model_name -> mean rank_metric score

        for classical_model in _CLASSICAL_MODELS:
            if classical_model not in args.models:
                continue
            print(f"  Running {classical_model}...")
            try:
                res = run_classical(
                    classical_model, dataset_name, train_val_data, test_data, info, args.seeds
                )
                _write_results(writer, out_file, dataset_name, classical_model, res, wb_table)
                acc = [r.get("Accuracy", float("nan")) for r in res]
                scores = [r.get(args.rank_metric, float("nan")) for r in res]
                mean_score = float(np.nanmean(scores))
                if not np.isnan(mean_score):
                    dataset_scores[classical_model] = mean_score
                print(f"  {classical_model} Accuracy: {np.mean(acc):.4f} ± {np.std(acc):.4f}")
            except Exception as e:
                print(f"  [ERROR] {classical_model}: {e}")

        if "tabicl" in args.models:
            print(f"  Running TabICL (max_context={args.max_context})...")
            try:
                res = run_tabicl(
                    dataset_name, train_val_data, test_data, info, args.seeds,
                    max_context=args.max_context,
                )
                _write_results(writer, out_file, dataset_name, "tabicl", res, wb_table)
                acc = [r.get("Accuracy", float("nan")) for r in res]
                scores = [r.get(args.rank_metric, float("nan")) for r in res]
                mean_score = float(np.nanmean(scores))
                if not np.isnan(mean_score):
                    dataset_scores["tabicl"] = mean_score
                print(f"  TabICL Accuracy: {np.mean(acc):.4f} ± {np.std(acc):.4f}")
            except Exception as e:
                print(f"  [ERROR] TabICL: {e}")

        if "autogluon" in args.models:
            print("  Running AutoGluon...")
            try:
                res = run_autogluon(
                    dataset_name, train_val_data, test_data, info, args.seeds
                )
                _write_results(writer, out_file, dataset_name, "autogluon", res, wb_table)
                if wb_run is not None:
                    _log_wandb_summary(wb_run, dataset_name, "autogluon", res)
                acc = [r.get("Accuracy", float("nan")) for r in res]
                scores = [r.get(args.rank_metric, float("nan")) for r in res]
                mean_score = float(np.nanmean(scores))
                if not np.isnan(mean_score):
                    dataset_scores["autogluon"] = mean_score
                print(f"  AutoGluon Accuracy: {np.mean(acc):.4f} ± {np.std(acc):.4f}")
            except Exception as e:
                print(f"  [ERROR] AutoGluon: {e}")

        if len(dataset_scores) > 1:
            _compute_and_write_rankings(
                writer, out_file, dataset_name, dataset_scores, args.rank_metric, wb_run, wb_ranking_rows, wb_table
            )

    out_file.close()
    print(f"\nResults saved to: {args.output}")

    if wb_run is not None:
        import wandb
        wb_run.log({"results": wb_table})
        if wb_ranking_rows:
            ranking_table = wandb.Table(
                columns=["dataset", "model", "rank", args.rank_metric],
                data=wb_ranking_rows,
            )
            wb_run.log({"ranking": ranking_table})
        wb_run.finish()
        print(f"W&B run: {wb_run.url}")


if __name__ == "__main__":
    main()
