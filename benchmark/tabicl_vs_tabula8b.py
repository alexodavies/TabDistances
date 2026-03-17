#!/usr/bin/env python3
"""
Benchmark TabICL (via TALENT) vs tabula-8b on classification datasets.

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
      --seeds 3 --max_context 32

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
import torch

warnings.filterwarnings("ignore")

# ── Make TALENT and rtfm importable ──────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "TALENT-main"))
# rtfm is used as a source checkout (pip install may not expose the module)
_RTFM_SRC = _ROOT / "rtfm-src"
if _RTFM_SRC.exists():
    sys.path.insert(0, str(_RTFM_SRC))

from TALENT.model.utils import get_method, set_seeds
from TALENT.model.lib.data import get_dataset


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


def run_tabicl(dataset_name, train_val_data, test_data, info, n_seeds):
    """Run TabICL for n_seeds seeds; return list of per-seed metric dicts."""
    results = []
    for seed in range(n_seeds):
        set_seeds(seed)
        args = _make_tabicl_args(seed, dataset_name)
        method = get_method("tabicl")(args, is_regression=False)

        t0 = time.time()
        method.fit(train_val_data, info)
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
# XGBoost helpers
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
# tabula-8b helpers
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


def _build_class_map(unique_labels, info):
    """Return {raw_label -> string_name} using info['target_intro'] if available."""
    target_intro = info.get("target_intro", {})
    result = {}
    for i, lbl in enumerate(unique_labels):
        key = str(lbl)
        result[lbl] = target_intro.get(key, f"class_{i}")
    return result


def _sample_context(train_df, y_train_raw, class_map, max_context, rng):
    """Return a balanced context DataFrame with string target column."""
    unique_labels = sorted(class_map.keys())
    per_class = max(1, max_context // len(unique_labels))
    chunks = []
    for lbl in unique_labels:
        idx = np.where(y_train_raw == lbl)[0]
        n = min(per_class, len(idx))
        chosen = rng.choice(idx, size=n, replace=False)
        sub = train_df.iloc[chosen].copy()
        sub["__target__"] = class_map[lbl]
        chunks.append(sub)
    ctx = pd.concat(chunks, ignore_index=True)
    return ctx.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(
        drop=True
    )


def _build_context_str(serializer, context_df, target_col, choice_names):
    """Serialize the labeled context examples into a single reusable string."""
    from rtfm.special_tokens import EOC_TOKEN, QA_SEP_TOKEN

    prefix = f"Predict the {target_col}"
    suffix = f"What is the value of {target_col}?"
    parts = []
    for _, row in context_df.iterrows():
        label = str(row[target_col])
        features = row.drop(labels=[target_col])
        serialized = serializer(
            features, prefix_text=prefix, suffix_text=suffix, choices=choice_names
        )
        parts.append(f"{serialized} {QA_SEP_TOKEN} {label} {EOC_TOKEN}")
    return " ".join(parts)


def _infer_tabula8b_batch(model, tokenizer, serializer, context_str,
                          query_df, target_col, choice_names,
                          batch_size=8, max_new_tokens=16):
    """Batched inference over query_df rows; returns list of predicted strings."""
    from rtfm.generation_utils import parse_generated_text
    from rtfm.special_tokens import QA_SEP_TOKEN

    prefix = f"Predict the {target_col}"
    suffix = f"What is the value of {target_col}?"

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    predictions = []
    for start in range(0, len(query_df), batch_size):
        batch = query_df.iloc[start:start + batch_size]
        prompts = []
        for _, row in batch.iterrows():
            features = row.drop(labels=[target_col], errors="ignore")
            serialized = serializer(
                features, prefix_text=prefix, suffix_text=suffix, choices=choice_names
            )
            prompts.append(f"{context_str} {serialized} {QA_SEP_TOKEN}")

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=tokenizer.model_max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=False)
            try:
                pred, _ = parse_generated_text(decoded)
            except ValueError:
                pred = choice_names[0]
            predictions.append(pred.strip())

    return predictions


def run_tabula8b(
    dataset_name,
    train_val_data,
    test_data,
    info,
    max_context,
    n_seeds,
    max_test_samples,
    model,
    tokenizer,
    serializer,
    batch_size=8,
):
    """Run tabula-8b inference; return list of per-seed metric dicts."""
    import sklearn.metrics as skm

    N_tv, C_tv, y_tv = train_val_data
    N_te, C_te, y_te = test_data

    train_df = _talent_to_df(N_tv, C_tv, "train")
    test_df = _talent_to_df(N_te, C_te, "test")

    y_train_raw = np.array(y_tv["train"])
    y_test_raw = np.array(y_te["test"])

    # Optionally cap test set size (tabula-8b inference is slow)
    if max_test_samples is not None and len(test_df) > max_test_samples:
        rng0 = np.random.default_rng(0)
        sel = rng0.choice(len(test_df), size=max_test_samples, replace=False)
        test_df = test_df.iloc[sel].reset_index(drop=True)
        y_test_raw = y_test_raw[sel]

    unique_labels = np.unique(y_train_raw)
    class_map = _build_class_map(unique_labels, info)
    # Choices must be a list of strings in a fixed order
    choice_names = [class_map[lbl] for lbl in sorted(class_map)]
    inv_map = {v: k for k, v in class_map.items()}

    # Add dummy target column to test_df for serializer compatibility
    test_df = test_df.copy()
    test_df["__target__"] = choice_names[0]

    results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        ctx = _sample_context(train_df, y_train_raw, class_map, max_context, rng)
        context_str = _build_context_str(serializer, ctx, "__target__", choice_names)

        t0 = time.time()
        pred_strs = _infer_tabula8b_batch(
            model=model,
            tokenizer=tokenizer,
            serializer=serializer,
            context_str=context_str,
            query_df=test_df,
            target_col="__target__",
            choice_names=choice_names,
            batch_size=batch_size,
            max_new_tokens=16,
        )

        pred_labels = np.array([inv_map.get(p, unique_labels[0]) for p in pred_strs])

        inference_time = time.time() - t0

        accuracy = skm.accuracy_score(y_test_raw, pred_labels)
        avg_recall = skm.balanced_accuracy_score(y_test_raw, pred_labels)
        avg_precision = skm.precision_score(
            y_test_raw, pred_labels, average="macro", zero_division=0
        )
        f1 = skm.f1_score(
            y_test_raw, pred_labels, average="macro", zero_division=0
        )

        row = {
            "seed": seed,
            "fit_time": 0.0,   # pure ICL: no per-dataset fitting
            "inference_time": inference_time,
            "Accuracy": accuracy,
            "Avg_Recall": avg_recall,
            "Avg_Precision": avg_precision,
            "F1": f1,
            # LogLoss and AUC omitted: require probability scores
        }
        results.append(row)
        print(
            f"    seed {seed}: Accuracy={accuracy:.4f}, F1={f1:.4f}, "
            f"inference={inference_time:.1f}s ({len(test_df)} samples)"
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
        description="Benchmark TabICL vs tabula-8b on TALENT classification datasets"
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
        default=32,
        help="Max labeled training examples in tabula-8b context (default: 32)",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Cap test-set size for tabula-8b to limit inference time (default: no cap)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tabicl"] + _CLASSICAL_MODELS,
        choices=["tabicl", "tabula8b"] + _CLASSICAL_MODELS,
        help="Which models to benchmark (default: all)",
    )
    parser.add_argument(
        "--tabula_batch_size",
        type=int,
        default=8,
        help="Batch size for tabula-8b inference (default: 8)",
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
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "datasets": dataset_names,
                "seeds": args.seeds,
                "max_context": args.max_context,
                "max_test_samples": args.max_test_samples,
                "models": args.models,
            },
        )
        wb_table = wandb.Table(columns=[
            "dataset", "model", "seed", "metric", "value", "fit_time", "inference_time"
        ])
        wb_ranking_rows = []

    # Load tabula-8b once (expensive: ~16 GB GPU RAM, several minutes)
    tabula_model = tabula_tokenizer = tabula_serializer = None
    tabula_load_time = None
    if "tabula8b" in args.models:
        print("\nLoading tabula-8b from HuggingFace (this may take a while)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from rtfm.serialization.serializers import BasicSerializerV2
        from rtfm.configs import SerializerConfig

        t_load = time.time()
        tabula_tokenizer = AutoTokenizer.from_pretrained("mlfoundations/tabula-8b")
        tabula_model = AutoModelForCausalLM.from_pretrained(
            "mlfoundations/tabula-8b",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tabula_serializer = BasicSerializerV2(SerializerConfig())
        tabula_model.generation_config.max_length = None
        tabula_load_time = time.time() - t_load
        print(f"tabula-8b loaded in {tabula_load_time:.1f}s")

    # Prepare output CSV
    os.makedirs(Path(args.output).parent, exist_ok=True)
    fieldnames = [
        "dataset", "model", "seed", "metric", "value", "fit_time", "inference_time"
    ]
    out_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    writer.writeheader()

    # Write tabula-8b model load time as a metadata row
    if tabula_load_time is not None:
        writer.writerow({
            "dataset": "__meta__",
            "model": "tabula8b",
            "seed": -1,
            "metric": "model_load_time",
            "value": tabula_load_time,
            "fit_time": float("nan"),
            "inference_time": float("nan"),
        })
        out_file.flush()
        if wb_run is not None:
            wb_run.summary["tabula8b/model_load_time"] = tabula_load_time

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
            print("  Running TabICL...")
            try:
                res = run_tabicl(
                    dataset_name, train_val_data, test_data, info, args.seeds
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

        if len(dataset_scores) > 1:
            _compute_and_write_rankings(
                writer, out_file, dataset_name, dataset_scores, args.rank_metric, wb_run, wb_ranking_rows, wb_table
            )

        if "tabula8b" in args.models:
            print(
                f"  Running tabula-8b (context={args.max_context}, "
                f"batch={args.tabula_batch_size}, "
                f"max_test={args.max_test_samples or 'all'})..."
            )
            try:
                res = run_tabula8b(
                    dataset_name,
                    train_val_data,
                    test_data,
                    info,
                    args.max_context,
                    args.seeds,
                    args.max_test_samples,
                    tabula_model,
                    tabula_tokenizer,
                    tabula_serializer,
                    batch_size=args.tabula_batch_size,
                )
                _write_results(writer, out_file, dataset_name, "tabula8b", res, wb_table)
                if wb_run is not None:
                    _log_wandb_summary(wb_run, dataset_name, "tabula8b", res)
                acc = [r.get("Accuracy", float("nan")) for r in res]
                print(
                    f"  tabula-8b Accuracy: {np.mean(acc):.4f} ± {np.std(acc):.4f}"
                )
            except Exception as e:
                print(f"  [ERROR] tabula-8b: {e}")
            finally:
                torch.cuda.empty_cache()

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
