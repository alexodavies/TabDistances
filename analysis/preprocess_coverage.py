"""Coverage (recall / precision) under TabICL preprocessing, with the paper's definitions.

For each pair (A = synthetic/candidate, B = real/reference) and arm of analysis/preprocess_sanity.py:
    recall    = fraction of B covered by A  (B in A)
    precision = fraction of A covered by B  (A in B)
via tabdistances.analysis.distances.knn_coverage (joint min-max scaling, k=5, 95th-percentile
adaptive radius), on the balanced samples used elsewhere in the sanity checks.

Features whose pooled std is below MIN_STD are dropped before scaling (as in
preprocess_rank_sanity.py): near-constant features would otherwise be stretched to [0, 1].

Alongside the point value on all tables, we report mean +- std over N_SUB random 80% subsamples
of each side, as a yardstick for arm-to-arm differences (coverage is deterministic given a sample).

Run from the repo root:  python analysis/preprocess_coverage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from preprocess_sanity import ARMS, OUT_DIR, PAIRS, SEED, build_source
from tabdistances.analysis.distances import knn_coverage
from tabdistances.config import Defaults
from tabdistances.discriminator.features import FEATURE_MODES

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODES    = ["full", "scalar_only", "hist_only"]
N_SUB    = 20
SUB_FRAC = 0.8
MIN_STD  = 1e-3
# ─────────────────────────────────────────────────────────────────────────────


def _cov(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(recall, precision) with B real, A synthetic."""
    keep = np.vstack([A, B]).std(axis=0) > MIN_STD
    return knn_coverage(B[:, keep], A[:, keep], k=Defaults.KNN_K, quantile=Defaults.COVERAGE_QUANTILE)


def main() -> None:
    srcs = {n: build_source(n) for n in {s for p in PAIRS for s in p}}
    rng = np.random.default_rng(SEED)
    jobs, meta = [], []
    for a_name, b_name in PAIRS:
        n = min(len(srcs[a_name]["raw"]), len(srcs[b_name]["raw"]))
        ia = rng.choice(len(srcs[a_name]["raw"]), n, replace=False)
        ib = rng.choice(len(srcs[b_name]["raw"]), n, replace=False)
        subs = [(rng.choice(n, int(SUB_FRAC * n), replace=False), rng.choice(n, int(SUB_FRAC * n), replace=False))
                for _ in range(N_SUB)]
        for arm in ARMS:
            for mode in MODES:
                idx = FEATURE_MODES[mode]
                A = np.nan_to_num(srcs[a_name][arm][ia][:, idx])
                B = np.nan_to_num(srcs[b_name][arm][ib][:, idx])
                for s in [None, *subs]:
                    a, b = (A, B) if s is None else (A[s[0]], B[s[1]])
                    jobs.append(delayed(_cov)(a, b))
                    meta.append((f"{a_name}_vs_{b_name}", n, arm, mode, s is None))

    out = Parallel(n_jobs=-1, verbose=0)(jobs)
    rows = {}
    for (pair, n, arm, mode, full), (rec, prec) in zip(meta, out):
        r = rows.setdefault((pair, n, arm, mode), {"sub_recall": [], "sub_precision": []})
        if full:
            r["recall"], r["precision"] = rec, prec
        else:
            r["sub_recall"].append(rec); r["sub_precision"].append(prec)
    df = pd.DataFrame([{"pair": k[0], "n_per_side": k[1], "arm": k[2], "mode": k[3],
                        "recall": v["recall"], "precision": v["precision"],
                        "recall_sub_mean": np.mean(v["sub_recall"]), "recall_sub_std": np.std(v["sub_recall"]),
                        "precision_sub_mean": np.mean(v["sub_precision"]), "precision_sub_std": np.std(v["sub_precision"])}
                       for k, v in rows.items()])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "coverage.csv", index=False)
    pd.set_option("display.width", 200)
    for mode in MODES:
        print(f"\n## {mode}   (recall = B in A, precision = A in B)")
        d = df[df["mode"] == mode].copy()
        d["recall"] = d.apply(lambda r: f"{r.recall:.3f} ±{r.recall_sub_std:.3f}", axis=1)
        d["precision"] = d.apply(lambda r: f"{r.precision:.3f} ±{r.precision_sub_std:.3f}", axis=1)
        print(d.pivot(index="pair", columns="arm", values="recall")[list(ARMS)].to_string())
        print(d.pivot(index="pair", columns="arm", values="precision")[list(ARMS)].to_string())


if __name__ == "__main__":
    main()
