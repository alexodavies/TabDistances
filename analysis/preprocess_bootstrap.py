"""Bootstrap-AUC robustness to preprocessing, using the paper's protocol unchanged.

Protocol (as analyse_bootstrap_auc in analysis/coverage_analysis.py): per iteration, resample both
sides with replacement, stratified 70/30 split, XGBClassifier(n_estimators=300), test ROC-AUC;
200 iterations, 99.5% percentile CI.

Each iteration's resample, split and XGBoost seed are shared across arms, so arms can be compared
with a paired difference (raw minus arm) and its own CI.

Run from the repo root:  python analysis/preprocess_bootstrap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

from preprocess_sanity import ARMS, PAIRS, SEED, GROUPS, OUT_DIR, build_source
from tabdistances.discriminator.features import FEATURE_MODES

# ── CONFIG ───────────────────────────────────────────────────────────────────
N_BOOT     = 200      # as coverage_analysis.py
BOOT_ALPHA = 0.005    # 99.5% CI
TEST_FRAC  = 0.3
MODES      = {**FEATURE_MODES, **GROUPS}
# (mode -> arms to run); attribution groups only on raw vs the TabICL-default power arm
MODE_ARMS  = {"full": list(ARMS), "hist_only": list(ARMS),
              "moments": ["raw", "tabicl_power"], "corr": ["raw", "tabicl_power"],
              "type_missing": ["raw", "tabicl_power"]}
# ─────────────────────────────────────────────────────────────────────────────


def _one_iteration(seed: int, A: dict, B: dict, idx: list[int], arms: list[str]) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_a, n_b = len(A["raw"]), len(B["raw"])
    a_idx = rng.integers(0, n_a, n_a)
    b_idx = rng.integers(0, n_b, n_b)
    y = np.array([0] * n_a + [1] * n_b)
    split_seed, xgb_seed = int(rng.integers(1e6)), int(rng.integers(1e6))
    out = {}
    for arm in arms:
        X = np.vstack([A[arm][a_idx], B[arm][b_idx]])[:, idx]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_FRAC, random_state=split_seed, stratify=y)
        clf = XGBClassifier(n_estimators=300, eval_metric="logloss", verbosity=0,
                            random_state=xgb_seed, n_jobs=1)
        clf.fit(X_tr, y_tr)
        out[arm] = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    return out


def _ci(x: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(x, [100 * BOOT_ALPHA / 2, 100 * (1 - BOOT_ALPHA / 2)]))


def main() -> None:
    srcs = {n: build_source(n) for n in {s for p in PAIRS for s in p}}
    rng = np.random.default_rng(SEED)
    seeds = np.random.SeedSequence(SEED).generate_state(N_BOOT)
    rows = []
    for a_name, b_name in PAIRS:
        n = min(len(srcs[a_name]["raw"]), len(srcs[b_name]["raw"]))
        ia = rng.choice(len(srcs[a_name]["raw"]), n, replace=False)
        ib = rng.choice(len(srcs[b_name]["raw"]), n, replace=False)
        A = {arm: srcs[a_name][arm][ia] for arm in ARMS}
        B = {arm: srcs[b_name][arm][ib] for arm in ARMS}
        for mode, arms in MODE_ARMS.items():
            res = Parallel(n_jobs=-1)(
                delayed(_one_iteration)(int(s), A, B, MODES[mode], arms)
                for s in tqdm(seeds, desc=f"{a_name}-{b_name} {mode}", leave=False))
            aucs = {arm: np.array([r[arm] for r in res]) for arm in arms}
            for arm in arms:
                lo, hi = _ci(aucs[arm])
                d = aucs["raw"] - aucs[arm]
                dlo, dhi = _ci(d)
                rows.append({"pair": f"{a_name}_vs_{b_name}", "n_per_side": n, "mode": mode, "arm": arm,
                             "auc_mean": aucs[arm].mean(), "auc_std": aucs[arm].std(), "ci_lo": lo, "ci_hi": hi,
                             "diff_vs_raw_mean": d.mean(), "diff_ci_lo": dlo, "diff_ci_hi": dhi})
                print(f"{a_name}-{b_name:3s} {mode:12s} {arm:13s} AUC {aucs[arm].mean():.4f} "
                      f"[{lo:.4f}, {hi:.4f}]  raw-diff {d.mean():+.4f} [{dlo:+.4f}, {dhi:+.4f}]", flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / "bootstrap_auc.csv", index=False)


if __name__ == "__main__":
    main()
