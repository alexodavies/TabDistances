"""Sanity check: does the real-vs-synthetic discriminator gap survive TabICL preprocessing?

Compares each synthetic source against a real source, for the pairs in PAIRS, under
three arms, using the same sampled tables in each:

    raw          df_to_vec on the raw tables (existing behaviour)
    tabicl_none  PreprocessingPipeline("none")  — standardise + 4-sigma outlier clip
    tabicl_power PreprocessingPipeline("power") — standardise + Yeo-Johnson + clip
    quantile     PreprocessingPipeline("quantile") — stress arm, not a TabICL default

Only columns inferred NUMERIC (on the raw table) are transformed. n_cols, type_ratio and
missing_rate come from the raw table, and column types are carried over from it, so the
preprocessing cannot change them (imputation would otherwise zero missing_rate).

Metric: 5-fold CV ROC-AUC of a random forest, per feature mode (full / scalar_only / hist_only).

Run from the repo root:  python analysis/preprocess_sanity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from tqdm import tqdm

from tabdistances.columns.typing import ColumnType, infer_col_types
from tabdistances.config import Defaults
from tabdistances.discriminator import features as F
from tabdistances.model.preprocessing import PreprocessingPipeline

# ── CONFIG ───────────────────────────────────────────────────────────────────
SOURCES    = {   # name -> (parquet dir, max tables to featurise)
    "fm":           (Path("tables-tabfm/parquet"), 324),
    "t4":           (Path("tables-t4/parquet"), 5000),
    "icl":          (Path("tables-tabicl/parquet"), 1000),
    "tabforestpfn": (Path("tables-tabforestpfn-forest/parquet"), 5000),
}
FM_DIR  = SOURCES["fm"][0]
ICL_DIR = SOURCES["icl"][0]
PAIRS      = [("icl", "fm"), ("t4", "fm"), ("tabforestpfn", "fm"), ("icl", "t4"), ("tabforestpfn", "t4")]
# (synthetic/candidate, real/reference); label 1 = first element. Balanced to the smaller side.
MAX_ROWS   = Defaults.MAX_ROWS
SEED       = Defaults.RANDOM_STATE
ARMS       = {"raw": None, "tabicl_none": "none", "tabicl_power": "power", "quantile": "quantile"}
N_FOLDS    = 5
N_REPEATS  = 3
OUT_DIR    = Path("results/preprocess-sanity")
CACHE_DIR  = Path("cache")
# ─────────────────────────────────────────────────────────────────────────────


def _load(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path).select_dtypes(include="number").iloc[:MAX_ROWS]
    except Exception:
        return None
    return None if df.shape[0] == 0 or df.shape[1] == 0 else df


def _vec_with_types(df: pd.DataFrame, col_types: dict) -> np.ndarray:
    """df_to_vec, but with column types fixed to those inferred on the raw table."""
    original = F.infer_col_types
    F.infer_col_types = lambda _df: col_types
    try:
        return F.df_to_vec(df)
    finally:
        F.infer_col_types = original


def _preprocess(df: pd.DataFrame, col_types: dict, method: str) -> pd.DataFrame:
    num_cols = [c for c, t in col_types.items() if t == ColumnType.NUMERIC]
    if not num_cols:
        return df
    X = df[num_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    X = SimpleImputer(keep_empty_features=True).fit_transform(X)
    Xt = PreprocessingPipeline(method, random_state=SEED).fit_transform(X)
    out = df.copy().astype(float)
    out[num_cols] = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _featurise(path: Path) -> dict[str, np.ndarray] | None:
    df = _load(path)
    return None if df is None else featurise_df(df)


def featurise_df(df: pd.DataFrame) -> dict[str, np.ndarray] | None:
    """Per-arm 70-D vectors for one (numeric, row-capped) table; None on failure."""
    try:
        col_types = infer_col_types(df)
        raw = _vec_with_types(df, col_types)
        vecs = {}
        for arm, method in ARMS.items():
            if method is None:
                vecs[arm] = raw
                continue
            v = _vec_with_types(_preprocess(df, col_types, method), col_types)
            v[:3] = raw[:3]   # n_cols, type_ratio, missing_rate from the raw table
            vecs[arm] = v
        return vecs
    except Exception:
        return None


def build_source(name: str) -> dict[str, np.ndarray]:
    """Per-arm feature arrays for one source (cached)."""
    cache = CACHE_DIR / f"pp_sanity_src_{name}.npz"
    if cache.exists():
        d = np.load(cache)
        return {a: d[a] for a in ARMS}

    directory, n_max = SOURCES[name]
    paths = np.array(sorted(directory.glob("*.parquet")))
    np.random.default_rng(SEED).shuffle(paths)
    results = Parallel(n_jobs=-1)(delayed(_featurise)(p) for p in tqdm(paths[: int(n_max * 1.2)], desc=name))
    kept = [r for r in results if r is not None][:n_max]
    print(f"  {name}: kept {len(kept)} tables")
    arrays = {a: np.nan_to_num(np.array([r[a] for r in kept]), nan=0.0, posinf=0.0, neginf=0.0) for a in ARMS}
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache, **arrays)
    return arrays


#: Extra single-group feature sets, to attribute the gap (indices into the 70-D vector).
GROUPS = {
    "moments":      [3, 4, 5, 6],
    "corr":         [7, 8, 9],
    "type_missing": [1, 2],
}


def main() -> None:
    srcs = {n: build_source(n) for n in SOURCES}
    names = F.FEATURE_NAMES
    modes = {**F.FEATURE_MODES, **GROUPS}
    rng = np.random.default_rng(SEED)
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    rows, diag = [], []
    for a_name, b_name in PAIRS:
        n = min(len(srcs[a_name]["raw"]), len(srcs[b_name]["raw"]))
        ia = rng.choice(len(srcs[a_name]["raw"]), n, replace=False)
        ib = rng.choice(len(srcs[b_name]["raw"]), n, replace=False)
        y = np.r_[np.ones(n), np.zeros(n)]
        for arm in ARMS:
            X = np.vstack([srcs[a_name][arm][ia], srcs[b_name][arm][ib]])
            for mode, idx in modes.items():
                clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
                auc = cross_val_score(clf, X[:, idx], y, cv=cv, scoring="roc_auc")
                rows.append({"pair": f"{a_name}_vs_{b_name}", "n_per_side": n, "arm": arm, "mode": mode,
                             "auc_mean": auc.mean(), "auc_std": auc.std()})
                print(f"{a_name}-{b_name:3s} n={n:5d} {arm:13s} {mode:12s} AUC = {auc.mean():.3f} ± {auc.std():.3f}")
            for i in range(1, F._N_SCALAR):
                diag.append({"pair": f"{a_name}_vs_{b_name}", "arm": arm, "feature": names[i],
                             "a_mean": X[y == 1, i].mean(), "b_mean": X[y == 0, i].mean()})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "auc.csv", index=False)
    pd.DataFrame(diag).to_csv(OUT_DIR / "scalar_feature_means.csv", index=False)


if __name__ == "__main__":
    main()
