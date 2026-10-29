"""RQ4 sanity check: does the rank-vs-distance result change under TabICL preprocessing?

Same protocol as benchmark/rank_vs_coverage_tables.py (mean k-NN distance from each TALENT
dataset to the TabICL prior in 70-D feature space, correlated with TabICL's rank), but the
features of both the TALENT datasets and the prior tables are computed after each arm of
analysis/preprocess_sanity.py (raw / tabicl_none / tabicl_power).

Run from the repo root:  python analysis/preprocess_rank_sanity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm

from preprocess_sanity import ARMS, MAX_ROWS, build_source, featurise_df
from tabdistances.analysis.distances import compute_nn_distances
from tabdistances.config import Defaults
from tabdistances.discriminator.features import FEATURE_MODES

# ── CONFIG ───────────────────────────────────────────────────────────────────
TALENT_DIR   = Path("data")
RANKINGS_CSV = Path("results/tabicl-rankings-full-context.csv")
OUT_DIR      = Path("results/preprocess-sanity")
CACHE        = Path("cache/pp_rank_sanity.npz")
REFERENCES   = ["icl", "tabforestpfn"]   # prior pools to measure distance to
K            = Defaults.KNN_K
MIN_REF_STD  = 1e-3   # drop reference-constant features: MinMax would blow tiny ranges up
# ─────────────────────────────────────────────────────────────────────────────


def _talent_df(ds_path: Path) -> pd.DataFrame | None:
    parts = []
    for split in ("train", "val", "test"):
        npy = ds_path / f"N_{split}.npy"
        if npy.exists():
            arr = np.load(npy, allow_pickle=True).astype(float)
            parts.append(pd.DataFrame(arr.reshape(len(arr), -1)))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df.columns = [f"num_{i}" for i in range(df.shape[1])]
    return df.iloc[:MAX_ROWS] if df.shape[0] and df.shape[1] else None


def _featurise_talent(ds: str) -> dict[str, np.ndarray] | None:
    df = _talent_df(TALENT_DIR / ds)
    return None if df is None else featurise_df(df)


def build(datasets: list[str]):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["names"].tolist(), {a: d[f"talent_{a}"] for a in ARMS}

    t = Parallel(n_jobs=-1)(delayed(_featurise_talent)(ds) for ds in tqdm(datasets, desc="TALENT"))
    names = [ds for ds, r in zip(datasets, t) if r is not None]
    talent = {a: np.array([r[a] for r in t if r is not None]) for a in ARMS}
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, names=np.array(names), **{f"talent_{a}": v for a, v in talent.items()})
    return names, talent


def main() -> None:
    rk = pd.read_csv(RANKINGS_CSV)
    mean_auc = rk.groupby("dataset")["AUC"].mean().rename("mean_AUC")
    tab = rk[rk["model"] == "tabicl"][["dataset", "rank", "AUC"]].merge(mean_auc, on="dataset")
    tab["relative_auc"] = tab["AUC"] / tab["mean_AUC"]
    tab = tab.drop_duplicates("dataset").set_index("dataset")

    names, talent = build(tab.index.tolist())
    refs = {r: build_source(r) for r in REFERENCES}
    y = tab.loc[names]

    rows = []
    for ref, arm in [(r, a) for r in REFERENCES for a in ARMS]:
        # ref-fitted MinMax scaling, via the package distance function
        for mode, idx in FEATURE_MODES.items():
            q = np.nan_to_num(talent[arm][:, idx]); r = np.nan_to_num(refs[ref][arm][:, idx])
            keep = r.std(axis=0) > MIN_REF_STD
            q, r = q[:, keep], r[:, keep]
            nn = compute_nn_distances(q, r, k=K, scale=True)
            for target in ("rank", "relative_auc"):
                rho, p = stats.spearmanr(nn, y[target])
                rows.append({"reference": ref, "n_ref": len(r), "arm": arm, "mode": mode, "target": target, "n": len(nn),
                             "spearman": rho, "p": p, "pearson": stats.pearsonr(nn, y[target])[0],
                             "n_feats": int(keep.sum()), "nn_mean": nn.mean(), "nn_std": nn.std()})
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "rank_vs_distance.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
