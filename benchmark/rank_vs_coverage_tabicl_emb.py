"""Rank-vs-coverage using pre-trained TabICL intermediate representations.

Embeds each dataset through the pre-trained TabICL foundation model's first two
stages (ColEmbedding → RowInteraction → mean-pool) to produce a 512-dim table
vector, then measures k-NN distance from each TALENT dataset to the TabICL
ICL prior pool in that representation space.

Usage
-----
    python benchmark/rank_vs_coverage_tabicl_emb.py [--output-dir results/rank-coverage]
                                                     [--n-bins 30]
                                                     [--n-icl 25000]
                                                     [--device cpu]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TALENT-main"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from tabdistances.model.preprocessing import TransformToNumerical

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
_ROOT  = Path(__file__).resolve().parent.parent
CACHE  = _ROOT / "cache"

RANKINGS_CSV = _ROOT / "results/tabicl-rankings-full-context.csv"
TALENT_DIR   = _ROOT / "data"
ICL_DIR      = _ROOT / "tables-tabicl/parquet"

TALENT_TABICL_EMB_CACHE = CACHE / "talent_tabicl_emb_512.npz"
ICL_TABICL_EMB_CACHE    = CACHE / "icl_tabicl_emb_25000_512.npz"

MAX_ROWS     = 512
KNN_K        = 5
N_ICL        = 25_000
RANDOM_STATE = 42
EMB_DIM      = 512   # row_num_cls(4) * embed_dim(128)

_PAPER_RC = {
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.5,
    "figure.dpi":      300,
}

_BLUE   = "#2980b9"
_PURPLE = "#8e44ad"
_RED    = "#ef0e0e"


# ------------------------------------------------------------------
# Encoder loading
# ------------------------------------------------------------------

def _load_encoder(device: str):
    """Load the pre-trained TabICL col_embedder + row_interactor (frozen)."""
    from TALENT.model.lib.tabicl.classifier import TabICLClassifier

    clf = TabICLClassifier(device=device)
    clf._load_model()
    model = clf.model_.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.col_embedder, model.row_interactor


# ------------------------------------------------------------------
# DataFrame → embedding
# ------------------------------------------------------------------

def _df_to_embedding(
    df: pd.DataFrame,
    col_embedder,
    row_interactor,
    device: str,
    max_rows: int = MAX_ROWS,
) -> np.ndarray:
    """Preprocess a DataFrame and return a (EMB_DIM,) numpy embedding.

    Preprocessing mirrors SmallTabICLDiscriminator._df_to_tensor():
      - cap rows
      - inf → NaN
      - TransformToNumerical
      - nan/inf → 0
      - per-column standardize + clip to [-5, 5]
    """
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    df = df.replace([np.inf, -np.inf], np.nan)

    tfm = TransformToNumerical()
    X_np = tfm.fit_transform(df).astype(np.float32)

    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

    col_mean = X_np.mean(axis=0, keepdims=True)
    col_std  = X_np.std(axis=0, keepdims=True).clip(min=1e-6)
    X_np = ((X_np - col_mean) / col_std).clip(-5.0, 5.0)

    X = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, H)

    with torch.no_grad():
        emb     = col_embedder(X)       # (1, T, H+C, E)
        row_rep = row_interactor(emb)   # (1, T, EMB_DIM)
        vec     = row_rep.mean(dim=1).squeeze(0)  # (EMB_DIM,)

    return vec.cpu().numpy()


# ------------------------------------------------------------------
# Shared distance utilities (mirrors combined_talent.py)
# ------------------------------------------------------------------

def _nn_distances(query: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    """Mean k-NN distance from each query point to the reference pool."""
    query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
    ref   = np.nan_to_num(ref,   nan=0.0, posinf=0.0, neginf=0.0)
    scaler = MinMaxScaler().fit(ref)
    q = scaler.transform(query)
    r = scaler.transform(ref)
    tree = BallTree(r, metric="euclidean")
    dists, _ = tree.query(q, k=min(k, len(r)))
    return dists.mean(axis=1)


def _trim_distances(
    nn_dists: np.ndarray, ranks: np.ndarray, nn_self: np.ndarray,
    lo_pct: float = 0.0, hi_pct: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = np.percentile(nn_dists, [lo_pct, hi_pct])
    mask      = (nn_dists >= lo) & (nn_dists <= hi)
    self_mask = (nn_self  >= lo) & (nn_self  <= hi)
    return nn_dists[mask], ranks[mask], nn_self[self_mask]


def _bin_metric(
    nn_dists: np.ndarray, ranks: np.ndarray, n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins        = np.linspace(nn_dists.min(), nn_dists.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    idx     = np.clip(np.digitize(nn_dists, bins) - 1, 0, n_bins - 1)
    cts     = np.array([(idx == i).sum() for i in range(n_bins)])
    density = cts.astype(float)
    density /= density.max() if density.max() > 0 else 1.0
    means   = np.array([ranks[idx == i].mean() if cts[i] > 0 else np.nan for i in range(n_bins)])
    stds    = np.array([ranks[idx == i].std()  if cts[i] > 1 else 0.0   for i in range(n_bins)])
    return bins, bin_centers, density, means, stds


def _stairs(ax, density, bins, color, label, alpha_fill=0.15, lw=1.5):
    patch = ax.stairs(density, bins, fill=True, label=label)
    patch.set_facecolor((*mpl.colors.to_rgb(color), alpha_fill))
    patch.set_edgecolor(color)
    patch.set_linewidth(lw)


# ------------------------------------------------------------------
# Rankings
# ------------------------------------------------------------------

def _load_rankings() -> tuple[dict, dict, dict, dict, dict, list[str]]:
    df = pd.read_csv(RANKINGS_CSV)
    df["AUC"] = df["AUC"].astype(float)

    tc = df[df["model"] == "tabicl"][["dataset", "rank", "AUC"]].copy()
    tc["rank"] = tc["rank"].astype(int)

    df_no_tc = df[df["model"] != "tabicl"]
    mean_auc = df_no_tc.groupby("dataset")["AUC"].mean()
    best_auc = df_no_tc.groupby("dataset")["AUC"].max()

    rank_map       = dict(zip(tc["dataset"], tc["rank"]))
    tabicl_auc_map = dict(zip(tc["dataset"], tc["AUC"]))
    mean_auc_map   = mean_auc.to_dict()

    rel_auc_map = {
        ds: auc / mean_auc[ds]
        for ds, auc in zip(tc["dataset"], tc["AUC"])
        if ds in mean_auc and mean_auc[ds] > 0
    }
    rel_best_auc_map = {
        ds: auc / best_auc[ds]
        for ds, auc in zip(tc["dataset"], tc["AUC"])
        if ds in best_auc and best_auc[ds] > 0
    }

    print(f"Rankings: {len(rank_map)} datasets")
    return rank_map, rel_auc_map, rel_best_auc_map, tabicl_auc_map, mean_auc_map, tc["dataset"].tolist()


# ------------------------------------------------------------------
# Embedding panels
# ------------------------------------------------------------------

def _compute_talent_embeddings(
    datasets: list[str],
    col_embedder,
    row_interactor,
    device: str,
) -> dict[str, np.ndarray]:
    """Compute or load cached TALENT dataset embeddings."""
    cached: dict[str, np.ndarray] = {}
    if TALENT_TABICL_EMB_CACHE.exists():
        data   = np.load(TALENT_TABICL_EMB_CACHE, allow_pickle=True)
        cached = {n: data["feats"][i] for i, n in enumerate(data["names"].tolist())
                  if n in set(datasets)}
        print(f"  TALENT embs: cache hit {len(cached)}/{len(datasets)} datasets")

    missing = [ds for ds in datasets if ds not in cached]
    if missing:
        print(f"  TALENT embs: computing {len(missing)} missing datasets …")
        skipped = 0
        for ds in tqdm(missing, desc="Embedding TALENT datasets"):
            ds_path = TALENT_DIR / ds
            if not ds_path.exists():
                skipped += 1
                continue
            parts = []
            for split in ("train", "val", "test"):
                npy = ds_path / f"N_{split}.npy"
                if npy.exists():
                    arr = np.load(npy, allow_pickle=True).astype(float)
                    parts.append(pd.DataFrame(
                        arr.reshape(-1, 1) if arr.ndim == 1 else arr,
                        columns=[f"num_{i}" for i in range(arr.shape[-1])],
                    ))
            if not parts:
                skipped += 1
                continue
            df = pd.concat(parts, ignore_index=True)
            if df.shape[0] == 0 or df.shape[1] == 0:
                skipped += 1
                continue
            try:
                cached[ds] = _df_to_embedding(df, col_embedder, row_interactor, device)
            except Exception as e:
                print(f"    [WARN] {ds}: {e}")
                skipped += 1
        print(f"  TALENT embs: computed {len(missing) - skipped} ({skipped} skipped)")
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(TALENT_TABICL_EMB_CACHE,
                 names=np.array(list(cached.keys())),
                 feats=np.array(list(cached.values())))
        print(f"  TALENT embs: cache saved -> {TALENT_TABICL_EMB_CACHE}")

    return cached


def _compute_icl_embeddings(
    col_embedder,
    row_interactor,
    device: str,
    n_icl: int,
) -> np.ndarray:
    """Compute or load cached ICL prior embeddings."""
    if ICL_TABICL_EMB_CACHE.exists():
        data = np.load(ICL_TABICL_EMB_CACHE)
        print(f"  ICL embs: loaded {len(data['feats'])} vectors from cache")
        return data["feats"]

    paths = sorted(ICL_DIR.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {ICL_DIR}")

    rng   = np.random.default_rng(RANDOM_STATE)
    paths = rng.choice(paths, size=min(n_icl, len(paths)), replace=False).tolist()

    embs = []
    skipped = 0
    for p in tqdm(paths, desc="Embedding ICL prior tables"):
        try:
            df = pd.read_parquet(p)
            # Drop label column if present (last column by convention)
            if df.shape[1] > 1:
                df = df.iloc[:, :-1]
            embs.append(_df_to_embedding(df, col_embedder, row_interactor, device))
        except Exception as e:
            skipped += 1
    print(f"  ICL embs: embedded {len(embs)} tables ({skipped} skipped)")

    arr = np.array(embs)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(ICL_TABICL_EMB_CACHE, feats=arr)
    print(f"  ICL embs: cache saved -> {ICL_TABICL_EMB_CACHE}")
    return arr


def _load_panel(
    rank_map: dict,
    rel_auc_map: dict,
    rel_best_auc_map: dict,
    tabicl_auc_map: dict,
    mean_auc_map: dict,
    datasets: list[str],
    rng: np.random.Generator,
    col_embedder,
    row_interactor,
    device: str,
    n_icl: int,
):
    talent_cached = _compute_talent_embeddings(datasets, col_embedder, row_interactor, device)
    icl_all       = _compute_icl_embeddings(col_embedder, row_interactor, device, n_icl)

    names         = [ds for ds in datasets if ds in talent_cached]
    talent_embs   = np.stack([talent_cached[ds] for ds in names])
    ranks         = np.array([rank_map[n]                      for n in names], dtype=float)
    rel_aucs      = np.array([rel_auc_map.get(n, np.nan)       for n in names], dtype=float)
    rel_best_aucs = np.array([rel_best_auc_map.get(n, np.nan)  for n in names], dtype=float)
    tabicl_aucs   = np.array([tabicl_auc_map.get(n, np.nan)    for n in names], dtype=float)
    mean_aucs     = np.array([mean_auc_map.get(n, np.nan)       for n in names], dtype=float)

    idx      = rng.choice(len(icl_all), size=min(n_icl, len(icl_all)), replace=False)
    icl_embs = icl_all[idx]
    print(f"  Panel: {len(talent_embs)} TALENT datasets, {len(icl_embs)} ICL tables")

    nn_dists = _nn_distances(talent_embs, icl_embs, KNN_K)
    half     = len(icl_embs) // 2
    nn_self  = _nn_distances(icl_embs[:half], icl_embs[half:], KNN_K)
    return nn_dists, nn_self, ranks, rel_aucs, rel_best_aucs, tabicl_aucs, mean_aucs


# ------------------------------------------------------------------
# Figure (single-panel version of plot_combined)
# ------------------------------------------------------------------

def plot_single(
    nn_dists: np.ndarray,
    nn_self: np.ndarray,
    ranks: np.ndarray,
    output_path: Path,
    n_bins: int = 30,
    title: str = "TabICL embeddings",
    metric_label: str = "Mean rank",
    y_label: str = "TabICL mean rank",
    invert_y: bool = True,
) -> None:
    nd, rk, ns = _trim_distances(nn_dists, ranks, nn_self)
    bins, bin_centers, talent_density, means, stds = _bin_metric(nd, rk, n_bins)

    si_idx     = np.clip(np.digitize(ns, bins) - 1, 0, n_bins - 1)
    si_cts     = np.array([(si_idx == i).sum() for i in range(n_bins)])
    si_density = si_cts / (si_cts.max() if si_cts.max() > 0 else 1.0)

    with mpl.rc_context(_PAPER_RC):
        fig, ax1 = plt.subplots(1, 1, figsize=(3.0, 2.6))
        fig.subplots_adjust(left=0.13, right=0.87, bottom=0.26, top=0.91)

        _stairs(ax1, talent_density, bins, _BLUE,   "TALENT datasets")
        _stairs(ax1, si_density,     bins, _PURPLE, "TabICL Prior (self)")
        ax1.set_xlim(bins[0], bins[-1])
        ax1.set_xlabel("Mean k-NN distance (normalised)")
        ax1.set_ylabel("Relative density")
        ax1.set_title(title)

        ax2 = ax1.twinx()
        mask = ~np.isnan(means)
        lo = np.clip(means - stds, 1, None)
        hi = means + stds
        ax2.plot(bin_centers[mask], means[mask],
                 color=_RED, lw=1.5, marker="o", ms=3, label=metric_label)
        ax2.fill_between(bin_centers[mask], lo[mask], hi[mask],
                         color=_RED, alpha=0.2)
        if invert_y:
            ax2.invert_yaxis()
        ax2.set_ylabel(y_label, color=_RED)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=invert_y, nbins=5, prune="upper"))
        ax2.tick_params(axis="y", labelcolor=_RED)

        h_left,  l_left  = ax1.get_legend_handles_labels()
        h_right, l_right = ax2.get_legend_handles_labels()
        fig.legend(
            h_left + h_right,
            l_left + l_right,
            loc="lower center",
            ncols=3,
            bbox_to_anchor=(0.5, 0.01),
            frameon=True,
            borderaxespad=0.5,
        )

        fig.canvas.draw()
        _bbox = dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=1.0)
        _trans_yd = mpl.transforms.blended_transform_factory
        trans = _trans_yd(ax2.transAxes, ax2.transData)
        ylim  = sorted(ax2.get_ylim())
        for loc in ax2.yaxis.get_majorticklocs():
            if ylim[0] <= loc <= ylim[1]:
                label_txt = str(int(round(loc))) if invert_y else f"{loc:.2f}"
                ax2.text(0.97, loc, label_txt,
                         transform=trans, ha="right", va="center",
                         fontsize=7, color=_RED, zorder=10, bbox=_bbox)

        if mask.sum() >= 2:
            r = np.corrcoef(bin_centers[mask], means[mask])[0, 1]
            ax2.text(0.5, 0.03, f"$r = {r:.2f}$",
                     transform=ax2.transAxes, va="bottom", ha="center",
                     fontsize=7, color="black", zorder=10, bbox=_bbox)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {output_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=_ROOT / "results/rank-coverage")
    parser.add_argument("--n-bins",  type=int, default=30)
    parser.add_argument("--n-icl",   type=int, default=N_ICL)
    parser.add_argument("--device",  type=str, default="cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_STATE)

    print("=== Loading rankings ===")
    rank_map, rel_auc_map, rel_best_auc_map, tabicl_auc_map, mean_auc_map, datasets = _load_rankings()

    print("\n=== Loading pre-trained TabICL encoder ===")
    col_embedder, row_interactor = _load_encoder(args.device)

    print("\n=== Computing embeddings and distances ===")
    nn_dists, nn_self, ranks, rel_aucs, rel_best_aucs, tabicl_aucs, mean_aucs = _load_panel(
        rank_map, rel_auc_map, rel_best_auc_map, tabicl_auc_map, mean_auc_map,
        datasets, rng, col_embedder, row_interactor, args.device, args.n_icl,
    )

    print("\n=== Generating figures ===")
    plot_single(
        nn_dists, nn_self, ranks,
        args.output_dir / "rank_vs_coverage_tabicl_emb.png",
        n_bins=args.n_bins,
        title="TabICL embeddings",
        metric_label="Mean rank",
        y_label="TabICL mean rank",
        invert_y=True,
    )
    plot_single(
        nn_dists, nn_self, rel_aucs,
        args.output_dir / "rel_auc_vs_coverage_tabicl_emb.png",
        n_bins=args.n_bins,
        title="TabICL embeddings",
        metric_label="Rel. AUC (vs mean)",
        y_label="TabICL AUC / mean AUC",
        invert_y=False,
    )
    plot_single(
        nn_dists, nn_self, rel_best_aucs,
        args.output_dir / "rel_best_auc_vs_coverage_tabicl_emb.png",
        n_bins=args.n_bins,
        title="TabICL embeddings",
        metric_label="Rel. AUC (vs best)",
        y_label="TabICL AUC / best AUC",
        invert_y=False,
    )


if __name__ == "__main__":
    main()
