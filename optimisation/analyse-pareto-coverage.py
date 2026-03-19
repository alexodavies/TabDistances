"""Plot recall vs precision from local pareto-coverage CSVs.

For each source (tabfm, t4), reads pareto_results_{source}.csv from --out-dir
and produces a scatter of recall vs precision coloured by discriminator AUC,
with the global Pareto front highlighted.

Usage
-----
    python analyse-pareto-coverage.py [--out-dir OUT_DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
OUT_DIR = Path("pareto-results")

# ------------------------------------------------------------------
# Pareto helper (maximise both objectives)
# ------------------------------------------------------------------

def _pareto_mask(recall: np.ndarray, precision: np.ndarray, auc: np.ndarray) -> np.ndarray:
    """True where a point is Pareto-optimal across all three objectives:
    maximise recall, maximise precision, minimise discriminator AUC."""
    n = len(recall)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        dominated[i] = np.any(
            (recall    >= recall[i])    &
            (precision >= precision[i]) &
            (auc       <= auc[i])       &
            ((recall > recall[i]) | (precision > precision[i]) | (auc < auc[i]))
        )
    return ~dominated


def _pareto_mask_2d(recall: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """True where a point is Pareto-optimal on recall and precision only."""
    n = len(recall)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        dominated[i] = np.any(
            (recall    >= recall[i])    &
            (precision >= precision[i]) &
            ((recall > recall[i]) | (precision > precision[i]))
        )
    return ~dominated


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def _scatter_panel(
    ax: plt.Axes,
    x: np.ndarray, y: np.ndarray, c: np.ndarray,
    pareto: np.ndarray,
    xlabel: str, ylabel: str, clabel: str,
    restarts: np.ndarray | None = None,
) -> plt.cm.ScalarMappable:
    norm = mcolors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = cm.RdYlGn_r

    ax.scatter(
        x[~pareto], y[~pareto],
        c=c[~pareto], cmap=cmap, norm=norm,
        s=8, alpha=0.3, zorder=2,
    )
    ax.scatter(
        x[pareto], y[pareto],
        c=c[pareto], cmap=cmap, norm=norm,
        s=70, zorder=5, marker="*",
        edgecolors="k", linewidths=0.4,
        label=f"Pareto ({pareto.sum()})",
    )

    pad = 0.02
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="best")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm, clabel


def plot_scatter(df: pd.DataFrame, source_name: str, out_dir: Path) -> None:
    recall    = df["recall"].values
    precision = df["precision"].values
    auc       = df["discriminator_auc"].values if "discriminator_auc" in df.columns \
                else df["auc"].values if "auc" in df.columns \
                else np.full(len(df), 0.75)
    pareto    = _pareto_mask(recall, precision, auc)
    pareto_2d = _pareto_mask_2d(recall, precision)

    # Build a unique integer restart id per (run_id, restart) pair so paths
    # from different W&B runs don't get merged together
    if "restart" in df.columns:
        if "run_id" in df.columns:
            key = df["run_id"].astype(str) + "_" + df["restart"].astype(str)
        else:
            key = df["restart"].astype(str)
        restarts = pd.factorize(key)[0]
    else:
        restarts = None

    # US letter width at comfortable margins: 10.5 x 3.8 inches
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))

    panels = [
        (recall,    precision, auc,       "Recall",    "Precision", "AUC"),
        (recall,    auc,       precision, "Recall",    "AUC",       "Precision"),
        (precision, auc,       recall,    "Precision", "AUC",       "Recall"),
    ]

    for ax, (x, y, c, xl, yl, cl) in zip(axes, panels):
        sm, clabel = _scatter_panel(ax, x, y, c, pareto, xl, yl, cl, restarts=restarts)
        cb = fig.colorbar(sm, ax=ax, shrink=0.85)
        cb.set_label(clabel, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"tabicl vs {source_name} — {len(df)} trials, {pareto.sum()} on 3-objective Pareto front",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / f"pareto_scatter_{source_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

    plot_recall_precision_large(df, source_name, out_dir, restarts, pareto, pareto_2d)
    plot_coverage_front_large(df, source_name, out_dir, pareto_2d)
    plot_coverage_front_3panel(df, source_name, out_dir, pareto_2d)
    plot_coverage_only(df, source_name, out_dir, pareto_2d)


def plot_recall_precision_large(
    df: pd.DataFrame, source_name: str, out_dir: Path,
    restarts: np.ndarray | None,
    pareto: np.ndarray,
    pareto_2d: np.ndarray = None,
) -> None:
    recall    = df["recall"].values
    precision = df["precision"].values
    auc       = df["discriminator_auc"].values if "discriminator_auc" in df.columns \
                else df["auc"].values if "auc" in df.columns \
                else np.full(len(df), 0.75)

    fig, ax = plt.subplots(figsize=(10, 9))

    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    cmap = cm.RdYlGn_r

    ax.scatter(
        recall[~pareto], precision[~pareto],
        c=auc[~pareto], cmap=cmap, norm=norm,
        s=18, alpha=0.4, zorder=2,
        label=f"All trials ({(~pareto).sum()})",
    )
    ax.scatter(
        recall[pareto], precision[pareto],
        c=auc[pareto], cmap=cmap, norm=norm,
        s=180, zorder=5, marker="*",
        edgecolors="k", linewidths=0.6,
        label=f"Pareto front ({pareto.sum()} pts)",
    )
    pf = df[pareto].sort_values("recall")
    ax.step(
        pf["recall"].values, pf["precision"].values,
        where="post", color="k", linewidth=1.8, zorder=4, alpha=0.7,
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.85)
    cb.set_label("Discriminator AUC  (0.5 = indistinguishable)", fontsize=11)

    pad = 0.02
    ax.set_xlim(recall.min() - pad,    recall.max() + pad)
    ax.set_ylim(precision.min() - pad, precision.max() + pad)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Recall  (coverage A $\\in$ B)", fontsize=13)
    ax.set_ylabel("Precision  (coverage B $\\in$ A)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_title(
        f"Recall vs Precision — tabicl vs {source_name}\n"
        f"{len(df)} trials · {pareto.sum()} on 3-objective Pareto front",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="lower left")
    fig.tight_layout()

    out_path = out_dir / f"recall_precision_large_{source_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_coverage_front_large(
    df: pd.DataFrame, source_name: str, out_dir: Path,
    pareto_2d: np.ndarray,
) -> None:
    """Large recall vs precision figure annotated with the coverage-only (2D) front."""
    recall    = df["recall"].values
    precision = df["precision"].values
    auc       = df["discriminator_auc"].values if "discriminator_auc" in df.columns \
                else df["auc"].values if "auc" in df.columns \
                else np.full(len(df), 0.75)

    fig, ax = plt.subplots(figsize=(10, 9))

    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    cmap = cm.RdYlGn_r

    ax.scatter(
        recall[~pareto_2d], precision[~pareto_2d],
        c=auc[~pareto_2d], cmap=cmap, norm=norm,
        s=18, alpha=0.4, zorder=2,
        label=f"All trials ({(~pareto_2d).sum()})",
    )
    ax.scatter(
        recall[pareto_2d], precision[pareto_2d],
        c=auc[pareto_2d], cmap=cmap, norm=norm,
        s=180, zorder=5, marker="*",
        edgecolors="k", linewidths=0.6,
        label=f"Coverage front ({pareto_2d.sum()} pts)",
    )
    pf = df[pareto_2d].sort_values("recall")
    ax.step(
        pf["recall"].values, pf["precision"].values,
        where="post", color="k", linewidth=1.8, zorder=4, alpha=0.7,
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.85)
    cb.set_label("Discriminator AUC  (0.5 = indistinguishable)", fontsize=11)

    pad = 0.02
    ax.set_xlim(recall.min() - pad,    recall.max() + pad)
    ax.set_ylim(precision.min() - pad, precision.max() + pad)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Recall  (coverage A $\\in$ B)", fontsize=13)
    ax.set_ylabel("Precision  (coverage B $\\in$ A)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_title(
        f"Recall vs Precision — tabicl vs {source_name}  [coverage-only front]\n"
        f"{len(df)} trials · {pareto_2d.sum()} on 2-objective Pareto front",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="lower left")
    fig.tight_layout()

    out_path = out_dir / f"coverage_front_large_{source_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_coverage_only(
    df: pd.DataFrame, source_name: str, out_dir: Path,
    pareto_2d: np.ndarray,
) -> None:
    """Recall vs Precision scatter with no AUC colouring — works for any dataset."""
    recall    = df["recall"].values
    precision = df["precision"].values

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.scatter(
        recall[~pareto_2d], precision[~pareto_2d],
        color="#4C72B0", s=18, alpha=0.4, zorder=2,
        label=f"All trials ({(~pareto_2d).sum()})",
    )
    ax.scatter(
        recall[pareto_2d], precision[pareto_2d],
        color="#C44E52", s=180, zorder=5, marker="*",
        edgecolors="k", linewidths=0.6,
        label=f"Coverage front ({pareto_2d.sum()} pts)",
    )
    pf = df[pareto_2d].sort_values("recall")
    ax.step(
        pf["recall"].values, pf["precision"].values,
        where="post", color="k", linewidth=1.8, zorder=4, alpha=0.7,
    )

    pad = 0.02
    ax.set_xlim(recall.min() - pad, recall.max() + pad)
    ax.set_ylim(precision.min() - pad, precision.max() + pad)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Recall  (coverage real $\\in$ synth)", fontsize=13)
    ax.set_ylabel("Precision  (coverage synth $\\in$ real)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_title(
        f"Recall vs Precision — tabicl vs {source_name}\n"
        f"{len(df)} trials · {pareto_2d.sum()} on 2-objective Pareto front",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="lower left")
    fig.tight_layout()

    out_path = out_dir / f"coverage_only_{source_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_coverage_front_3panel(
    df: pd.DataFrame, source_name: str, out_dir: Path,
    pareto_2d: np.ndarray,
) -> None:
    """3-panel figure using the coverage-only (2D) Pareto front."""
    recall    = df["recall"].values
    precision = df["precision"].values
    auc       = df["discriminator_auc"].values if "discriminator_auc" in df.columns \
                else df["auc"].values if "auc" in df.columns \
                else np.full(len(df), 0.75)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))

    panels = [
        (recall,    precision, auc,       "Recall",    "Precision", "AUC"),
        (recall,    auc,       precision, "Recall",    "AUC",       "Precision"),
        (precision, auc,       recall,    "Precision", "AUC",       "Recall"),
    ]

    for ax, (x, y, c, xl, yl, cl) in zip(axes, panels):
        sm, clabel = _scatter_panel(ax, x, y, c, pareto_2d, xl, yl, cl)
        cb = fig.colorbar(sm, ax=ax, shrink=0.85)
        cb.set_label(clabel, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"tabicl vs {source_name}  [coverage-only front] — "
        f"{len(df)} trials, {pareto_2d.sum()} on 2-objective Pareto front",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / f"coverage_front_3panel_{source_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ------------------------------------------------------------------
# Combined figure
# ------------------------------------------------------------------

SOURCE_LABELS = {
    "tabfm": "TabFM",
    "t4":    "T4",
}


def plot_coverage_combined(
    datasets: dict[str, tuple[pd.DataFrame, np.ndarray]],
    out_dir: Path,
) -> None:
    """2-column US-letter figure: one coverage panel per source.

    Both panels share the same data-units-per-inch scale, but each panel's
    x/y extent follows its own data range (so panels are generally rectangular).
    """
    pad = 0.02

    # Per-dataset data ranges
    panel_info = {}
    for source_name, (df, pareto_2d) in datasets.items():
        r, p = df["recall"].values, df["precision"].values
        x_span = r.max() - r.min() + 2 * pad
        y_span = p.max() - p.min() + 2 * pad
        panel_info[source_name] = dict(
            recall=r, precision=p, pareto_2d=pareto_2d,
            x_lo=r.min() - pad, x_hi=r.max() + pad,
            y_lo=p.min() - pad, y_hi=p.max() + pad,
            x_span=x_span, y_span=y_span,
        )

    # Common scale: data-units-per-inch, set so the largest span fits in max_dim inches
    max_dim   = 3.2   # inches — maximum panel dimension
    max_span  = max(v[k] for v in panel_info.values() for k in ("x_span", "y_span"))
    scale     = max_span / max_dim   # data-units per inch

    # Panel physical sizes
    for info in panel_info.values():
        info["pw"] = info["x_span"] / scale   # panel width  in inches
        info["ph"] = info["y_span"] / scale   # panel height in inches

    # Figure layout (manual axes placement — tight_layout fights aspect ratios)
    n       = len(panel_info)
    hgap    = 0.0    # panels joined
    lmargin = 0.7    # room for ylabel
    rmargin = 0.2
    tmargin = 0.45   # room for title
    bmargin = 0.55   # room for xlabel + ticks
    max_ph  = max(v["ph"] for v in panel_info.values())
    fig_w   = lmargin + sum(v["pw"] for v in panel_info.values()) + (n - 1) * hgap + rmargin
    fig_h   = tmargin + max_ph + bmargin

    fig  = plt.figure(figsize=(fig_w, fig_h))
    x_cursor = lmargin
    axes = []
    for info in panel_info.values():
        ax = fig.add_axes([
            x_cursor / fig_w,
            bmargin / fig_h,
            info["pw"] / fig_w,
            info["ph"] / fig_h,
        ])
        axes.append(ax)
        x_cursor += info["pw"] + hgap

    for i, (ax, (source_name, info)) in enumerate(zip(axes, panel_info.items())):
        recall    = info["recall"]
        precision = info["precision"]
        pareto_2d = info["pareto_2d"]
        label     = SOURCE_LABELS.get(source_name, source_name)

        ax.scatter(
            recall[~pareto_2d], precision[~pareto_2d],
            color="#4C72B0", s=8, alpha=0.35, zorder=2,
            label="All trials",
        )
        ax.scatter(
            recall[pareto_2d], precision[pareto_2d],
            color="#C44E52", s=60, zorder=5, marker="*",
            edgecolors="k", linewidths=0.4,
            label="Coverage front",
        )
        pf = datasets[source_name][0][pareto_2d].sort_values("recall")
        ax.step(
            pf["recall"].values, pf["precision"].values,
            where="post", color="k", linewidth=1.2, zorder=4, alpha=0.7,
        )

        ax.set_xlim(info["x_lo"], info["x_hi"])
        ax.set_ylim(info["y_lo"], info["y_hi"])
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Recall  Real $\\in$ Synth", fontsize=9)
        if i == 0:
            ax.set_ylabel("Precision\nSynth $\\in$ Real", fontsize=9)
        ax.tick_params(labelsize=8)
        if i > 0:
            ax.tick_params(labelleft=False)
        ax.set_title(f"TabICL vs {label}", fontsize=9)

    out_path = out_dir / "coverage_combined.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ------------------------------------------------------------------
# LaTeX table
# ------------------------------------------------------------------

def make_latex_table(
    datasets: dict[str, tuple[pd.DataFrame, np.ndarray]],
    out_dir: Path,
) -> None:
    """Summary LaTeX table: one row per source."""
    rows = []
    for source_name, (df, pareto_2d) in datasets.items():
        label  = SOURCE_LABELS.get(source_name, source_name)
        front  = df[pareto_2d]
        r, p   = front["recall"].values, front["precision"].values

        # Knee: point on front maximising recall + precision
        knee_idx = (r + p).argmax()

        rows.append({
            "Source":        label,
            "Trials":        len(df),
            "Front pts":     pareto_2d.sum(),
            "Max recall":    df["recall"].max(),
            "Max precision": df["precision"].max(),
            "Knee recall":   r[knee_idx],
            "Knee precision": p[knee_idx],
        })

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Source & Trials & Front pts & Max recall & Max precision"
        r" & Knee recall & Knee precision \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['Source']} & {row['Trials']:,} & {row['Front pts']} &"
            f" {row['Max recall']:.3f} & {row['Max precision']:.3f} &"
            f" {row['Knee recall']:.3f} & {row['Knee precision']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Summary of Pareto coverage optimisation. "
        r"\emph{Knee} denotes the front point maximising recall $+$ precision.}",
        r"\label{tab:pareto-summary}",
        r"\end{table}",
    ]

    out_path = out_dir / "pareto_summary.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved -> {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR), type=Path,
                        help="Directory for output plots and input CSVs (default: pareto-results/)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = ["tabfm", "t4"]
    datasets = {}
    for source_name in sources:
        print(f"\n=== {source_name} ===")
        csv_path = args.out_dir / f"pareto_results_{source_name}.csv"

        if not csv_path.exists():
            print(f"  CSV not found: {csv_path} — skipping.")
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={"real_by_synth": "recall", "synth_by_real": "precision",
                           "auc": "discriminator_auc"}, inplace=True)
        print(f"  Loaded {len(df)} trials from {csv_path}")

        plot_scatter(df, source_name, args.out_dir)

        pareto_2d = _pareto_mask_2d(df["recall"].values, df["precision"].values)
        datasets[source_name] = (df, pareto_2d)

    if len(datasets) > 1:
        print("\n=== combined ===")
        plot_coverage_combined(datasets, args.out_dir)

    print("\n=== latex table ===")
    make_latex_table(datasets, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
