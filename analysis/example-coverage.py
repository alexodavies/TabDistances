import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# font sizes appropriate for a full-width figure on US letter (6.5" usable)
plt.rcParams.update({
    "font.size":        7,
    "axes.titlesize":   8,
    "axes.labelsize":   7,
    "legend.fontsize":  6.5,
    "figure.dpi":       150,
})

k = 5
n_real = 100
n_synth = 100

# region encoding: 0=neither, 1=rev only, 2=fwd only, 3=both
REGION_COLORS = ["#fa8383", "#5eaaf7", "#69f969", "#f2d30c"]
REGION_LABELS = ["neither", "A only", "B only", "both"]

scenarios = [
    dict(
        title="B within A",
        real_mu=[0.5, 0.5], real_std=[0.28, 0.28],
        synth_mu=[0.5, 0.5], synth_std=[0.14, 0.14],
        seed=9,
    ),
    dict(
        title="No Overlap",
        real_mu=[0.3, 0.3], real_std=[0.10, 0.10],
        synth_mu=[0.7, 0.7], synth_std=[0.07, 0.07],
        seed=7,
    ),
    dict(
        title="Partial Overlap",
        real_mu=[0.4, 0.5], real_std=[0.07, 0.07],
        synth_mu=[0.6, 0.5], synth_std=[0.07, 0.07],
        seed=12,
    ),
]

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8))

for ax, sc in zip(axes, scenarios):
    np.random.seed(sc["seed"])
    real  = np.random.normal(sc["real_mu"],  sc["real_std"],  (n_real,  2))
    synth = np.random.normal(sc["synth_mu"], sc["synth_std"], (n_synth, 2))

    # normalize jointly
    data  = np.vstack([real, synth])
    min_v = data.min(axis=0)
    max_v = data.max(axis=0)
    real  = (real  - min_v) / (max_v - min_v)
    synth = (synth - min_v) / (max_v - min_v)

    # optionally centre on zero
    if sc.get("centre_zero"):
        shift = np.vstack([real, synth]).mean(axis=0)
        real  -= shift
        synth -= shift

    # --- forward: synth covers real ---
    # threshold from synthetic internal distances
    nbrs_s     = NearestNeighbors(n_neighbors=k+1).fit(synth)
    d, _       = nbrs_s.kneighbors(synth)
    delta_fwd  = np.percentile(d[:,1:].mean(axis=1), 95)

    nbrs_cross   = NearestNeighbors(n_neighbors=k).fit(synth)
    dist_real, _ = nbrs_cross.kneighbors(real)
    mean_real    = dist_real.mean(axis=1)
    uncovered    = mean_real > delta_fwd

    # --- reverse: real covers synth ---
    # threshold from real internal distances
    nbrs_r      = NearestNeighbors(n_neighbors=k+1).fit(real)
    d, _        = nbrs_r.kneighbors(real)
    delta_rev   = np.percentile(d[:,1:].mean(axis=1), 95)

    nbrs_cross_r   = NearestNeighbors(n_neighbors=k).fit(real)
    dist_synth, _  = nbrs_cross_r.kneighbors(synth)
    mean_synth_pts = dist_synth.mean(axis=1)
    uncovered_synth = mean_synth_pts > delta_rev

    # --- grid (range from data extent) ---
    all_pts = np.vstack([real, synth])
    lo, hi  = all_pts.min(axis=0), all_pts.max(axis=0)
    pad     = (hi - lo) * 0.05
    gx, gy  = np.meshgrid(
        np.linspace(lo[0] - pad[0], hi[0] + pad[0], 300),
        np.linspace(lo[1] - pad[1], hi[1] + pad[1], 300),
    )
    grid = np.vstack([gx.ravel(), gy.ravel()]).T

    mean_grid_fwd = nbrs_cross.kneighbors(grid)[0].mean(axis=1).reshape(gx.shape)
    mean_grid_rev = nbrs_cross_r.kneighbors(grid)[0].mean(axis=1).reshape(gx.shape)

    fwd_grid = mean_grid_fwd <= delta_fwd
    rev_grid = mean_grid_rev <= delta_rev

    # 0=neither, 1=rev only, 2=fwd only, 3=both
    region_grid = (rev_grid.astype(int) + 2 * fwd_grid.astype(int)).astype(float)

    # --- boundary points for NN line annotations ---
    boundary_idx = np.argsort(np.abs(mean_real - delta_fwd))[:200][::40]
    _, nn_idx    = nbrs_cross.kneighbors(real[boundary_idx])

    # --- shaded 4-region background ---
    ax.contourf(gx, gy, region_grid,
                levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                colors=REGION_COLORS,
                alpha=0.5)

    # boundary contour lines for each direction
    ax.contour(gx, gy, mean_grid_fwd, levels=[delta_fwd],
               colors="forestgreen", linewidths=1.2, linestyles="--")
    ax.contour(gx, gy, mean_grid_rev, levels=[delta_rev],
               colors="royalblue",   linewidths=1.2, linestyles="--")

    # synthetic points: blue=covered by real, orange=not covered by real
    ax.scatter(synth[~uncovered_synth][:,0], synth[~uncovered_synth][:,1],
               color="royalblue", alpha=0.5, s=8)
    ax.scatter(synth[uncovered_synth][:,0], synth[uncovered_synth][:,1],
               color="orange", alpha=0.5, s=8)

    # real points: green=covered by synth, red=not covered
    ax.scatter(real[~uncovered][:,0], real[~uncovered][:,1],
               color="green", alpha=0.25, s=12, marker="x")
    ax.scatter(real[uncovered][:,0], real[uncovered][:,1],
               color="red", alpha=0.25, s=12, marker="x")

    # highlight boundary examples with NN lines and bubbles
    for ii, i in enumerate(boundary_idx):
        x, y  = real[i]
        r     = mean_real[i]
        color = "red" if uncovered[i] else "green"

        for j in nn_idx[ii]:
            sx, sy = synth[j]
            ax.plot([x, sx], [y, sy],
                    color=color, linewidth=0.5, alpha=0.5, zorder=3)

        ax.scatter(x, y, color=color, s=20, zorder=5)
        ax.add_patch(plt.Circle((x, y), r, fill=False,
                                color=color, linewidth=0.5))
        # ax.annotate(f"d={r:.3f}", (x, y),
                    # xytext=(4, 4), textcoords="offset points", fontsize=6)

    # axis limits
    if sc.get("centre_zero"):
        half = np.abs(all_pts).max() * 1.08
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
    else:
        cx   = (all_pts[:,0].min() + all_pts[:,0].max()) / 2
        cy   = (all_pts[:,1].min() + all_pts[:,1].max()) / 2
        half = max(all_pts[:,0].max() - all_pts[:,0].min(),
                   all_pts[:,1].max() - all_pts[:,1].min()) / 2 * 1.05
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    cov_fwd = (~uncovered).mean()
    cov_rev = (~uncovered_synth).mean()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{sc['title']}\nA $\\in$ B={cov_fwd:.1%}  B $\\in$ A={cov_rev:.1%}")
    ax.set_xticks([])
    ax.set_yticks([])

# legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [
    Patch(facecolor=REGION_COLORS[2], alpha=1.0, label="Covered by B"),
    Patch(facecolor=REGION_COLORS[1], alpha=1.0, label="Covered by A"),
    Patch(facecolor=REGION_COLORS[3], alpha=1.0, label="Covered by both"),
    Patch(facecolor=REGION_COLORS[0], alpha=1.0, label="Covered by neither"),
    Line2D([0],[0], color="forestgreen", lw=1.2, ls="--", label="B boundary $(\delta_B)$"),
    Line2D([0],[0], color="royalblue",   lw=1.2, ls="--", label="A boundary $(\delta_A)$"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.005), framealpha=1.0)

plt.tight_layout(rect=[0, 0.125, 1, 0.9])
plt.savefig("example_coverage.png", dpi=600, bbox_inches="tight")
# plt.show()
