"""
For each uncertainty propagation method (interval arithmetic, affine
arithmetic, 1st-order Taylor model): a pie chart of the midpoint S_k/S_Cp/
S_rho/S_hU split (in percent) on top, and an error-bar chart below showing
the lower-upper range of each index (also in percent), at t=450s.

All four indices are propagated directly through their own formula in
compute_Si_bounds() (sobol_1st_order_ia_aa_tm.py), not derived from each
other -- the complement identity sum_i S_i = 1 only holds exactly at a fixed
sigma_hU (see TSE_1st_order_derivation.md); it does not hold for the bounds
of each index individually, since different indices attain their own extrema
at different points in the sigma_hU interval. Both scripts pull straight
from sobol_1st_order_ia_aa_tm.py, so changing the model there is picked up
automatically here.

Each method is saved as its own separate figure file.
"""
import matplotlib.pyplot as plt
from sobol_1st_order_ia_aa_tm import compute_Si_bounds

variables = ["k", "Cp", "rho", "hU"]
labels = {"k": r"$S_k$", "Cp": r"$S_{C_p}$", "rho": r"$S_\rho$", "hU": r"$S_{h_U}$"}
colors = {"k": "#2a78d6", "Cp": "#1baf7a", "rho": "#eda100", "hU": "#008300"}

_results = {v: compute_Si_bounds(v) for v in variables}

methods = {
    "interval_arithmetic": "Interval arithmetic",
    "affine_arithmetic": "Affine arithmetic",
    "taylor_model": "Taylor model (1st order)",
}
SURFACE = "#fcfcfb"


def plot_one(method_key, filename):
    bounds = {v: _results[v][method_key] for v in variables}
    mids_pct = {v: (lo + hi) / 2 * 100 for v, (lo, hi) in bounds.items()}

    fig, (ax_pie, ax_bar) = plt.subplots(
        2, 1, figsize=(4.5, 6), gridspec_kw={"height_ratios": [4, 2.2]}
    )

    # ---- pie chart: midpoint split, no percentage labels (ax.pie() silently
    # renormalizes wedges to sum to 100%, which would misrepresent the S_i
    # midpoints now that they no longer sum to exactly 1 for every method).
    # Identity is carried by color, matched to the labeled rows below. ----
    ax_pie.pie(
        [mids_pct[v] for v in variables],
        colors=[colors[v] for v in variables],
        startangle=90,
        counterclock=True,
        wedgeprops={"edgecolor": SURFACE, "linewidth": 2},
    )

    # ---- error bar chart: range of each S_i (horizontal) ----
    y = range(len(variables))
    for yi, v in zip(y, variables):
        lo, hi = bounds[v][0] * 100, bounds[v][1] * 100
        mid = (lo + hi) / 2
        ax_bar.errorbar(
            mid, yi, xerr=[[mid - lo], [hi - mid]],
            fmt="o", color=colors[v], ecolor="black", elinewidth=2, capsize=6,
            markersize=11, markeredgecolor="black", markeredgewidth=1, zorder=3,
        )
        ax_bar.text(hi + 1.5, yi, f"{hi:.1f}%", va="center", fontsize=8)
        ax_bar.text(lo - 1.5, yi, f"{lo:.1f}%", va="center", ha="right", fontsize=8)

    all_los = [bounds[v][0] * 100 for v in variables]
    all_his = [bounds[v][1] * 100 for v in variables]
    x_max = max(100, *all_his) + 8
    x_min = min(0, *all_los) - 8

    ax_bar.axvline(100, color="gray", linestyle="--", linewidth=1)
    ax_bar.set_yticks(list(y))
    ax_bar.set_yticklabels([labels[v] for v in variables], fontsize=11)
    ax_bar.set_ylim(-0.6, len(variables) - 0.4)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Sobol index (%)")
    ax_bar.set_xlim(x_min, x_max)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.grid(axis="x", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


for key in methods:
    plot_one(key, f"figures/sobol_1st_order_pie_{key}.png")

print("Saved: " + ", ".join(f"figures/sobol_1st_order_pie_{k}.png" for k in methods))
