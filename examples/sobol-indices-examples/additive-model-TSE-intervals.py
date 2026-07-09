"""
For each uncertainty propagation method: a pie chart of the midpoint
S1/S2 split (in percent) on top, and a bar chart below showing the
lower-upper range of S1 and S2 (also in percent).

Both S1 and S2 are propagated directly (each through its own formula via
compute_S1_bounds() / compute_S2_bounds() in sobol_ia_aa_tm.py), rather than
deriving S2 as 1 - S1. The complement identity only holds because this toy
model has exactly two additive, non-interacting variables -- it doesn't
generalize to more variables or to models with interactions, so the direct
method is what's shown here. Both scripts pull straight from
sobol_ia_aa_tm.py, so changing the model or bounds there is picked up
automatically.

Each method is saved as its own separate figure file.
"""

import matplotlib.pyplot as plt
from sobol_ia_aa_tm import compute_S1_bounds, compute_S2_bounds

_results_S1 = compute_S1_bounds()
_results_S2 = compute_S2_bounds()

methods = {
    "interval_arithmetic": "Interval arithmetic",
    "affine_arithmetic": "Affine arithmetic",
    "taylor_model": "Taylor model (1st order)",
}
SURFACE = "#fcfcfb"
color_S1 = "#378ADD"
color_S2 = "#1D9E75"


def plot_one(title, S1_bounds, S2_bounds, filename):
    S1_lo, S1_hi = S1_bounds
    S2_lo, S2_hi = S2_bounds

    S1_mid_pct = (S1_lo + S1_hi) / 2 * 100
    S2_mid_pct = (S2_lo + S2_hi) / 2 * 100

    fig, (ax_pie, ax_bar) = plt.subplots(
        2, 1, figsize=(4.5, 4.5), gridspec_kw={"height_ratios": [4, 1.4]}
    )

    # ---- pie chart: midpoint split, no percentage labels (ax.pie() silently
    # renormalizes wedges to sum to 100%, which would misrepresent S1_mid/S2_mid
    # now that they no longer sum to 1 for every method) ----
    ax_pie.pie(
        [S1_mid_pct, S2_mid_pct],
        colors=[color_S1, color_S2],
        startangle=90,
        counterclock=True,
        wedgeprops={"edgecolor": SURFACE, "linewidth": 2},
    )
    # ax_pie.set_title(title, fontsize=11)

    # ---- error bar chart: range of S1 and S2 (horizontal) ----
    rows = [
        (r"$S_1$", S1_lo * 100, S1_hi * 100, color_S1),
        (r"$S_2$", S2_lo * 100, S2_hi * 100, color_S2),
    ]
    y = range(len(rows))

    for yi, (_, lo, hi, c) in zip(y, rows):
        mid = (lo + hi) / 2
        ax_bar.errorbar(
            mid, yi, xerr=[[mid - lo], [hi - mid]],
            fmt="o", color=c, ecolor="black", elinewidth=2, capsize=6,
            markersize=11, markeredgecolor="black", markeredgewidth=1, zorder=3,
        )
        ax_bar.text(hi + 1.5, yi, f"{hi:.1f}%", va="center", fontsize=8)
        ax_bar.text(lo - 1.5, yi, f"{lo:.1f}%", va="center", ha="right", fontsize=8)

    x_max = max(100, *(hi for _, _, hi, _ in rows)) + 8
    x_min = min(0, *(lo for _, lo, _, _ in rows)) - 8

    ax_bar.axvline(100, color="gray", linestyle="--", linewidth=1)
    # ax_bar.text(
    #     100, 1.06, "100% (valid limit)", transform=ax_bar.get_xaxis_transform(),
    #     ha="center", va="bottom", fontsize=7.5, color="gray", clip_on=False,
    # )
    ax_bar.set_yticks(list(y))
    ax_bar.set_yticklabels([label for label, *_ in rows], fontsize=11)
    ax_bar.set_ylim(-0.6, len(rows) - 0.4)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Sobol index (%)")
    ax_bar.set_xlim(x_min, x_max)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.grid(axis="x", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


for key, title in methods.items():
    plot_one(title, _results_S1[key], _results_S2[key], f"sobol_pie_{key}.png")

print("Saved: " + ", ".join(f"sobol_pie_{k}.png" for k in methods))
