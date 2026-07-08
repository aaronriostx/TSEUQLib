"""
For each uncertainty propagation method: a pie chart of the midpoint
S1/S2 split (in percent) on top, and a bar chart below showing the
lower-upper range of S1 and S2 (also in percent).

S1 + S2 = 1 exactly for Y = X1 + 2*X2 (linear, no interaction), so S2 is
derived as the complement of S1's interval rather than propagated
separately. S1's bounds are computed live by importing compute_S1_bounds()
from sobol_ia_aa_tm.py, so both scripts always agree on the same numbers --
change the model or bounds there and this script picks it up automatically.

Each method is saved as its own separate figure file.
"""

import matplotlib.pyplot as plt
from sobol_ia_aa_tm import compute_S1_bounds

# pull S1 bounds straight from the IA/AA/TM computation
_results = compute_S1_bounds()
methods = {
    "interval_arithmetic": ("Interval arithmetic", *_results["interval_arithmetic"]),
    "affine_arithmetic": ("Affine arithmetic", *_results["affine_arithmetic"]),
    "taylor_model": ("Taylor model (1st order)", *_results["taylor_model"]),
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
color_S1 = "#378ADD"
color_S2 = "#1D9E75"


def plot_one(title, lo, hi, filename):
    S1_mid = (lo + hi) / 2
    S2_mid = 1 - S1_mid
    S1_lo_pct, S1_hi_pct, S1_mid_pct = lo * 100, hi * 100, S1_mid * 100
    S2_lo_pct, S2_hi_pct, S2_mid_pct = (1 - hi) * 100, (1 - lo) * 100, S2_mid * 100

    fig, (ax_pie, ax_bar) = plt.subplots(
        2, 1, figsize=(4, 4), gridspec_kw={"height_ratios": [4, 1]}
    )

    # ---- pie chart: midpoint split only, percentages inside the wedges ----
    _, _, autotexts = ax_pie.pie(
        [S1_mid_pct, S2_mid_pct],
        labeldistance=1.12,
        autopct="%1.1f%%",
        pctdistance=0.6,
        colors=[color_S1, color_S2],
        startangle=90,
        counterclock=True,
        wedgeprops={"edgecolor": SURFACE, "linewidth": 2},
        textprops={"color": INK},
    )
    for t in autotexts:
        t.set_color("black")
        t.set_fontweight("bold")

    # ---- error bar chart: range of S1 and S2 (horizontal) ----
    labels = [r"$S_{1}$", r"$S_{2}$"]
    los = [S1_lo_pct, S2_lo_pct]
    his = [S1_hi_pct, S2_hi_pct]
    mids = [S1_mid_pct, S2_mid_pct]
    colors = [color_S1, color_S2]
    y = range(len(labels))

    for yi, l, h, m, c in zip(y, los, his, mids, colors):
        ax_bar.errorbar(
            m, yi, xerr=[[m - l], [h - m]],
            fmt="o", color=c, ecolor="black", elinewidth=2, capsize=6,
            markersize=11, markeredgecolor="black", markeredgewidth=1, zorder=3,
        )
        # ax_bar.text(h + 1.5, yi, f"{h:.1f}%", va="center", fontsize=10)
        # ax_bar.text(l - 1.5, yi, f"{l:.1f}%", va="center", ha="right", fontsize=10)
        # ax_bar.text(m, yi + 0.18, f"{m:.1f}%", ha="center", fontsize=9, color="black")

    ax_bar.set_yticks(list(y))
    ax_bar.set_yticklabels(labels, fontsize=11)
    ax_bar.set_ylim(-0.6, len(labels) - 0.4)
    ax_bar.set_xlabel("Sobol index (%)")
    ax_bar.set_xlim(0, 100)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.grid(axis="x", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


for key, (title, lo, hi) in methods.items():
    plot_one(title, lo, hi, f"sobol_pie_{key}.png")

print("Saved: " + ", ".join(f"sobol_pie_{k}.png" for k in methods))