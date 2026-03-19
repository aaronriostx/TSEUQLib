"""
P-box (Probability Box) analysis for y = 2*x1 + x2^2
  - x1 ~ Normal(mu, sigma)   — aleatory uncertainty
  - x2 in [a, b]             — epistemic uncertainty (fixed but unknown value)

The output is a family of CDFs indexed by x2, enveloped into a p-box.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Parameters ────────────────────────────────────────────────────────────────
MU    = 1.0   # mean of x1
SIGMA = 1.0   # std dev of x1
A     = 0.0   # lower bound of x2 interval
B     = 2.0   # upper bound of x2 interval

N_SAMPLES = 20_000   # Monte Carlo draws per x2 slice
N_SLICES  = 2       # number of x2 values swept across [A, B]
N_GRID    = 200      # resolution of the CDF x-axis

RNG = np.random.default_rng(seed=42)

# ── Sampling ──────────────────────────────────────────────────────────────────
def sample_y(x2_fixed: float, n: int) -> np.ndarray:
    """Draw n samples of y = 2*x1 + x2^2 for a fixed x2."""
    x1 = RNG.normal(MU, SIGMA, n)
    return 2.0 * x1 + x2_fixed ** 2

def empirical_cdf(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Evaluate the empirical CDF of `samples` at each point in `x_grid`."""
    sorted_s = np.sort(samples)
    return np.searchsorted(sorted_s, x_grid, side="right") / len(sorted_s)

# ── Compute p-box ─────────────────────────────────────────────────────────────
x2_values = np.linspace(A, B, N_SLICES)

# Global y range (with a little padding)
y_min_global =  np.inf
y_max_global = -np.inf
all_samples = []
for x2 in x2_values:
    s = sample_y(x2, N_SAMPLES)
    all_samples.append(s)
    y_min_global = min(y_min_global, s.min())
    y_max_global = max(y_max_global, s.max())

pad = (y_max_global - y_min_global) * 0.05
y_grid = np.linspace(y_min_global - pad, y_max_global + pad, N_GRID)

cdfs = np.array([empirical_cdf(s, y_grid) for s in all_samples])  # (N_SLICES, N_GRID)

lower_env = cdfs.min(axis=0)   # F̲  — most optimistic (rightmost)
upper_env = cdfs.max(axis=0)   # F̄  — most pessimistic (leftmost)

# ── Analytical bounds (exact) ─────────────────────────────────────────────────
# E[y | x2] = 2*mu + x2^2  →  extremes at endpoints / zero-crossing of x2^2
x2sq = x2_values ** 2
mean_lo = 2 * MU + x2sq.min()
mean_hi = 2 * MU + x2sq.max()

# Var[y | x2] = 4*sigma^2  (x2 is fixed, so only x1 contributes)
var_y = 4 * SIGMA ** 2
std_y = np.sqrt(var_y)

# ── Print summary ─────────────────────────────────────────────────────────────
print("=" * 52)
print("  P-BOX SUMMARY  |  y = 2·x1 + x2²")
print("=" * 52)
print(f"  x1 ~ N({MU}, {SIGMA}²)")
print(f"  x2 ∈ [{A}, {B}]  (epistemic, not random)")
print()
print(f"  E[y]   ∈ [{mean_lo:.3f}, {mean_hi:.3f}]  (interval, not a point)")
print(f"  Var[y]  = {var_y:.3f}  (exact — x2 fixed, only x1 contributes)")
print(f"  Std[y]  = {std_y:.3f}")
print(f"  x2² range: [{x2sq.min():.3f}, {x2sq.max():.3f}]")
print("=" * 52)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    r"P-box for $y = 2 x_1 + x_2^2$"
    f"\n"
    r"$x_1 \sim \mathcal{{N}}$"
    f"({MU}, {SIGMA}²)    "
    r"$x_2 \in$"
    f" [{A}, {B}]  (epistemic)",
    fontsize=12, y=1.01
)

PURPLE = "#534AB7"
GREEN  = "#3B6D11"
FILL   = "#534AB7"

# ── Left panel: the full p-box ────────────────────────────────────────────────
ax = axes[0]

# inner CDFs (faint)
cmap = plt.cm.coolwarm
for i, (x2, cdf) in enumerate(zip(x2_values, cdfs)):
    color = cmap(i / (N_SLICES - 1))
    ax.plot(y_grid, cdf, color=color, alpha=0.25, linewidth=0.9)

# envelope
ax.plot(y_grid, upper_env, color=GREEN,  linewidth=2.2, label="Upper envelope F\u0305")
ax.plot(y_grid, lower_env, color=PURPLE, linewidth=2.2, label="Lower envelope F\u0332")
ax.fill_betweenx(
    np.linspace(0, 1, N_GRID),
    np.interp(np.linspace(0, 1, N_GRID), upper_env, y_grid),
    np.interp(np.linspace(0, 1, N_GRID), lower_env, y_grid),
    alpha=0.10, color=FILL
)
ax.fill_between(y_grid, lower_env, upper_env, alpha=0.08, color=FILL)

ax.set_xlabel("y", fontsize=11)
ax.set_ylabel("P(Y ≤ y)", fontsize=11)
ax.set_title("P-box (CDF envelope)", fontsize=11)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3, linewidth=0.5)

inner_patch = Line2D([0], [0], color="gray", alpha=0.5, linewidth=1,
                     label=f"CDF slices ({N_SLICES} values of x₂)")
ax.legend(handles=[
    Line2D([0], [0], color=GREEN,  linewidth=2, label=r"Upper envelope $\bar{F}$"),
    Line2D([0], [0], color=PURPLE, linewidth=2, label="Lower envelope F\u0332"),
    inner_patch,
], fontsize=9, loc="lower right")

# ── Right panel: p-box width = epistemic gap ──────────────────────────────────
ax2 = axes[1]

width = np.interp(
    np.linspace(0, 1, N_GRID),
    lower_env,
    y_grid
) - np.interp(
    np.linspace(0, 1, N_GRID),
    upper_env,
    y_grid
)
prob_levels = np.linspace(0, 1, N_GRID)

ax2.fill_betweenx(prob_levels, 0, width, alpha=0.25, color=FILL)
ax2.plot(width, prob_levels, color=PURPLE, linewidth=2)
ax2.axvline(width.max(), color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax2.text(width.max() * 1.02, 0.5, f"max gap\n{width.max():.2f}",
         fontsize=8, color="gray", va="center")

ax2.set_xlabel("P-box width  (epistemic gap in y)", fontsize=11)
ax2.set_ylabel("Probability level", fontsize=11)
ax2.set_title("Epistemic uncertainty gap", fontsize=11)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlim(left=0)
ax2.grid(True, alpha=0.3, linewidth=0.5)

# ── Annotation box ────────────────────────────────────────────────────────────
info = (
    f"E[y] ∈ [{mean_lo:.2f}, {mean_hi:.2f}]\n"
    f"Var[y] = {var_y:.2f}  (aleatory only)\n"
    f"Std[y] = {std_y:.2f}"
)
fig.text(0.5, -0.04, info, ha="center", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                   edgecolor="#cccccc", linewidth=0.8))

plt.tight_layout()
out_path = "pbox_simulation.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
plt.show()