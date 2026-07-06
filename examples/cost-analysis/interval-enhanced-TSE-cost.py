"""
Surface plot of T(d+p, N) = C(d+p+N, N), the number of terms in the
combined n-th order Taylor series expansion, for N = 1, 2, 3 overlaid
on a single set of 3D axes.

d = n + m  (aleatory variables + interval variables, combined dimension)
p          (number of interval-valued hyperparameters)

Tweak d_max, p_max, N_values, or colors below to explore other ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from math import comb

# ---- parameters you can tweak ----
d_max = 20          # d ranges from 1 to d_max
p_max = 10          # p ranges from 0 to p_max
N_values = [1, 2, 3, 4, 5]
colors = ['blue', 'green', 'yellow', 'orange', 'red']   # one solid color per N
alpha = 0.75
log_z = True        # set False to plot raw T instead of log10(T)
# -----------------------------------

d_vals = np.arange(1, d_max)
p_vals = np.arange(0, p_max + 1)
D, P = np.meshgrid(d_vals, p_vals)


def T(d, p, N):
    return comb(int(d) + int(p) + N, N)


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

legend_handles = []
for N, color in zip(N_values, colors):
    Z = np.array([[T(d, p, N) for d in d_vals] for p in p_vals], dtype=float)
    Z_plot = np.log10(Z) if log_z else Z

    ax.plot_surface(D, P, Z_plot, color=color, alpha=alpha, edgecolor='none', antialiased=True)
    legend_handles.append(Patch(facecolor=color, alpha=alpha, label=f'N = {N} (max T = {int(Z.max()):,})'))

ax.set_xlabel(r'Total number of variables, $d = r + s$')
ax.set_ylabel(r'Number of interval hyperparameters, $p$ ')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.view_init(elev=25, azim=-60)
ax.legend(handles=legend_handles, loc='upper left')
 
if log_z:
    # Data is already log10(T), so the z-axis is linear in log space.
    # Force gridlines at whole decades and relabel them as the real T value
    # (1, 10, 100, 1,000, ...) instead of the raw log10 number.
    ax.zaxis.set_major_locator(MultipleLocator(1))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(round(val))}}}$'))
    ax.set_zlabel('Total number of terms (log scale)')
else:
    ax.set_zlabel('Total number of terms')

plt.tight_layout()
plt.savefig('taylor_term_surfaces.png', dpi=300, transparent=True)