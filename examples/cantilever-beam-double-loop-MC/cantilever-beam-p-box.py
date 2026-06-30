"""
Cantilever beam, maximum bending stress, via double-loop Monte Carlo + p-box.

    sigma_max = (y_max * F * L) / I        [Pa]

End-loaded cantilever: max moment at the wall is M = F*L, and bending stress is
sigma = M*y_max/I.

Uncertainty split
-----------------
EPISTEMIC (intervals -- we only know bounds, no distribution assumed):
    F : end load   in [4.5, 5.5] kN
    L : length     in [1.9, 2.1] m
ALEATORY (random -- manufacturing variability, given distributions):
    y_max : distance to extreme fibre  ~ Normal(50 mm, 1.5 mm)
    I     : second moment of area      ~ Lognormal(mean 4.167e-6 m^4, CoV 5%)

Outer loop sweeps the (F,L) epistemic box; inner loop propagates (y_max, I).
The envelope of the resulting conditional CDFs is the probability box.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# ----- epistemic intervals (outer) -------------------------------------------
F_lo, F_hi = 4500.0, 5500.0          # N
L_lo, L_hi = 1.9, 2.1                # m

# ----- aleatory distributions (inner) ----------------------------------------
ymax_mean, ymax_std = 0.050, 0.0015  # m  (Normal)
I_mean, I_cov = 4.1667e-6, 0.05      # m^4 (Lognormal)
# convert lognormal mean/CoV -> underlying normal params
I_sln = np.sqrt(np.log(1 + I_cov**2))
I_mln = np.log(I_mean) - 0.5 * I_sln**2

def stress_MPa(ymax, F, L, I):
    return (ymax * F * L / I) / 1e6      # Pa -> MPa

# ----- loop sizes ------------------------------------------------------------
M_grid = 21                 # outer points per epistemic variable
N_inner = 60_000            # aleatory samples per outer point

F_vals = np.linspace(F_lo, F_hi, M_grid)
L_vals = np.linspace(L_lo, L_hi, M_grid)
EE = [(f, l) for f in F_vals for l in L_vals]
print(EE)

# common stress grid for pointwise envelope (from a pilot pass)
ymp = np.clip(rng.normal(ymax_mean, ymax_std, 200_000), 1e-6, None)
Ip = rng.lognormal(I_mln, I_sln, 200_000)
s_all = [stress_MPa(ymp, f, l, Ip) for f, l in EE]
s_lo = min(v.min() for v in s_all)
s_hi = max(v.max() for v in s_all)
grid = np.linspace(s_lo, s_hi, 500)

# ----- double loop -----------------------------------------------------------
cond = np.empty((len(EE), grid.size))
for k, (f, l) in enumerate(EE):
    ymax = np.clip(rng.normal(ymax_mean, ymax_std, N_inner), 1e-6, None)
    I = rng.lognormal(I_mln, I_sln, N_inner)        # INNER aleatory loop
    S = np.sort(stress_MPa(ymax, f, l, I))
    cond[k] = np.searchsorted(S, grid, side="right") / N_inner

cdf_left = cond.min(axis=0)     # upper-left CDF bound
cdf_right = cond.max(axis=0)    # lower-right CDF bound

# ----- contrast: single loop that fakes uniform priors on F, L ---------------
N1 = 5_000_000
F1 = rng.uniform(F_lo, F_hi, N1); L1 = rng.uniform(L_lo, L_hi, N1)
y1 = np.clip(rng.normal(ymax_mean, ymax_std, N1), 1e-6, None)
I1 = rng.lognormal(I_mln, I_sln, N1)
S1 = np.sort(stress_MPa(y1, F1, L1, I1))
cdf1 = np.searchsorted(S1, grid, side="right") / N1

# ----- design check: P(sigma > allowable) ------------------------------------
allow = 140.0   # MPa
j = np.searchsorted(grid, allow)
pf_lo = 1 - cdf_right[j]    # smallest possible failure prob
pf_hi = 1 - cdf_left[j]     # largest possible failure prob
print(f"P(sigma > {allow} MPa) bounded to [{pf_lo:.4f}, {pf_hi:.4f}]")
print(f"single faked-uniform run reports a single {1-cdf1[j]:.4f}")

# ----- plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 4))
for k in range(len(EE)):
    ax.plot(grid, cond[k], color="0.82", lw=0.4, zorder=1)
ax.fill_between(grid, cdf_left, cdf_right, color="#185FA5", alpha=0.18, zorder=2,
                label="P-box")
ax.plot(grid, cdf_left, color="#0C447C", lw=2.5, zorder=3, label="lower CDF bound")
ax.plot(grid, cdf_right, color="#185FA5", lw=2.5, zorder=3, label="upper CDF bound")
# ax.plot(grid, cdf1, color="#D85A30", lw=2.0, ls="--", zorder=4,
#         label="single loop w/ faked uniform F,L")
ax.axvline(allow, color="#A32D2D", lw=1.2, ls=":", zorder=5)
ax.text(allow + 1, 0.5, f"Failure threshold: \n{allow:.0f} MPa", color="#A32D2D", fontsize=10)
ax.set_xlabel(r"$\sigma_{max}$ (MPa)")
ax.set_ylabel(r"CDF, $P(\sigma_{max} ≤ s)$")
ax.set_ylim(0, 1)
# ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("cantilever_pbox.png", transparent=True, dpi=300)