"""
Cantilever deflection intervals vs x
=====================================
d = P*x**2 * (3*L - x) / (6*E*I)

x subdivided into crisp sub-intervals.
At each x, parameters P, L, E, I are treated as intervals (±5%).
Three enclosure methods are computed and plotted as shaded bands.
"""

import numpy as np
import matplotlib.pyplot as plt
import affapy.aa as aa
import affapy.ia as ia

# ── Parameters ─────────────────────────────────────────────────────
P_m, L_m, E_m, I_m = 500.0, 2.0, 210e9, 9e-8
f = 0.1       # ±10% uncertainty on each parameter
N = 10         # number of x points
 
xs = np.linspace(0, L_m, N, endpoint=True)
 
ia_lo, ia_hi   = np.zeros(N), np.zeros(N)
aa_lo, aa_hi   = np.zeros(N), np.zeros(N)
aas_lo, aas_hi = np.zeros(N), np.zeros(N)
 
for i, xv in enumerate(xs):
    xv = float(xv)
 
    # ── Interval Arithmetic ─────────────────────────────────────
    P = ia.Interval(P_m*(1-f), P_m*(1+f))
    L = ia.Interval(L_m*(1-f), L_m*(1+f))
    E = ia.Interval(E_m*(1-f), E_m*(1+f))
    I = ia.Interval(I_m*(1-f), I_m*(1+f))
    d = (P * xv**2 * (3*L - xv)) / (6 * E * I)
    print(d)
    ia_lo[i], ia_hi[i] = float(d.inf), float(d.sup)
 
    # ── Affine Arithmetic — independent symbols ──────────────────
    P = aa.Affine([P_m*(1-f), P_m*(1+f)])
    L = aa.Affine([L_m*(1-f), L_m*(1+f)])
    E = aa.Affine([E_m*(1-f), E_m*(1+f)])
    I = aa.Affine([I_m*(1-f), I_m*(1+f)])
    d = (P * xv**2 * (3*L - xv)) / (6 * E * I)
    aa_lo[i], aa_hi[i] = float(d.interval.inf), float(d.interval.sup)
 
    # ── Affine Arithmetic — shared symbol ε₁ ────────────────────
    P = aa.Affine(x0=P_m, xi={1: P_m*f})
    L = aa.Affine(x0=L_m, xi={1: L_m*f})
    E = aa.Affine(x0=E_m, xi={1: E_m*f})
    I = aa.Affine(x0=I_m, xi={1: I_m*f})
    d = (P * xv**2 * (3*L - xv)) / (6 * E * I)
    aas_lo[i], aas_hi[i] = float(d.interval.inf), float(d.interval.sup)
 
exact = P_m * xs**2 * (3*L_m - xs) / (6*E_m*I_m)
 
# Midpoints and half-widths for error bars
ia_mid  = (ia_lo  + ia_hi)  / 2;  ia_err  = (ia_hi  - ia_lo)  / 2
aa_mid  = (aa_lo  + aa_hi)  / 2;  aa_err  = (aa_hi  - aa_lo)  / 2
aas_mid = (aas_lo + aas_hi) / 2;  aas_err = (aas_hi - aas_lo) / 2
 
# Offset x slightly so the three sets of bars don't overlap
dx = L_m / N * 0.25
xs_ia  = xs - dx
xs_aa  = xs
xs_aas = xs + dx
 
# ── Plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
 
eb_kw = dict(fmt="o", markersize=2.5, capsize=2, lw=0.8, elinewidth=0.8)
 
ax.errorbar(xs_ia,  ia_mid,  yerr=ia_err,  color="#378ADD", label="IA",               **eb_kw)
ax.errorbar(xs_aa,  aa_mid,  yerr=aa_err,  color="#1D9E75", label="AA independent",   **eb_kw)
ax.errorbar(xs_aas, aas_mid, yerr=aas_err, color="#D85A30", label="AA shared ε₁",    **eb_kw)
ax.plot(xs, exact, '-o', lw=1.5, color="black", zorder=5, label="Exact midpoint")
 
ax.set_xlabel("x [m]", fontsize=16)
ax.set_ylabel("Displacement  δ  [m]", fontsize=16)
# ax.set_title(f"Cantilever deflection intervals  (±{int(f*100)}% uncertainty)")
ax.legend()
ax.grid(True, lw=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig("cantilever_intervals.png", dpi=150, bbox_inches="tight")
print("Saved: cantilever_intervals.png")