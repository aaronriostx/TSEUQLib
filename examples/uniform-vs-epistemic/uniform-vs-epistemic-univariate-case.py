"""
Propagating X through the square: distribution vs interval.

Builds a 2x2 figure comparing how a uniform random variable X ~ U[0,2]
and a closed interval X = [0,2] behave when squared.

  Top-left   : PDF of X ~ U[0,2]            (flat density = 1/2)
  Top-right  : interval X = [0,2]           (vertical lines at 0 and 2)
  Bottom-left: PDF of Y = X^2 = 1/(4*sqrt(y)) on (0,4], mean = 4/3
  Bottom-right: interval X^2 = [0,4]        (vertical lines at 0 and 4)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

blue = "#185FA5"
coral = "#D85A30"

fig, ax = plt.subplots(2, 2, figsize=(8, 5))

# ---- Top-left: X ~ U[0,2] PDF ------------------------------------------------
a = ax[0, 0]
a.plot([0, 0, 2, 2], [0, 0.5, 0.5, 0], color=blue, lw=2.5)
a.fill_between([0, 2], [0.5, 0.5], color=blue, alpha=0.15)
a.axvline(1, color="black", ls="--", lw=1.3)
a.text(1 + 0.08, 0.65, r"$\mathbb{E}[X]$ = 1", color="black", fontsize=14)
a.set_xlim(-1, 5)
a.set_ylim(0, 0.9)
a.set_title("X ~ U[0,2] (PDF)")
a.set_xlabel("x")
a.set_ylabel("Density f(x)")

# ---- Top-right: interval X = [0,2] -------------------------------------------
b = ax[0, 1]
for xv in (0, 2):
    b.axvline(xv, color=coral, lw=2.5)
b.annotate("", xy=(2, 0.5), xytext=(0, 0.5),
           arrowprops=dict(arrowstyle="<->", color=coral, lw=1.5))
b.set_xlim(-1, 5)
b.set_ylim(0, 1)
b.set_title("X = [0,2] (interval)")
b.set_xlabel("x")
b.set_yticks([])
b.text(1, 0.55, "any value", ha="center", color=coral, fontsize=14)

# ---- Bottom-left: PDF of Y = X^2 = 1/(4*sqrt(y)) -----------------------------
c = ax[1, 0]
y = np.linspace(0.02, 4, 400)
c.plot(y, 1 / (4 * np.sqrt(y)), color=blue, lw=2.5)
c.fill_between(y, 1 / (4 * np.sqrt(y)), color=blue, alpha=0.15)
c.axvline(4 / 3, color="black", ls="--", lw=1.3)
c.text(4 / 3 + 0.08, 0.75, r"$\mathbb{E}[X^2]$ = 4/3", color="black", fontsize=14)
c.set_xlim(-1, 5)
c.set_ylim(0, 1.3)
c.set_title(r"X$^2$  =  1/(4$\sqrt{y}$) (PDF)")
c.set_xlabel("y = x²")
c.set_ylabel("Density f(y)")

# ---- Bottom-right: interval X^2 = [0,4] --------------------------------------
d = ax[1, 1]
for xv in (0, 4):
    d.axvline(xv, color=coral, lw=2.5)
d.annotate("", xy=(4, 0.5), xytext=(0, 0.5),
           arrowprops=dict(arrowstyle="<->", color=coral, lw=1.5))
d.set_xlim(-1, 5)
d.set_ylim(0, 1)
d.set_title("X² = [0,4]  (interval)")
d.set_xlabel("y = x²")
d.set_yticks([])
d.text(2, 0.55, "any value", ha="center", color=coral, fontsize=14)

for axx in ax.flat:
    axx.grid(alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("uniform_vs_interval_Xsquared.png", dpi=150, bbox_inches="tight", transparent=True)