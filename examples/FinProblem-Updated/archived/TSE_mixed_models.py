import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import fin_problem as fp
from fin_params_mixed import MEAN, Tinf, Tw, b, d, L, x, t

os.makedirs("figures", exist_ok=True)


# ── Nominal point ──────────────────────────────────────────────────────────────
k_nom    = MEAN[0]
Cp_nom   = MEAN[1]
rho_nom  = MEAN[2]
hU_nom   = MEAN[3]
Tinf_nom = (Tinf[0] + Tinf[1]) / 2
Tw_nom   = (Tw[0]   + Tw[1])   / 2
b_nom    = (b[0]    + b[1])    / 2
T0_nom   = Tinf_nom

delta_Tinf = (Tinf[1] - Tinf[0]) / 2
delta_Tw   = (Tw[1]   - Tw[0])   / 2
delta_b    = (b[1]    - b[0])    / 2

def eval_T(Tinf_v, Tw_v, b_v):
    T, _, _, _ = fp.analytic_solution(
        x, t, b_v, d, L, rho_nom, Cp_nom, k_nom, hU_nom,
        T0=Tinf_v, Tw=Tw_v, Tinf=Tinf_v
    )
    return T

T_nom = eval_T(Tinf_nom, Tw_nom, b_nom)

h_Tinf = 1e-5 * Tinf_nom
h_Tw   = 1e-5 * Tw_nom
h_b    = 1e-5 * b_nom

dT_dTinf = (eval_T(Tinf_nom + h_Tinf, Tw_nom, b_nom) -
            eval_T(Tinf_nom - h_Tinf, Tw_nom, b_nom)) / (2 * h_Tinf)
dT_dTw   = (eval_T(Tinf_nom, Tw_nom + h_Tw, b_nom) -
            eval_T(Tinf_nom, Tw_nom - h_Tw, b_nom)) / (2 * h_Tw)
dT_db    = (eval_T(Tinf_nom, Tw_nom, b_nom + h_b) -
            eval_T(Tinf_nom, Tw_nom, b_nom - h_b)) / (2 * h_b)


# ── Affine and Taylor model classes ────────────────────────────────────────────

class AffineForm:
    """
    Affine arithmetic representation:
        x̂ = x₀ + c₁·ε₁ + c₂·ε₂ + ... + cₙ·εₙ,   εᵢ ∈ [-1, +1]

    Each εᵢ is a symbolic noise variable tied to one independent
    source of epistemic uncertainty.
    """

    def __init__(self, center: float, coeffs: dict):
        self.center = float(center)
        self.coeffs = coeffs          # {symbol_name: coefficient}

    def radius(self) -> float:
        return sum(abs(c) for c in self.coeffs.values())

    def to_interval(self) -> tuple:
        r = self.radius()
        return (self.center - r, self.center + r)

    def __repr__(self):
        terms = "  ".join(
            f"({c:+.6f})·{name}" for name, c in self.coeffs.items()
        )
        return f"{self.center:.6f}  +  {terms}"


class TaylorModel:
    """
    1st-order Taylor model:
        TM = (p, R)
        p(ε) = x₀ + c₁·ε₁ + c₂·ε₂ + ... + cₙ·εₙ   (polynomial part)
        R = [r_lo, r_hi]                               (remainder interval)

    R bounds the approximation error from truncating the Taylor series
    at first order, i.e., the contribution of 2nd-order and higher terms.
    """

    def __init__(self, affine: AffineForm, remainder: tuple):
        self.affine    = affine
        self.remainder = remainder    # (r_lo, r_hi)

    def to_interval(self) -> tuple:
        lo, hi = self.affine.to_interval()
        return (lo + self.remainder[0], hi + self.remainder[1])

    def __repr__(self):
        r_lo, r_hi = self.remainder
        return (f"{self.affine!r}\n"
                f"  + R,   R ∈ [{r_lo:+.6f}, {r_hi:+.6f}]")


# ── Extract scalar values at the last time step (t = 450 s) ───────────────────
idx = -1

T0       = T_nom[idx]
c_Tinf   = dT_dTinf[idx] * delta_Tinf   # affine coefficient for ε_Tinf
c_Tw     = dT_dTw[idx]   * delta_Tw     # affine coefficient for ε_Tw
c_b      = dT_db[idx]    * delta_b      # affine coefficient for ε_b

# ── Affine model ───────────────────────────────────────────────────────────────
af = AffineForm(
    center = T0,
    coeffs = {"ε_Tinf": c_Tinf, "ε_Tw": c_Tw, "ε_b": c_b},
)

# ── Taylor model remainder via corner evaluation ───────────────────────────────
# Remainder = actual T at corner  −  1st-order approximation at that corner.
# Evaluated at all 2³ = 8 corners of the epistemic box.
errors = []
for s1, s2, s3 in product([-1, 1], [-1, 1], [-1, 1]):
    T_actual = eval_T(
        Tinf_nom + s1 * delta_Tinf,
        Tw_nom   + s2 * delta_Tw,
        b_nom    + s3 * delta_b,
    )[idx]
    T_approx = T0 + s1 * c_Tinf + s2 * c_Tw + s3 * c_b
    errors.append(T_actual - T_approx)

remainder = (min(errors), max(errors))
tm = TaylorModel(af, remainder)

# ── Print results ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  t = {t[idx]:.1f} s")
print(f"{'='*60}")

print(f"\nPartial derivatives at nominal point:")
print(f"  ∂T/∂Tinf = {dT_dTinf[idx]:.6f}  K/K")
print(f"  ∂T/∂Tw   = {dT_dTw[idx]:.6f}  K/K")
print(f"  ∂T/∂b    = {dT_db[idx]:.6f}  K/mm")

print(f"\nEpistemic half-widths (Δyᵢ):")
print(f"  ΔTinf = {delta_Tinf:.4f}  K")
print(f"  ΔTw   = {delta_Tw:.4f}  K")
print(f"  Δb    = {delta_b:.4f}  mm")

print(f"\n--- Affine model ---")
print(f"  Ê[T] = {af}")
print(f"\n  Individual half-width contributions:")
for name, c in af.coeffs.items():
    print(f"    |{name} term| = {abs(c):.6f} K")
print(f"  Total radius  = {af.radius():.6f} K")
print(f"  Interval      = [{af.to_interval()[0]:.6f}, {af.to_interval()[1]:.6f}] K")

print(f"\n--- Taylor model (order 1) ---")
print(f"  E[T] = {tm}")
print(f"  Enclosure = [{tm.to_interval()[0]:.6f}, {tm.to_interval()[1]:.6f}] K")

# ── Epistemic contribution plot ────────────────────────────────────────────────
# Absolute half-width each source contributes at every time step
contrib = {
    r"$T_\infty$": np.abs(dT_dTinf) * delta_Tinf,
    r"$T_W$":      np.abs(dT_dTw)   * delta_Tw,
    r"$b$":        np.abs(dT_db)    * delta_b,
}
total     = sum(contrib.values())
colors    = ["royalblue", "tomato", "seagreen"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=300)

# Panel 1 — absolute contributions vs time
for (label, vals), color in zip(contrib.items(), colors):
    ax1.plot(t, vals, label=label, color=color, linewidth=1.5)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Half-width contribution [K]")
ax1.set_title("Absolute epistemic contributions")
ax1.legend(fontsize=8)
ax1.grid(True, linestyle="--", alpha=0.4)

# Panel 2 — fractional contributions (stacked area, normalised to 100 %)
fracs  = [v / total * 100 for v in contrib.values()]
labels = list(contrib.keys())
ax2.stackplot(t, *fracs, labels=labels, colors=colors, alpha=0.8)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Fractional contribution [%]")
ax2.set_title("Relative epistemic contributions")
ax2.set_ylim(0, 100)
# ax2.legend(loc="center right", fontsize=8)
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("figures/TSE_mixed_epistemic_contributions.png", transparent=True)
plt.close()
print("\nContribution plot saved to figures/TSE_mixed_epistemic_contributions.png")
