import os
import numpy as np
import matplotlib.pyplot as plt
import fin_problem as fp
from fin_params_mixed import MEAN, Tinf, Tw, b, d, L, x, t

os.makedirs("figures", exist_ok=True)


class Interval:
    """Closed interval [lo, hi] with numpy array-compatible bounds."""

    __array_ufunc__ = None  # prevent numpy from intercepting arithmetic

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.lo + other.lo, self.hi + other.hi)
        return Interval(self.lo + other, self.hi + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, scalar):
        a = scalar * self.lo
        b = scalar * self.hi
        return Interval(np.minimum(a, b), np.maximum(a, b))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


# ── Nominal point ─────────────────────────────────────────────────────────────
k_nom    = MEAN[0]
Cp_nom   = MEAN[1]
rho_nom  = MEAN[2]
hU_nom   = MEAN[3]
Tinf_nom = (Tinf[0] + Tinf[1]) / 2
Tw_nom   = (Tw[0]   + Tw[1])   / 2
b_nom    = (b[0]    + b[1])    / 2
T0_nom   = Tinf_nom  # T0 = Tinf assumption

def eval_T(Tinf_v, Tw_v, b_v):
    """Evaluate T at the fin tip with stochastic params fixed at their means.
    T0 is coupled to Tinf per the model assumption T0 = Tinf."""
    T, _, _, _ = fp.analytic_solution(
        x, t, b_v, d, L, rho_nom, Cp_nom, k_nom, hU_nom,
        T0=Tinf_v, Tw=Tw_v, Tinf=Tinf_v
    )
    return T

T_nom = eval_T(Tinf_nom, Tw_nom, b_nom)

# ── Partial derivatives via central finite differences ─────────────────────────
h_Tinf = 1e-5 * Tinf_nom
h_Tw   = 1e-5 * Tw_nom
h_b    = 1e-5 * b_nom

dT_dTinf = (eval_T(Tinf_nom + h_Tinf, Tw_nom, b_nom) -
            eval_T(Tinf_nom - h_Tinf, Tw_nom, b_nom)) / (2 * h_Tinf)

dT_dTw   = (eval_T(Tinf_nom, Tw_nom + h_Tw, b_nom) -
            eval_T(Tinf_nom, Tw_nom - h_Tw, b_nom)) / (2 * h_Tw)

dT_db    = (eval_T(Tinf_nom, Tw_nom, b_nom + h_b) -
            eval_T(Tinf_nom, Tw_nom, b_nom - h_b)) / (2 * h_b)

# ── Centered deviation intervals ───────────────────────────────────────────────
delta_Tinf = (Tinf[1] - Tinf[0]) / 2   # half-width
delta_Tw   = (Tw[1]   - Tw[0])   / 2
delta_b    = (b[1]    - b[0])    / 2

I_Tinf = Interval(-delta_Tinf, delta_Tinf)
I_Tw   = Interval(-delta_Tw,   delta_Tw)
I_b    = Interval(-delta_b,    delta_b)

# ── 1st-order TSE interval arithmetic ─────────────────────────────────────────
# Each term: (∂T/∂y_i) * [-Δy_i, +Δy_i]
# Sum of intervals: [lo1+lo2+lo3, hi1+hi2+hi3]
bounds = dT_dTinf * I_Tinf + dT_dTw * I_Tw + dT_db * I_b

T_lb = T_nom + bounds.lo
T_ub = T_nom + bounds.hi

# ── Save ───────────────────────────────────────────────────────────────────────
np.savez("TSE_mixed_data.npz",
         t=t,
         T_expectation=T_nom,
         T_lb=T_lb,
         T_ub=T_ub,
         dT_dTinf=dT_dTinf,
         dT_dTw=dT_dTw,
         dT_db=dT_db)

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.figure(figsize=(4, 3), dpi=300)
plt.fill_between(t, T_lb, T_ub, alpha=0.3, color='b', label='Epistemic bounds')
plt.plot(t, T_nom, 'b', label=r'$\mathbb{E}[T]$ (nominal)')
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel(r'$\mathbb{E}[T]$ [K]')
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig('figures/TSE_mixed_T_expectation_vs_t.png')
plt.close()
