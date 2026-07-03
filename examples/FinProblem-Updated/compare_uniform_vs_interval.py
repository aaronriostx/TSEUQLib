import os
import numpy as np
import matplotlib.pyplot as plt
import fin_problem as fp
from fin_params_all_aleatoric import (
    num_of_samples, k, Cp, rho, hU,
    Tinf as Tinf_u, Tw as Tw_u, b as b_u, T0 as T0_u,
    d, L, x, t, MEAN, LOW, HIGH,
)
from fin_params_mixed import (
    Tinf as Tinf_iv, Tw as Tw_iv, b as b_iv,
)

os.makedirs("figures", exist_ok=True)


# ── Case 1: Uniform distributions → MCS expectation ──────────────────────────
T_samples = np.zeros((num_of_samples, len(t)))
for i in range(num_of_samples):
    T, _, _, _ = fp.analytic_solution(
        x, t, b_u[i], d, L, rho[i], Cp[i], k[i], hU[i],
        T0_u[i], Tw_u[i], Tinf_u[i],
    )
    T_samples[i, :] = T

T_expectation = np.mean(T_samples, axis=0)


# ── Case 2: 1st-order TSE about the mean → E[T] ≈ T(μ) ──────────────────────
# For a 1st-order expansion, E[T(X)] ≈ T(μ) because E[X - μ] = 0 for any distribution.
# Uniform mean = (LOW + HIGH) / 2; normal mean = MEAN[0:4].
T_tse, _, _, _ = fp.analytic_solution(
    x, t,
    (LOW[2] + HIGH[2]) / 2,          # b mean
    d, L,
    MEAN[2],                          # rho mean
    MEAN[1],                          # Cp mean
    MEAN[0],                          # k mean
    MEAN[3],                          # hU mean
    (LOW[0] + HIGH[0]) / 2,          # T0 mean (= Tinf mean)
    (LOW[1] + HIGH[1]) / 2,          # Tw mean
    (LOW[0] + HIGH[0]) / 2,          # Tinf mean
)


# ── Case 3: Intervals → TSE interval arithmetic ──────────────────────────────
class Interval:
    __array_ufunc__ = None

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
        b_prod = scalar * self.hi
        return Interval(np.minimum(a, b_prod), np.maximum(a, b_prod))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


k_nom   = MEAN[0]
Cp_nom  = MEAN[1]
rho_nom = MEAN[2]
hU_nom  = MEAN[3]

Tinf_nom = (Tinf_iv[0] + Tinf_iv[1]) / 2
Tw_nom   = (Tw_iv[0]   + Tw_iv[1])   / 2
b_nom    = (b_iv[0]    + b_iv[1])    / 2
T0_nom   = Tinf_nom


def eval_T(Tinf_v, Tw_v, b_v):
    T, _, _, _ = fp.analytic_solution(
        x, t, b_v, d, L, rho_nom, Cp_nom, k_nom, hU_nom,
        Tinf_v, Tw_v, Tinf_v,
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

delta_Tinf = (Tinf_iv[1] - Tinf_iv[0]) / 2
delta_Tw   = (Tw_iv[1]   - Tw_iv[0])   / 2
delta_b    = (b_iv[1]    - b_iv[0])    / 2

I_Tinf = Interval(-delta_Tinf, delta_Tinf)
I_Tw   = Interval(-delta_Tw,   delta_Tw)
I_b    = Interval(-delta_b,    delta_b)

bounds = dT_dTinf * I_Tinf + dT_dTw * I_Tw + dT_db * I_b
T_lb = T_nom + bounds.lo
T_ub = T_nom + bounds.hi


# ── Plot ───────────────────────────────────────────────────────────────────────
plt.figure(figsize=(5, 4), dpi=300)
plt.plot(t, T_expectation, 'r',  label=r'Case 1: Uniform MCS ($\mathbb{E}[T]$)')
plt.plot(t, T_tse,         'g--', label=r'Case 2: 1st-order TSE ($\mathbb{E}[T]$)')
plt.fill_between(t, T_lb, T_ub, alpha=0.3, color='b', label='Case 3: Interval bounds')
plt.plot(t, T_nom, 'b--', linewidth=1, label='Case 3: Nominal')
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel(r'$\mathbb{E}[T]$ [K]')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('figures/compare_uniform_vs_interval.png', transparent=True)
plt.close()
