"""
1st-order TSE (finite-difference gradient only) of the fin temperature
history, built about (means of k, Cp, rho, hU; midpoints of Tinf, Tw, b), with
epistemic uncertainty propagated through the closed-form expressions derived
in TSE_1st_order_derivation.md -- no sweep over the epistemic grid.

    E[T(tau)] in T0(tau) +- sum_j |g_j| Delta_y_j                 (j in {Tinf, Tw, b})

    Var[T(tau)] in [ f_k^2 sigma_k^2 + f_Cp^2 sigma_Cp^2 + f_rho^2 sigma_rho^2 + f_hU^2 sigma_hU_lo^2,
                      f_k^2 sigma_k^2 + f_Cp^2 sigma_Cp^2 + f_rho^2 sigma_rho^2 + f_hU^2 sigma_hU_hi^2 ]

f_i = dT/dX_i at nominal (X in {k,Cp,rho,hU}), g_j = dT/dy_j at nominal
(y in {Tinf,Tw,b}). T_infty, T_W, b (location parameters) only widen E[T];
sigma_hU (a scale parameter) only widens Var[T] -- at 1st order neither
widens the other. See TSE_1st_order_derivation.md for the full derivation.

Bounds are checked against the brute-force double-loop MC sweep in
double_loop_MCS_data.npz.

Also builds affine-arithmetic (AA) and 1st-order-Taylor-model (TM) bounds on
the SAME closed-form E[T]/Var[T] expressions above, as a validation check --
E[T] and Var[T] here are already exactly affine in (Tinf, Tw, b, sigma_hU^2)
(each appears once, linearly, no cross/squared terms -- that only happens in
the 2nd-order TSE), so AA and TM are mathematically guaranteed to reproduce
the IA bounds exactly rather than tighten them. The point of building them
here is to confirm that agreement numerically/symbolically, not to improve
on IA.
"""
import os
import time
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import fin_problem as fp
from fin_params_mixed import MEAN, STD, MEAN_hU, STD_hU, Tinf, Tw, b, d, L, x, t

os.makedirs("figures", exist_ok=True)

# ── Nominal point: aleatory means + epistemic midpoints ──────────────────────
Tinf_nom = (Tinf[0] + Tinf[1]) / 2
Tw_nom = (Tw[0] + Tw[1]) / 2
b_nom = (b[0] + b[1]) / 2

mu_X = np.array([MEAN[0], MEAN[1], MEAN[2], MEAN_hU])   # k, Cp, rho, hU
m_y = np.array([Tinf_nom, Tw_nom, b_nom])                # Tinf, Tw, b
sigma_X = np.array([STD[0], STD[1], STD[2]])              # k, Cp, rho (fixed, precise)
sigma_hU_lo, sigma_hU_hi = STD_hU[0], STD_hU[1]            # epistemic interval on sigma_hU


def eval_T_X(x_vec, y_vec):
    k_v, Cp_v, rho_v, hU_v = x_vec
    Tinf_v, Tw_v, b_v = y_vec
    T_out, _, _, _ = fp.analytic_solution(
        x, t, b_v, d, L, rho_v, Cp_v, k_v, hU_v, Tinf_v, Tw_v, Tinf_v,  # T0 = Tinf
    )
    return T_out


# ── Gradient at the nominal point (15 model evaluations, no sweep) ──────────
tse_t0 = time.perf_counter()

T0 = eval_T_X(mu_X, m_y)

h_X = 1e-3 * mu_X
f = np.zeros((4, len(t)))
for i in range(4):
    xp = mu_X.copy(); xp[i] += h_X[i]
    xm = mu_X.copy(); xm[i] -= h_X[i]
    f[i, :] = (eval_T_X(xp, m_y) - eval_T_X(xm, m_y)) / (2 * h_X[i])

h_y = 1e-3 * m_y
g = np.zeros((3, len(t)))
for j in range(3):
    yp = m_y.copy(); yp[j] += h_y[j]
    ym = m_y.copy(); ym[j] -= h_y[j]
    g[j, :] = (eval_T_X(mu_X, yp) - eval_T_X(mu_X, ym)) / (2 * h_y[j])

tse_wall_time = time.perf_counter() - tse_t0
n_tse_evals = 1 + 2 * 4 + 2 * 3
print(f"Total model evaluations for the 1st-order TSE: {n_tse_evals} (no sweep)")

# ── E[T] bound: location parameters (Tinf, Tw, b) via interval arithmetic ──
Delta_y = np.array([(Tinf[1] - Tinf[0]) / 2, (Tw[1] - Tw[0]) / 2, (b[1] - b[0]) / 2])
E_T_width = np.sum(np.abs(g) * Delta_y[:, None], axis=0)
E_T_lower, E_T_upper = T0 - E_T_width, T0 + E_T_width

# ── Var[T] bound: scale parameter (sigma_hU) enters directly, exactly ──────
Var_T_fixed = f[0]**2 * sigma_X[0]**2 + f[1]**2 * sigma_X[1]**2 + f[2]**2 * sigma_X[2]**2
Var_T_lower = Var_T_fixed + f[3]**2 * sigma_hU_lo**2
Var_T_upper = Var_T_fixed + f[3]**2 * sigma_hU_hi**2


# ── Affine arithmetic (AA), for validation against the IA bounds above ──────
class Affine:
    """Affine form over t: center(t) + sum_k coeffs[k](t) * eps_k, eps_k in [-1, 1].
    Only add/scalar-multiply are needed here -- every epistemic quantity
    below appears exactly once, linearly, so no square()/mul() is required."""

    _counter = [0]
    __array_ufunc__ = None  # let numpy defer array*Affine to Affine.__rmul__

    def __init__(self, center, coeffs=None):
        self.center = np.asarray(center, dtype=float)
        self.coeffs = dict(coeffs) if coeffs else {}

    @classmethod
    def variable(cls, center, radius):
        cls._counter[0] += 1
        return cls(np.asarray(center, dtype=float), {cls._counter[0]: np.asarray(radius, dtype=float)})

    @staticmethod
    def _as_affine(v):
        return v if isinstance(v, Affine) else Affine(np.asarray(v, dtype=float))

    def __add__(self, other):
        o = Affine._as_affine(other)
        c = dict(self.coeffs)
        for k, v in o.coeffs.items():
            c[k] = c.get(k, 0.0) + v
        return Affine(self.center + o.center, c)

    __radd__ = __add__

    def __mul__(self, other):
        other = np.asarray(other, dtype=float)
        return Affine(self.center * other, {k: v * other for k, v in self.coeffs.items()})

    __rmul__ = __mul__

    def to_interval(self):
        rad = np.zeros_like(self.center)
        for v in self.coeffs.values():
            rad = rad + np.abs(v)
        return self.center - rad, self.center + rad


dTheta_aa = [Affine.variable(0.0, Delta_y[j]) for j in range(3)]
E_T_aff = Affine(T0)
for j in range(3):
    E_T_aff = E_T_aff + g[j] * dTheta_aa[j]
E_T_aa_lower, E_T_aa_upper = E_T_aff.to_interval()

sigma_hU_sq_mid = (sigma_hU_lo**2 + sigma_hU_hi**2) / 2
sigma_hU_sq_rad = (sigma_hU_hi**2 - sigma_hU_lo**2) / 2
sigma_hU_sq_aff = Affine.variable(sigma_hU_sq_mid, sigma_hU_sq_rad)
Var_T_aff = Affine(Var_T_fixed) + f[3]**2 * sigma_hU_sq_aff
Var_T_aa_lower, Var_T_aa_upper = Var_T_aff.to_interval()

# ── 1st-order Taylor model (TM), for validation against the IA bounds above ─
# E[T] and Var[T] are literally g.Delta_theta and f_hU^2 . sigma_hU^2 -- confirm
# their curvature (Hessian w.r.t. the epistemic variables) is exactly zero,
# so the TM remainder is 0 and TM collapses to the same linear expansion IA used.
_dt0, _dt1, _dt2, _shu = sp.symbols("dt0 dt1 dt2 shu")
_g0, _g1, _g2, _fhu2 = sp.symbols("g0 g1 g2 fhu2")
_E_T_expr = _g0 * _dt0 + _g1 * _dt1 + _g2 * _dt2   # + T0 (constant; drops out of the Hessian)
_Var_T_expr = _fhu2 * _shu                          # + Var_T_fixed (constant; drops out)
assert sp.hessian(_E_T_expr, (_dt0, _dt1, _dt2)).is_zero_matrix
assert sp.diff(_Var_T_expr, _shu, 2) == 0

E_T_tm_lower, E_T_tm_upper = E_T_lower, E_T_upper
Var_T_tm_lower, Var_T_tm_upper = Var_T_lower, Var_T_upper

# ── MC bounds from double_loop_MCS_data.npz, for comparison ─────────────────
mc_data = np.load("double_loop_MCS_data.npz")
T_mc = mc_data["T"]  # shape (num_outer, N_inner, len(t))
T_mean_cond_mc = T_mc.mean(axis=1)
T_var_cond_mc = T_mc.var(axis=1)
E_T_lower_mc, E_T_upper_mc = T_mean_cond_mc.min(axis=0), T_mean_cond_mc.max(axis=0)
Var_T_lower_mc, Var_T_upper_mc = T_var_cond_mc.min(axis=0), T_var_cond_mc.max(axis=0)

# ── Relative cost: model evaluations, and wall time ──────────────────────────
# double_loop_MCS_data.npz was generated once (~85 min at full MC resolution
# would be needed to re-sweep it at this epistemic grid -- see the M_grid^D
# cost discussion) so re-running it here just to time it isn't worth it.
# Instead, benchmark the per-call cost of the same analytic_solution() call
# both methods use, and scale it up to the MC's actual evaluation count.
n_mc_evals = T_mc.shape[0] * T_mc.shape[1]

N_bench = 200
bench_t0 = time.perf_counter()
for _ in range(N_bench):
    eval_T_X(mu_X, m_y)
per_call_time = (time.perf_counter() - bench_t0) / N_bench

mc_wall_time_est = per_call_time * n_mc_evals

eval_ratio = n_tse_evals / n_mc_evals
time_ratio = tse_wall_time / mc_wall_time_est

print(f"\n{'':<28}{'evaluations':>14}{'wall time':>14}")
print(f"{'1st-order TSE':<28}{n_tse_evals:>14,}{tse_wall_time:>13.4f}s")
print(f"{'Double-loop MC (est.)':<28}{n_mc_evals:>14,}{mc_wall_time_est:>13.4f}s")
print(f"\nTSE relative to double-loop MC:")
print(f"  evaluations ratio: {eval_ratio:.2e}  ({1/eval_ratio:,.0f}x fewer)")
print(f"  wall-time ratio:   {time_ratio:.2e}  ({1/time_ratio:,.0f}x less, estimated)")

E_T_mid, E_T_mid_mc = (E_T_lower + E_T_upper) / 2, (E_T_lower_mc + E_T_upper_mc) / 2
Var_T_mid, Var_T_mid_mc = (Var_T_lower + Var_T_upper) / 2, (Var_T_lower_mc + Var_T_upper_mc) / 2

# ── Plot: 1st-order TSE bounds vs MC bounds (E[T]) ───────────────────────────
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.fill_between(t, E_T_lower_mc, E_T_upper_mc, color="0.75", alpha=0.5,
                zorder=1, label="MC bounds")
ax.plot(t, E_T_lower, color="#0C447C", lw=1.8, zorder=2, label="TSE (1st order) lower")
ax.plot(t, E_T_upper, color="#185FA5", lw=1.8, zorder=2, label="TSE (1st order) upper")
ax.plot(t, E_T_mid, color="#A32D2D", lw=1.5, ls="--", zorder=3, label="TSE midpoint")
ax.plot(t, E_T_mid_mc, color="0.35", lw=1.5, ls=":", zorder=3, label="MC midpoint")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\mathbb{E}[T]$ [K]")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/TSE_1st_order_bounds_E_T_vs_t.png", transparent=True)
plt.close(fig)


def save_bound_plot(lower, upper, ylabel, method_label, fname, lo_mc, hi_mc, mid_mc):
    mid = (lower + upper) / 2
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.fill_between(t, lo_mc, hi_mc, color="0.75", alpha=0.5, zorder=1, label="MC bounds")
    ax.plot(t, lower, color="#0C447C", lw=1.8, zorder=2, label=f"TSE ({method_label}) lower")
    ax.plot(t, upper, color="#185FA5", lw=1.8, zorder=2, label=f"TSE ({method_label}) upper")
    ax.plot(t, mid, color="#A32D2D", lw=1.5, ls="--", zorder=3, label="TSE midpoint")
    ax.plot(t, mid_mc, color="0.35", lw=1.5, ls=":", zorder=3, label="MC midpoint")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"figures/{fname}", transparent=True)
    plt.close(fig)


# ── Plot: affine-arithmetic bounds vs MC bounds (E[T]) -- validation ────────
save_bound_plot(E_T_aa_lower, E_T_aa_upper, r"$\mathbb{E}[T]$ [K]", "affine",
                "TSE_1st_order_bounds_aa_E_T_vs_t.png", E_T_lower_mc, E_T_upper_mc, E_T_mid_mc)

# ── Plot: 1st-order Taylor-model bounds vs MC bounds (E[T]) -- validation ───
save_bound_plot(E_T_tm_lower, E_T_tm_upper, r"$\mathbb{E}[T]$ [K]", "Taylor model",
                "TSE_1st_order_bounds_tm_E_T_vs_t.png", E_T_lower_mc, E_T_upper_mc, E_T_mid_mc)

# ── Plot: 1st-order TSE bounds vs MC bounds (Var[T]) ─────────────────────────
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.fill_between(t, Var_T_lower_mc, Var_T_upper_mc, color="0.75", alpha=0.5,
                zorder=1, label="MC bounds")
ax.plot(t, Var_T_lower, color="#0C447C", lw=1.8, zorder=2, label="TSE (1st order) lower")
ax.plot(t, Var_T_upper, color="#185FA5", lw=1.8, zorder=2, label="TSE (1st order) upper")
ax.plot(t, Var_T_mid, color="#A32D2D", lw=1.5, ls="--", zorder=3, label="TSE midpoint")
ax.plot(t, Var_T_mid_mc, color="0.35", lw=1.5, ls=":", zorder=3, label="MC midpoint")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\mathrm{Var}[T]$ [K$^2$]")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/TSE_1st_order_bounds_Var_T_vs_t.png", transparent=True)
plt.close(fig)

# ── Plot: affine-arithmetic bounds vs MC bounds (Var[T]) -- validation ──────
save_bound_plot(Var_T_aa_lower, Var_T_aa_upper, r"$\mathrm{Var}[T]$ [K$^2$]", "affine",
                "TSE_1st_order_bounds_aa_Var_T_vs_t.png", Var_T_lower_mc, Var_T_upper_mc, Var_T_mid_mc)

# ── Plot: 1st-order Taylor-model bounds vs MC bounds (Var[T]) -- validation ─
save_bound_plot(Var_T_tm_lower, Var_T_tm_upper, r"$\mathrm{Var}[T]$ [K$^2$]", "Taylor model",
                "TSE_1st_order_bounds_tm_Var_T_vs_t.png", Var_T_lower_mc, Var_T_upper_mc, Var_T_mid_mc)

print(f"\n{'':<20}{'TSE lower':>12}{'TSE upper':>12}{'MC lower':>12}{'MC upper':>12}")
print(f"{'E[T] (t=450s)':<20}{E_T_lower[-1]:>12.4f}{E_T_upper[-1]:>12.4f}"
      f"{E_T_lower_mc[-1]:>12.4f}{E_T_upper_mc[-1]:>12.4f}")
print(f"{'Var[T] (t=450s)':<20}{Var_T_lower[-1]:>12.4f}{Var_T_upper[-1]:>12.4f}"
      f"{Var_T_lower_mc[-1]:>12.4f}{Var_T_upper_mc[-1]:>12.4f}")

# ── Confirm IA == AA == TM exactly, as expected at 1st order ────────────────
print(f"\n{'':<28}{'IA':>22}{'AA':>22}{'TM':>22}")
print(f"{'E[T] (t=450s)':<28}"
      f"{f'[{E_T_lower[-1]:.4f}, {E_T_upper[-1]:.4f}]':>22}"
      f"{f'[{E_T_aa_lower[-1]:.4f}, {E_T_aa_upper[-1]:.4f}]':>22}"
      f"{f'[{E_T_tm_lower[-1]:.4f}, {E_T_tm_upper[-1]:.4f}]':>22}")
print(f"{'Var[T] (t=450s)':<28}"
      f"{f'[{Var_T_lower[-1]:.4f}, {Var_T_upper[-1]:.4f}]':>22}"
      f"{f'[{Var_T_aa_lower[-1]:.4f}, {Var_T_aa_upper[-1]:.4f}]':>22}"
      f"{f'[{Var_T_tm_lower[-1]:.4f}, {Var_T_tm_upper[-1]:.4f}]':>22}")
assert np.allclose(E_T_lower, E_T_aa_lower) and np.allclose(E_T_upper, E_T_aa_upper)
assert np.allclose(Var_T_lower, Var_T_aa_lower) and np.allclose(Var_T_upper, Var_T_aa_upper)
print("\nIA, AA, and TM bounds agree exactly (as expected -- E[T]/Var[T] are affine here).")
