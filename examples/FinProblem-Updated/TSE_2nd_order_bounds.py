"""
2nd-order TSE (finite-difference gradient + Hessian) of the fin temperature
history, built ONCE about (means of k, Cp, rho, hU; midpoints of Tinf, Tw, b),
with the epistemic uncertainty (Tinf, Tw, b, STD_hU) propagated through that
single expansion via interval arithmetic -- no re-evaluation of the model
outside this one-time derivative computation.

Implements E[T](Delta_theta) from TSE_2nd_order_derivation.md:

    E_x[T](Delta_theta) = [T0 + 0.5 sum_i H_ii sigma_i^2]
                           + sum_j g_j Delta_theta_j
                           + 0.5 sum_jj' G_jj' Delta_theta_j Delta_theta_j'

where i ranges over the aleatory indices {k, Cp, rho, hU} (sigma_hU^2 an
epistemic interval -- Case 3) and j, j' range over the epistemic indices
{Tinf, Tw, b} (Case 1, including the one-sided Delta_theta_j^2 square).

Also implements Var[T] from TSE_2nd_order_derivation.md's boxed Variance
result:

    Var_X[T](Delta_theta) = sum_i ( f_i + sum_j C_ij Delta_theta_j )^2 sigma_i^2
                             + 0.5 sum_i sum_i' H_ii'^2 sigma_i^2 sigma_i'^2

where f_i = dT/dX_i, C_ij = d^2T/dX_i dy_j (mixed aleatory-epistemic
Hessian), H_ii' = d^2T/dX_i dX_i' (aleatory-aleatory Hessian), all at the
nominal point; sigma_i^2 is the epistemic interval [sigma_hU_lo^2,
sigma_hU_hi^2] at i=hU (Case 3) and crisp otherwise. The bracket is squared
with the dedicated one-sided Interval.square() (it's a single affine
combination of independent Delta_theta_j's, so this is exact); sigma_hU^4 is
likewise squared via Interval.square() rather than Interval*Interval, to
avoid the repeated-variable dependency problem.

Both E[T] and Var[T] bounds are checked against two references: the
brute-force double-loop MC sweep in double_loop_MCS_data.npz, and the
double-loop TSE hybrid (1st-order-in-X gradient re-evaluated at every outer
epistemic grid point, see double_loop_TSE_1st_order.py) in
double_loop_TSE_1st_order_data.npz.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import fin_problem as fp
from fin_params_mixed import MEAN, STD, MEAN_hU, STD_hU, Tinf, Tw, b, d, L, x, t

os.makedirs("figures", exist_ok=True)


# ── Interval arithmetic ──────────────────────────────────────────────────────
class Interval:
    """Closed interval [lo, hi]; lo/hi may be scalars or arrays (e.g. over t)."""

    __array_ufunc__ = None  # let numpy defer array*Interval to Interval.__rmul__

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    @staticmethod
    def _as_interval(v):
        return v if isinstance(v, Interval) else Interval(v, v)

    def __add__(self, other):
        o = Interval._as_interval(other)
        return Interval(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __mul__(self, other):
        o = Interval._as_interval(other)
        products = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Interval(np.minimum.reduce(products), np.maximum.reduce(products))

    __rmul__ = __mul__

    def square(self):
        """Tight range of x^2 for x in [lo, hi] -- Case 1's one-sided square
        (avoids the naive-multiply dependency-problem blowup of treating x*x
        as two independent copies)."""
        lo2, hi2 = self.lo**2, self.hi**2
        contains_zero = (self.lo <= 0) & (self.hi >= 0)
        lower = np.where(contains_zero, 0.0, np.minimum(lo2, hi2))
        upper = np.maximum(lo2, hi2)
        return Interval(lower, upper)


# ── Nominal point: aleatory means + epistemic midpoints ──────────────────────
Tinf_nom = (Tinf[0] + Tinf[1]) / 2
Tw_nom = (Tw[0] + Tw[1]) / 2
b_nom = (b[0] + b[1]) / 2

# Combined variable vector z = [k, Cp, rho, hU, Tinf, Tw, b]
z0 = np.array([MEAN[0], MEAN[1], MEAN[2], MEAN_hU, Tinf_nom, Tw_nom, b_nom])
n = len(z0)
A = slice(0, 4)   # aleatory indices: k, Cp, rho, hU
B = slice(4, 7)   # epistemic indices: Tinf, Tw, b
hz = 1e-3 * z0    # finite-difference step sizes (relative)


def eval_T_z(z):
    k_v, Cp_v, rho_v, hU_v, Tinf_v, Tw_v, b_v = z
    T_out, _, _, _ = fp.analytic_solution(
        x, t, b_v, d, L, rho_v, Cp_v, k_v, hU_v, Tinf_v, Tw_v, Tinf_v,  # T0 = Tinf
    )
    return T_out


# ── Gradient + Hessian of T(t), computed ONCE at z0 (no sweep) ──────────────
tse_t0 = time.perf_counter()

T_z0 = eval_T_z(z0)

Zp = np.zeros((n, len(t)))
Zm = np.zeros((n, len(t)))
for i in range(n):
    zp = z0.copy(); zp[i] += hz[i]
    zm = z0.copy(); zm[i] -= hz[i]
    Zp[i, :] = eval_T_z(zp)
    Zm[i, :] = eval_T_z(zm)

grad = (Zp - Zm) / (2 * hz[:, None])  # shape (n, len(t))

Hess = np.zeros((n, n, len(t)))
for i in range(n):
    Hess[i, i, :] = (Zp[i, :] - 2 * T_z0 + Zm[i, :]) / hz[i]**2

for i in range(n):
    for j in range(i + 1, n):
        zpp = z0.copy(); zpp[i] += hz[i]; zpp[j] += hz[j]
        zpm = z0.copy(); zpm[i] += hz[i]; zpm[j] -= hz[j]
        zmp = z0.copy(); zmp[i] -= hz[i]; zmp[j] += hz[j]
        zmm = z0.copy(); zmm[i] -= hz[i]; zmm[j] -= hz[j]
        Hij = (eval_T_z(zpp) - eval_T_z(zpm) - eval_T_z(zmp) + eval_T_z(zmm)) / (4 * hz[i] * hz[j])
        Hess[i, j, :] = Hij
        Hess[j, i, :] = Hij

tse_wall_time = time.perf_counter() - tse_t0

n_evals = 1 + 2 * n + 4 * n * (n - 1) // 2
print(f"Total model evaluations for the combined 2nd-order TSE: {n_evals} (no sweep)")

g = grad[B]         # epistemic gradient, (3, len(t))
G_BB = Hess[B, B]   # epistemic-epistemic Hessian, (3, 3, len(t))

# ── Epistemic deviations as intervals (Case 1: Tinf, Tw, b) ─────────────────
dTheta = [
    Interval(-(Tinf[1] - Tinf[0]) / 2, (Tinf[1] - Tinf[0]) / 2),
    Interval(-(Tw[1] - Tw[0]) / 2, (Tw[1] - Tw[0]) / 2),
    Interval(-(b[1] - b[0]) / 2, (b[1] - b[0]) / 2),
]

# sigma_hU^2 (Case 3): substituted directly as its interval, exact, no split
sigma_hU_sq = Interval(STD_hU[0]**2, STD_hU[1]**2)
sigma_A_sq = [STD[0]**2, STD[1]**2, STD[2]**2, sigma_hU_sq]  # k, Cp, rho fixed; hU interval

# ── E_x[T](Delta_theta): constant + linear + quadratic in Delta_theta ───────
E_T_interval = Interval(T_z0, T_z0)
for i in range(4):
    E_T_interval = E_T_interval + 0.5 * Hess[i, i] * sigma_A_sq[i]

for j in range(3):
    E_T_interval = E_T_interval + g[j] * dTheta[j]

for i in range(3):
    for j in range(3):
        cross = dTheta[i].square() if i == j else (dTheta[i] * dTheta[j])
        E_T_interval = E_T_interval + 0.5 * G_BB[i, j] * cross

E_T_lower, E_T_upper = E_T_interval.lo, E_T_interval.hi

# ── Var_X[T](Delta_theta): bracket^2 term + aleatory-aleatory Hessian term ──
f = grad[A]      # aleatory gradient, (4, len(t))
C = Hess[A, B]    # mixed aleatory-epistemic Hessian, (4, 3, len(t))
H_AA = Hess[A, A]  # aleatory-aleatory Hessian, (4, 4, len(t))


def sq_sigma_product(i, ip):
    """sigma_i^2 * sigma_i'^2, using the tight one-sided square when i==i'
    and sigma_i^2 is itself an epistemic interval (avoids treating the two
    occurrences of sigma_hU^2 as independent)."""
    si, sip = sigma_A_sq[i], sigma_A_sq[ip]
    if i == ip:
        return si.square() if isinstance(si, Interval) else si**2
    return si * sip


Var_T_interval = Interval(np.zeros_like(t), np.zeros_like(t))
for i in range(4):
    bracket = Interval(f[i], f[i])
    for j in range(3):
        bracket = bracket + C[i, j] * dTheta[j]
    Var_T_interval = Var_T_interval + bracket.square() * sigma_A_sq[i]

for i in range(4):
    for ip in range(4):
        Var_T_interval = Var_T_interval + 0.5 * (H_AA[i, ip]**2) * sq_sigma_product(i, ip)

Var_T_lower, Var_T_upper = Var_T_interval.lo, Var_T_interval.hi

# ── MC bounds from double_loop_MCS_data.npz, for comparison only ────────────
mc_data = np.load("double_loop_MCS_data.npz")
T_mc = mc_data["T"]  # shape (num_outer, N_inner, len(t)), from the brute-force sweep
T_mean_cond_mc = T_mc.mean(axis=1)
T_var_cond_mc = T_mc.var(axis=1)
E_T_lower_mc, E_T_upper_mc = T_mean_cond_mc.min(axis=0), T_mean_cond_mc.max(axis=0)
Var_T_lower_mc, Var_T_upper_mc = T_var_cond_mc.min(axis=0), T_var_cond_mc.max(axis=0)

E_T_mid, E_T_mid_mc = (E_T_lower + E_T_upper) / 2, (E_T_lower_mc + E_T_upper_mc) / 2
Var_T_mid, Var_T_mid_mc = (Var_T_lower + Var_T_upper) / 2, (Var_T_lower_mc + Var_T_upper_mc) / 2

# ── Plot: interval-arithmetic TSE bounds vs MC bounds (E[T]) ────────────────
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.fill_between(t, E_T_lower_mc, E_T_upper_mc, color="0.75", alpha=0.5,
                zorder=1, label="MC bounds")
ax.plot(t, E_T_lower, color="#0C447C", lw=1.8, zorder=2, label="TSE (2nd order) lower")
ax.plot(t, E_T_upper, color="#185FA5", lw=1.8, zorder=2, label="TSE (2nd order) upper")
ax.plot(t, E_T_mid, color="#A32D2D", lw=1.5, ls="--", zorder=3, label="TSE midpoint")
ax.plot(t, E_T_mid_mc, color="0.35", lw=1.5, ls=":", zorder=3, label="MC midpoint")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\mathbb{E}[T]$ [K]")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/TSE_2nd_order_bounds_E_T_vs_t.png", transparent=True)
plt.close(fig)

print(f"\n{'':<20}{'TSE lower':>12}{'TSE upper':>12}{'MC lower':>12}{'MC upper':>12}")
print(f"{'E[T] (t=450s)':<20}{E_T_lower[-1]:>12.4f}{E_T_upper[-1]:>12.4f}"
      f"{E_T_lower_mc[-1]:>12.4f}{E_T_upper_mc[-1]:>12.4f}")

# ── Relative error of the TSE lower/midpoint/upper vs MC, at t=450s ─────────
rel_err_lower = abs(E_T_lower[-1] - E_T_lower_mc[-1]) / abs(E_T_lower_mc[-1]) * 100
rel_err_mid = abs(E_T_mid[-1] - E_T_mid_mc[-1]) / abs(E_T_mid_mc[-1]) * 100
rel_err_upper = abs(E_T_upper[-1] - E_T_upper_mc[-1]) / abs(E_T_upper_mc[-1]) * 100

print(f"\n{'':<12}{'TSE':>12}{'MC':>12}{'rel. error':>14}")
print(f"{'Lower':<12}{E_T_lower[-1]:>12.4f}{E_T_lower_mc[-1]:>12.4f}{rel_err_lower:>13.4f}%")
print(f"{'Midpoint':<12}{E_T_mid[-1]:>12.4f}{E_T_mid_mc[-1]:>12.4f}{rel_err_mid:>13.4f}%")
print(f"{'Upper':<12}{E_T_upper[-1]:>12.4f}{E_T_upper_mc[-1]:>12.4f}{rel_err_upper:>13.4f}%")

# ── Plot: interval-arithmetic TSE bounds vs MC bounds (Var[T]) ──────────────
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.fill_between(t, Var_T_lower_mc, Var_T_upper_mc, color="0.75", alpha=0.5,
                zorder=1, label="MC bounds")
ax.plot(t, Var_T_lower, color="#0C447C", lw=1.8, zorder=2, label="TSE (2nd order) lower")
ax.plot(t, Var_T_upper, color="#185FA5", lw=1.8, zorder=2, label="TSE (2nd order) upper")
ax.plot(t, Var_T_mid, color="#A32D2D", lw=1.5, ls="--", zorder=3, label="TSE midpoint")
ax.plot(t, Var_T_mid_mc, color="0.35", lw=1.5, ls=":", zorder=3, label="MC midpoint")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\mathrm{Var}[T]$ [K$^2$]")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/TSE_2nd_order_bounds_Var_T_vs_t.png", transparent=True)
plt.close(fig)

print(f"\n{'':<20}{'TSE lower':>12}{'TSE upper':>12}{'MC lower':>12}{'MC upper':>12}")
print(f"{'Var[T] (t=450s)':<20}{Var_T_lower[-1]:>12.4f}{Var_T_upper[-1]:>12.4f}"
      f"{Var_T_lower_mc[-1]:>12.4f}{Var_T_upper_mc[-1]:>12.4f}")

# ── Relative error of the TSE lower/midpoint/upper vs MC, at t=450s ─────────
var_rel_err_lower = abs(Var_T_lower[-1] - Var_T_lower_mc[-1]) / abs(Var_T_lower_mc[-1]) * 100
var_rel_err_mid = abs(Var_T_mid[-1] - Var_T_mid_mc[-1]) / abs(Var_T_mid_mc[-1]) * 100
var_rel_err_upper = abs(Var_T_upper[-1] - Var_T_upper_mc[-1]) / abs(Var_T_upper_mc[-1]) * 100

print(f"\n{'':<12}{'TSE':>12}{'MC':>12}{'rel. error':>14}")
print(f"{'Lower':<12}{Var_T_lower[-1]:>12.4f}{Var_T_lower_mc[-1]:>12.4f}{var_rel_err_lower:>13.4f}%")
print(f"{'Midpoint':<12}{Var_T_mid[-1]:>12.4f}{Var_T_mid_mc[-1]:>12.4f}{var_rel_err_mid:>13.4f}%")
print(f"{'Upper':<12}{Var_T_upper[-1]:>12.4f}{Var_T_upper_mc[-1]:>12.4f}{var_rel_err_upper:>13.4f}%")

# ── DL-TSE bounds from double_loop_TSE_1st_order_data.npz, for comparison ───
# The double-loop TSE hybrid (double_loop_TSE_1st_order.py) re-evaluates the
# 1st-order aleatory gradient f_i(y) exactly at every outer epistemic grid
# point instead of linearizing its y-dependence -- i.e. it implicitly
# captures the full C_ij mixed-Hessian coupling this 2nd-order expansion
# only linearizes. Comparing against it (rather than only the brute-force
# double-loop MC) isolates how much of that coupling the mixed Hessian
# term recovers, independent of the remaining aleatory-truncation gap that
# also separates both TSE methods from the MC ground truth.
dlt_data = np.load("double_loop_TSE_1st_order_data.npz")
T_mean_cond_dlt = dlt_data["T_mean_cond"]
T_var_cond_dlt = dlt_data["T_var_cond"]
n_dlt_evals = int(dlt_data["n_tse_evals"])
E_T_lower_dlt, E_T_upper_dlt = T_mean_cond_dlt.min(axis=0), T_mean_cond_dlt.max(axis=0)
Var_T_lower_dlt, Var_T_upper_dlt = T_var_cond_dlt.min(axis=0), T_var_cond_dlt.max(axis=0)
E_T_mid_dlt = (E_T_lower_dlt + E_T_upper_dlt) / 2
Var_T_mid_dlt = (Var_T_lower_dlt + Var_T_upper_dlt) / 2

print(f"\nCombined 2nd-order TSE: {n_evals} evaluations (no sweep) "
      f"vs. double-loop TSE: {n_dlt_evals:,} evaluations ({dlt_data['EE'].shape[0]:,} outer points)")


def save_dlt_comparison_plot(lower, upper, mid, lower_dlt, upper_dlt, mid_dlt, ylabel, fname):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.fill_between(t, lower_dlt, upper_dlt, color="0.75", alpha=0.5,
                     zorder=1, label="Double loop TSE bounds")
    ax.plot(t, lower, color="#0C447C", lw=1.8, zorder=2, label="TSE (2nd order) lower")
    ax.plot(t, upper, color="#185FA5", lw=1.8, zorder=2, label="TSE (2nd order) upper")
    ax.plot(t, mid, color="#A32D2D", lw=1.5, ls="--", zorder=3, label="TSE midpoint")
    ax.plot(t, mid_dlt, color="0.35", lw=1.5, ls=":", zorder=3, label="Double loop TSE midpoint")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"figures/{fname}", transparent=True)
    plt.close(fig)


# ── Plot: 2nd-order TSE bounds vs double-loop TSE bounds (E[T]) ─────────────
save_dlt_comparison_plot(E_T_lower, E_T_upper, E_T_mid, E_T_lower_dlt, E_T_upper_dlt, E_T_mid_dlt,
                          r"$\mathbb{E}[T]$ [K]", "TSE_2nd_order_bounds_vs_double_loop_TSE_E_T_vs_t.png")

# ── Plot: 2nd-order TSE bounds vs double-loop TSE bounds (Var[T]) ───────────
save_dlt_comparison_plot(Var_T_lower, Var_T_upper, Var_T_mid, Var_T_lower_dlt, Var_T_upper_dlt, Var_T_mid_dlt,
                          r"$\mathrm{Var}[T]$ [K$^2$]", "TSE_2nd_order_bounds_vs_double_loop_TSE_Var_T_vs_t.png")

print(f"\n{'':<20}{'IA lower':>12}{'IA upper':>12}{'DL-TSE lower':>14}{'DL-TSE upper':>14}")
print(f"{'E[T] (t=450s)':<20}{E_T_lower[-1]:>12.4f}{E_T_upper[-1]:>12.4f}"
      f"{E_T_lower_dlt[-1]:>14.4f}{E_T_upper_dlt[-1]:>14.4f}")
print(f"{'Var[T] (t=450s)':<20}{Var_T_lower[-1]:>12.4f}{Var_T_upper[-1]:>12.4f}"
      f"{Var_T_lower_dlt[-1]:>14.4f}{Var_T_upper_dlt[-1]:>14.4f}")

# ── Runtime: TSE (measured) vs double-loop MC (estimated) ──────────────────
# double_loop_MCS_data.npz was generated once (~90s at this resolution), so
# re-running it here just to time it isn't worth it -- instead benchmark the
# per-call cost of the same analytic_solution() call both methods use, and
# scale it up to the MC's actual evaluation count.
n_mc_evals = T_mc.shape[0] * T_mc.shape[1]

N_bench = 200
bench_t0 = time.perf_counter()
for _ in range(N_bench):
    eval_T_z(z0)
per_call_time = (time.perf_counter() - bench_t0) / N_bench
mc_wall_time_est = per_call_time * n_mc_evals

eval_ratio = n_evals / n_mc_evals
time_ratio = tse_wall_time / mc_wall_time_est

print(f"\n{'':<28}{'evaluations':>14}{'wall time':>14}")
print(f"{'2nd-order TSE':<28}{n_evals:>14,}{tse_wall_time:>13.4f}s")
print(f"{'Double-loop MC (est.)':<28}{n_mc_evals:>14,}{mc_wall_time_est:>13.4f}s")
print(f"\nTSE relative to double-loop MC:")
print(f"  evaluations ratio: {eval_ratio:.2e}  ({1/eval_ratio:,.0f}x fewer)")
print(f"  wall-time ratio:   {time_ratio*100:.4f}%  ({1/time_ratio:,.0f}x less, estimated)")
