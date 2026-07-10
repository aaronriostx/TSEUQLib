"""
2nd-order TSE (finite-difference gradient + Hessian) of the fin temperature
history, built ONCE about (means of k, Cp, rho, hU; midpoints of Tinf, Tw, b),
with the epistemic uncertainty (Tinf, Tw, b, STD_hU) propagated through that
single expansion via interval arithmetic -- no re-evaluation of the model
outside this one-time derivative computation.

Split variables into aleatory A = {k, Cp, rho, hU} and epistemic B =
{Tinf, Tw, b}. Writing Delta_x for aleatory deviations and Delta_theta for
epistemic deviations about the nominal point, and taking the expectation
E_x[.] over the aleatory randomness only (Delta_theta left symbolic):

    E_x[T](Delta_theta) = [T0 + 0.5 sum_i H_ii sigma_i^2]
                           + sum_i g_i Delta_theta_i
                           + 0.5 sum_ij G_ij Delta_theta_i Delta_theta_j

    Var_x[T](Delta_theta) = sum_i [f_i + sum_j C_ij Delta_theta_j]^2 sigma_i^2
                             + 0.5 sum_ij H_ij^2 sigma_i^2 sigma_j^2

where f_i, H_ij are the aleatory-aleatory gradient/Hessian (as in
TSE_2nd_order.py), g_i, G_ij are the epistemic-epistemic gradient/Hessian, and
C_ij is the mixed aleatory-epistemic Hessian block. STD_hU folds in directly
as an interval on sigma_hU^2 (it never appears as a model argument).

These are now closed-form polynomials purely in the epistemic deviations
Delta_theta and sigma_hU^2 -- interval arithmetic is applied to THOSE, not to
sweeps of the physics model.

Caveat: plain interval arithmetic is exact for affine combinations of
independent intervals (the linear terms, and each squared linear bracket
above), but summing several such exact-per-term quantities that share the
same Delta_theta_i can still overestimate the true joint range (the
"dependency problem"/wrapping effect) -- so these bounds are a valid but
possibly conservative outer enclosure, not provably tight. Affine arithmetic
or Taylor models would tighten this further.
"""
import os
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
        """Tight range of x^2 for x in [lo, hi] (avoids the naive-multiply
        dependency-problem blowup of treating x*x as two independent copies)."""
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

n_evals = 1 + 2 * n + 4 * n * (n - 1) // 2
print(f"Total model evaluations for the combined 2nd-order TSE: {n_evals} (no sweep)")

f = grad[A]         # aleatory gradient, (4, len(t))
g = grad[B]         # epistemic gradient, (3, len(t))
H_AA = Hess[A, A]   # aleatory-aleatory Hessian, (4, 4, len(t))
G_BB = Hess[B, B]   # epistemic-epistemic Hessian, (3, 3, len(t))
C_AB = Hess[A, B]   # mixed Hessian, (4, 3, len(t))

# ── Epistemic deviations as intervals (about the nominal point) ─────────────
dTheta = [
    Interval(-(Tinf[1] - Tinf[0]) / 2, (Tinf[1] - Tinf[0]) / 2),
    Interval(-(Tw[1] - Tw[0]) / 2, (Tw[1] - Tw[0]) / 2),
    Interval(-(b[1] - b[0]) / 2, (b[1] - b[0]) / 2),
]

sigma_hU_sq = Interval(STD_hU[0]**2, STD_hU[1]**2)  # STD_hU is itself epistemic
sigma_A_sq = [STD[0]**2, STD[1]**2, STD[2]**2, sigma_hU_sq]  # k, Cp, rho fixed; hU interval

# ── E_x[T](Delta_theta): constant + linear + quadratic in Delta_theta ───────
E_T_interval = Interval(T_z0, T_z0)
for i in range(4):
    E_T_interval = E_T_interval + 0.5 * H_AA[i, i] * sigma_A_sq[i]

for j in range(3):
    E_T_interval = E_T_interval + g[j] * dTheta[j]

for i in range(3):
    for j in range(3):
        cross = dTheta[i].square() if i == j else (dTheta[i] * dTheta[j])
        E_T_interval = E_T_interval + 0.5 * G_BB[i, j] * cross

# ── Var_x[T](Delta_theta): sum_i bracket_i^2 sigma_i^2 + aleatory quad term ──
brackets = []
for i in range(4):
    br = Interval(f[i], f[i])
    for j in range(3):
        br = br + C_AB[i, j] * dTheta[j]
    brackets.append(br)

Var_T_interval = Interval(np.zeros_like(T_z0), np.zeros_like(T_z0))
for i in range(4):
    Var_T_interval = Var_T_interval + brackets[i].square() * sigma_A_sq[i]

for i in range(4):
    for j in range(4):
        Var_T_interval = Var_T_interval + 0.5 * H_AA[i, j]**2 * sigma_A_sq[i] * sigma_A_sq[j]

E_T_lower, E_T_upper = E_T_interval.lo, E_T_interval.hi
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
ax.plot(t, E_T_lower, color="#0C447C", lw=1.8, zorder=2, label="TSE (interval) lower")
ax.plot(t, E_T_upper, color="#185FA5", lw=1.8, zorder=2, label="TSE (interval) upper")
ax.plot(t, E_T_mid, color="#A32D2D", lw=1.5, ls="--", zorder=3, label="TSE midpoint")
ax.plot(t, E_T_mid_mc, color="0.35", lw=1.5, ls=":", zorder=3, label="MC midpoint")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\mathbb{E}[T]$ [K]")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/TSE_2nd_order_bounds_E_T_vs_t.png", transparent=True)
plt.close(fig)

# ── Plot: interval-arithmetic TSE bounds vs MC bounds (Var[T]) ──────────────
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.fill_between(t, Var_T_lower_mc, Var_T_upper_mc, color="0.75", alpha=0.5,
                zorder=1, label="MC bounds")
ax.plot(t, Var_T_lower, color="#0C447C", lw=1.8, zorder=2, label="TSE (interval) lower")
ax.plot(t, Var_T_upper, color="#185FA5", lw=1.8, zorder=2, label="TSE (interval) upper")
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
print(f"{'E[T] (t=450s)':<20}{E_T_lower[-1]:>12.4f}{E_T_upper[-1]:>12.4f}"
      f"{E_T_lower_mc[-1]:>12.4f}{E_T_upper_mc[-1]:>12.4f}")
print(f"{'Var[T] (t=450s)':<20}{Var_T_lower[-1]:>12.4f}{Var_T_upper[-1]:>12.4f}"
      f"{Var_T_lower_mc[-1]:>12.4f}{Var_T_upper_mc[-1]:>12.4f}")
