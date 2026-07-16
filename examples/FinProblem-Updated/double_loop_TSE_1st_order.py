"""
Double-loop UQ for the thermal fin, with the inner (aleatory) loop replaced
by a 1st-order Taylor series expansion instead of Monte Carlo sampling.

Structure mirrors double_loop_MCS_generate_data.py exactly on the outer
loop: the same M_grid^4 epistemic grid over (Tinf, Tw, b, sigma_hU). At each
outer point, instead of drawing N_inner=1000 aleatory samples of
(k, Cp, rho, hU) and running the physics model on each, a 1st-order gradient
of T w.r.t. (k, Cp, rho, hU) is computed at the aleatory means (9 model
evaluations via central differences), and the delta-method formulas give the
conditional moments directly:

    E[T | epistemic pt]   ~= T0
    Var[T | epistemic pt] ~= sum_i f_i^2 sigma_i^2      (sigma_hU = this
                                                           outer point's value)

Taking min/max of these conditional moments across the outer grid gives the
same kind of envelope on E[T](t)/Var[T](t) as double_loop_MCS_post_process.py
computes from the brute-force inner samples -- but at 9 evals/outer-point
instead of 1000.

This differs from TSE_1st_order_bounds.py, which builds a single gradient at
the epistemic midpoint and propagates the whole epistemic box through it
algebraically (interval arithmetic, no sweep). Here the outer loop is still
swept explicitly, so this is a direct apples-to-apples replacement of only
the inner MC loop, checked against the same outer grid as the ground-truth
double_loop_MCS_data.npz.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import fin_problem as fp
from fin_params_mixed import MEAN, STD, MEAN_hU, STD_hU, Tinf, Tw, b, d, L, x, t

os.makedirs("figures", exist_ok=True)


def eval_T_X(x_vec, y_vec):
    k_v, Cp_v, rho_v, hU_v = x_vec
    Tinf_v, Tw_v, b_v = y_vec
    T_out, _, _, _ = fp.analytic_solution(
        x, t, b_v, d, L, rho_v, Cp_v, k_v, hU_v, Tinf_v, Tw_v, Tinf_v,  # T0 = Tinf
    )
    return T_out


# ── Outer loop: same epistemic grid as double_loop_MCS_generate_data.py ─────
M_grid = 10
Tinf_vals = np.linspace(Tinf[0], Tinf[1], M_grid)
Tw_vals = np.linspace(Tw[0], Tw[1], M_grid)
b_vals = np.linspace(b[0], b[1], M_grid)
STD_hU_vals = np.linspace(STD_hU[0], STD_hU[1], M_grid)

EE = [(Tinf_v, Tw_v, b_v, stdhU_v)
      for Tinf_v in Tinf_vals
      for Tw_v in Tw_vals
      for b_v in b_vals
      for stdhU_v in STD_hU_vals]
num_outer = len(EE)

# ── Inner "loop": 1st-order TSE (gradient) at the aleatory means ────────────
mu_X = np.array([MEAN[0], MEAN[1], MEAN[2], MEAN_hU])   # k, Cp, rho, hU
h_X = 1e-3 * mu_X                                          # fixed relative step

T_mean_cond = np.zeros((num_outer, len(t)))
T_var_cond = np.zeros((num_outer, len(t)))

tse_t0 = time.perf_counter()
for m, (Tinf_v, Tw_v, b_v, stdhU_v) in enumerate(EE):
    y_vec = np.array([Tinf_v, Tw_v, b_v])

    T0 = eval_T_X(mu_X, y_vec)

    f = np.zeros((4, len(t)))
    for i in range(4):
        xp = mu_X.copy(); xp[i] += h_X[i]
        xm = mu_X.copy(); xm[i] -= h_X[i]
        f[i, :] = (eval_T_X(xp, y_vec) - eval_T_X(xm, y_vec)) / (2 * h_X[i])

    sigma_X_local = np.array([STD[0], STD[1], STD[2], stdhU_v])
    T_mean_cond[m, :] = T0
    T_var_cond[m, :] = np.sum((f**2) * sigma_X_local[:, None]**2, axis=0)

tse_wall_time = time.perf_counter() - tse_t0
n_tse_evals = num_outer * (1 + 2 * 4)
n_mc_evals = num_outer * 1000  # N_inner from double_loop_MCS_generate_data.py

print(f"Total model evaluations: {n_tse_evals:,} "
      f"(vs. {n_mc_evals:,} for the double-loop MC inner samples, "
      f"{n_mc_evals / n_tse_evals:,.0f}x fewer)")

# ── Envelope across the outer epistemic grid ─────────────────────────────────
T_lower, T_upper = T_mean_cond.min(axis=0), T_mean_cond.max(axis=0)
T_mid = (T_lower + T_upper) / 2

Tvar_lower, Tvar_upper = T_var_cond.min(axis=0), T_var_cond.max(axis=0)
Tvar_mid = (Tvar_lower + Tvar_upper) / 2

np.savez(
    "double_loop_TSE_1st_order_data.npz",
    EE=np.array(EE), t=t,
    T_mean_cond=T_mean_cond, T_var_cond=T_var_cond,
    n_tse_evals=n_tse_evals,
)

# ── Ground truth for comparison: double_loop_MCS_data.npz ───────────────────
mc_data = np.load("double_loop_MCS_data.npz")
T_mc = mc_data["T"]  # shape (num_outer, N_inner, len(t))
T_mean_cond_mc = T_mc.mean(axis=1)
T_var_cond_mc = T_mc.var(axis=1)
T_lower_mc, T_upper_mc = T_mean_cond_mc.min(axis=0), T_mean_cond_mc.max(axis=0)
Tvar_lower_mc, Tvar_upper_mc = T_var_cond_mc.min(axis=0), T_var_cond_mc.max(axis=0)

# ── Plot: E[T] bounds, double-loop TSE ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
for m in range(num_outer):
    ax.plot(t, T_mean_cond[m, :], color="0.82", lw=0.4, zorder=1)
ax.fill_between(t, T_lower, T_upper, color="#185FA5", alpha=0.18, zorder=2,
                 label=r"Bounds on $\mathbb{E}[T]$")
ax.plot(t, T_lower, color="#0C447C", lw=1.8, zorder=3, label="Double loop TSE lower")
ax.plot(t, T_upper, color="#185FA5", lw=1.8, zorder=3, label="Double loop TSE upper")
ax.plot(t, T_mid, color="#A32D2D", lw=1.5, ls="--", zorder=4, label="Double loop TSE midpoint")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\mathbb{E}[T]$ [K]")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/double_loop_TSE_1st_order_E_T_vs_t.png", transparent=True)
plt.close(fig)

# ── Plot: Var[T] bounds, double-loop TSE ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
for m in range(num_outer):
    ax.plot(t, T_var_cond[m, :], color="0.82", lw=0.4, zorder=1)
ax.fill_between(t, Tvar_lower, Tvar_upper, color="#185FA5", alpha=0.18, zorder=2,
                 label=r"Bounds on $\mathrm{Var}[T]$")
ax.plot(t, Tvar_lower, color="#0C447C", lw=1.8, zorder=3, label="Double loop TSE lower")
ax.plot(t, Tvar_upper, color="#185FA5", lw=1.8, zorder=3, label="Double loop TSE upper")
ax.plot(t, Tvar_mid, color="#A32D2D", lw=1.5, ls="--", zorder=4, label="Double loop TSE midpoint")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\mathrm{Var}[T]$ [K$^2$]")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/double_loop_TSE_1st_order_Var_T_vs_t.png", transparent=True)
plt.close(fig)

print(f"\n{'':<20}{'DL-TSE lower':>14}{'DL-TSE upper':>14}{'MC lower':>12}{'MC upper':>12}")
print(f"{'E[T] (t=450s)':<20}{T_lower[-1]:>14.4f}{T_upper[-1]:>14.4f}"
      f"{T_lower_mc[-1]:>12.4f}{T_upper_mc[-1]:>12.4f}")
print(f"{'Var[T] (t=450s)':<20}{Tvar_lower[-1]:>14.4f}{Tvar_upper[-1]:>14.4f}"
      f"{Tvar_lower_mc[-1]:>12.4f}{Tvar_upper_mc[-1]:>12.4f}")
