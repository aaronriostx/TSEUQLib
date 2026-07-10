"""
First-order (main-effect) Sobol' indices for the thermal fin problem, comparing
two Monte Carlo estimators for the partial variances of k, Cp, rho, hU:

    Crude MC       : fix the other aleatory vars at their means and vary Xi
                      alone (see additive-model-MC.py for the same convention).
    Double-loop MC : Vi = Var_Xi[ E_X~i(T | Xi) ], estimated by an outer loop
                      over Xi and an inner loop resampling the other vars.

Epistemic variables (Tinf, Tw, b, STD_hU) are fixed at their interval
midpoints for both methods, so each method returns one scalar-in-time index
per variable rather than an interval.
"""
import numpy as np
import matplotlib.pyplot as plt
import fin_problem as fp
from fin_params_mixed import SEED, MEAN, STD, MEAN_hU, STD_hU, Tinf, Tw, b, d, L, x, t

rng_seed = SEED + 1000  # separate seed block from data-generation scripts

# ── Nominal (midpoint) epistemic point ───────────────────────────────────────
Tinf_nom = (Tinf[0] + Tinf[1]) / 2
Tw_nom = (Tw[0] + Tw[1]) / 2
b_nom = (b[0] + b[1]) / 2
T0_nom = Tinf_nom
STD_hU_nom = (STD_hU[0] + STD_hU[1]) / 2

# ── Aleatory variable specs: k, Cp, rho, hU ──────────────────────────────────
names = ["k", "Cp", "rho", "hU"]
means = [MEAN[0], MEAN[1], MEAN[2], MEAN_hU]
stds = [STD[0], STD[1], STD[2], STD_hU_nom]
n_vars = len(names)

# ── Loop sizes ────────────────────────────────────────────────────────────────
N_total = 20_000   # shared baseline sample -> total variance denominator
N_crude = 5_000    # crude MC: samples per variable
N_outer = 25       # double-loop: outer (Sobol) samples per variable
N_inner = 200       # double-loop: inner resamples of the complementary vars


def draw(idx, n, seed):
    return fp.sample("normal", n, {"mean": means[idx], "std": stds[idx]}, seed=seed)


def eval_T(k_s, Cp_s, rho_s, hU_s):
    """Evaluate the fin temperature history for arrays of aleatory samples,
    with epistemic parameters fixed at their midpoints."""
    n = len(k_s)
    T_out = np.zeros((n, len(t)))
    for j in range(n):
        T, _, _, _ = fp.analytic_solution(
            x, t, b_nom, d, L, rho_s[j], Cp_s[j], k_s[j], hU_s[j],
            T0_nom, Tw_nom, Tinf_nom,
        )
        T_out[j, :] = T
    return T_out


# ── Shared baseline: total variance Var[T](t) at the nominal epistemic point ─
seed = rng_seed
k_b = draw(0, N_total, seed); seed += 1
Cp_b = draw(1, N_total, seed); seed += 1
rho_b = draw(2, N_total, seed); seed += 1
hU_b = draw(3, N_total, seed); seed += 1
T_base = eval_T(k_b, Cp_b, rho_b, hU_b)
Var_total = T_base.var(axis=0)  # shape (len(t),)

# ── Method 1: Crude MC (others fixed at mean, vary Xi alone) ─────────────────
V_crude = np.zeros((n_vars, len(t)))
for i in range(n_vars):
    samples = [np.full(N_crude, means[j]) for j in range(n_vars)]
    samples[i] = draw(i, N_crude, seed); seed += 1
    T_i = eval_T(*samples)
    V_crude[i, :] = T_i.var(axis=0)

with np.errstate(invalid="ignore"):
    S_crude = np.where(Var_total > 0, V_crude / Var_total, 0.0)

# ── Method 2: Double-loop MC (nested conditional variance) ──────────────────
V_double = np.zeros((n_vars, len(t)))
for i in range(n_vars):
    x_i_outer = draw(i, N_outer, seed); seed += 1
    cond_mean = np.zeros((N_outer, len(t)))
    for m in range(N_outer):
        samples = [None] * n_vars
        for j in range(n_vars):
            if j == i:
                samples[j] = np.full(N_inner, x_i_outer[m])
            else:
                samples[j] = draw(j, N_inner, seed); seed += 1
        T_inner = eval_T(*samples)
        cond_mean[m, :] = T_inner.mean(axis=0)
    V_double[i, :] = cond_mean.var(axis=0)

with np.errstate(invalid="ignore"):
    S_double = np.where(Var_total > 0, V_double / Var_total, 0.0)

# ── Summary at final time ────────────────────────────────────────────────────
print(f"Total model evaluations: "
      f"{N_total + n_vars * N_crude + n_vars * N_outer * N_inner:,}\n")
print(f"{'var':<6}{'S_i crude (t=450s)':>22}{'S_i double-loop (t=450s)':>28}")
for i, name in enumerate(names):
    print(f"{name:<6}{S_crude[i, -1]:>22.4f}{S_double[i, -1]:>28.4f}")
print(f"{'sum':<6}{S_crude[:, -1].sum():>22.4f}{S_double[:, -1].sum():>28.4f}")

# ── Plot: S_i(t) for both methods, small multiples ──────────────────────────
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]

fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300, sharey=True)

for i, name in enumerate(names):
    axes[0].plot(t, S_crude[i, :], color=CATEGORICAL[i], lw=1.8, label=name)
    axes[1].plot(t, S_double[i, :], color=CATEGORICAL[i], lw=1.8, label=name)

axes[0].set_title("Crude MC", fontsize=10)
axes[1].set_title("Double-loop MC", fontsize=10)
for ax in axes:
    ax.set_xlabel("Time [s]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
axes[0].set_ylabel(r"$S_i(t)$")
axes[1].legend(fontsize=8, loc="upper right")

fig.tight_layout()
fig.savefig("figures/sobol_crude_vs_double_loop.png", transparent=True)
plt.close(fig)
