import numpy as np
import fin_problem as fp
from fin_params_mixed import (
    SEED, MEAN, STD, MEAN_hU, STD_hU, Tinf, Tw, b, d, L, x, t,
)

# ── Loop sizes ─────────────────────────────────────────────────────────────
M_grid = 4        # outer (epistemic) grid points per variable
N_inner = 1000     # inner (aleatory) samples per outer point

# ── Outer loop: epistemic grid over Tinf, Tw, b, STD_hU ─────────────────────
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

# ── Allocate storage ─────────────────────────────────────────────────────────
T_data = np.zeros((num_outer, N_inner, len(t)))
theta_data = np.zeros((num_outer, N_inner, len(t)))
theta_tau_data = np.zeros((num_outer, N_inner, len(t)))
theta_ss_data = np.zeros((num_outer, N_inner, len(t)))

k_data = np.zeros((num_outer, N_inner))
Cp_data = np.zeros((num_outer, N_inner))
rho_data = np.zeros((num_outer, N_inner))
hU_data = np.zeros((num_outer, N_inner))

# ── Double loop: outer over epistemic grid, inner over aleatory samples ─────
for m, (Tinf_v, Tw_v, b_v, stdhU_v) in enumerate(EE):
    T0_v = Tinf_v  # T0 = Tinf assumption

    k_s = fp.sample("normal", N_inner, {"mean": MEAN[0], "std": STD[0]}, seed=SEED + 4 * m)
    Cp_s = fp.sample("normal", N_inner, {"mean": MEAN[1], "std": STD[1]}, seed=SEED + 4 * m + 1)
    rho_s = fp.sample("normal", N_inner, {"mean": MEAN[2], "std": STD[2]}, seed=SEED + 4 * m + 2)
    hU_s = fp.sample("normal", N_inner, {"mean": MEAN_hU, "std": stdhU_v}, seed=SEED + 4 * m + 3)

    k_data[m, :] = k_s
    Cp_data[m, :] = Cp_s
    rho_data[m, :] = rho_s
    hU_data[m, :] = hU_s

    for j in range(N_inner):
        T, theta, theta_tau, theta_ss = fp.analytic_solution(
            x, t, b_v, d, L, rho_s[j], Cp_s[j], k_s[j], hU_s[j],
            T0_v, Tw_v, Tinf_v,
        )
        T_data[m, j, :] = T
        theta_data[m, j, :] = theta
        theta_tau_data[m, j, :] = theta_tau
        theta_ss_data[m, j, :] = theta_ss

# ── Export as .npz ────────────────────────────────────────────────────────────
np.savez(
    "double_loop_MCS_data.npz",
    Tinf_vals=Tinf_vals, Tw_vals=Tw_vals, b_vals=b_vals, STD_hU_vals=STD_hU_vals,
    EE=np.array(EE),
    t=t,
    k=k_data, Cp=Cp_data, rho=rho_data, hU=hU_data,
    T=T_data, theta=theta_data, theta_tau=theta_tau_data, theta_ss=theta_ss_data,
)
