import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# Load data
data = np.load("double_loop_MCS_data.npz")
t = data["t"]
T = data["T"]  # shape (num_outer, N_inner, len(t))
num_outer, N_inner, _ = T.shape

# Conditional expectation E[T | epistemic point] for each outer (epistemic) sample
T_mean_cond = T.mean(axis=1)  # shape (num_outer, len(t))

# Envelope across the epistemic grid: bounds on E[T](t)
T_lower = T_mean_cond.min(axis=0)
T_upper = T_mean_cond.max(axis=0)
T_mid = (T_lower + T_upper) / 2

# Figure: E[T] vs t, with upper/lower bound envelope from the outer epistemic loop
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

for m in range(num_outer):
    ax.plot(t, T_mean_cond[m, :], color="0.82", lw=0.4, zorder=1)

ax.fill_between(t, T_lower, T_upper, color="#185FA5", alpha=0.18, zorder=2,
                 label=r"Bounds on $\mathbb{E}[T]$")
ax.plot(t, T_lower, color="#0C447C", lw=2.0, zorder=3, label="Lower bound")
ax.plot(t, T_upper, color="#185FA5", lw=2.0, zorder=3, label="Upper bound")
ax.plot(t, T_mid, color="#A32D2D", lw=1.5, ls="--", zorder=4, label="Midpoint")

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\mathbb{E}[T]$ [K]")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/double_loop_MCS_T_expectation_vs_t.png", transparent=True)
plt.close(fig)

# Conditional variance Var[T | epistemic point] for each outer (epistemic) sample
T_var_cond = T.var(axis=1)  # shape (num_outer, len(t))

# Envelope across the epistemic grid: bounds on Var[T](t)
Tvar_lower = T_var_cond.min(axis=0)
Tvar_upper = T_var_cond.max(axis=0)
Tvar_mid = (Tvar_lower + Tvar_upper) / 2

# Figure: Var[T] vs t, with upper/lower bound envelope from the outer epistemic loop
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

for m in range(num_outer):
    ax.plot(t, T_var_cond[m, :], color="0.82", lw=0.4, zorder=1)

ax.fill_between(t, Tvar_lower, Tvar_upper, color="#185FA5", alpha=0.18, zorder=2,
                 label=r"Bounds on $\mathrm{Var}[T]$")
ax.plot(t, Tvar_lower, color="#0C447C", lw=2.0, zorder=3, label="Lower bound")
ax.plot(t, Tvar_upper, color="#185FA5", lw=2.0, zorder=3, label="Upper bound")
ax.plot(t, Tvar_mid, color="#A32D2D", lw=1.5, ls="--", zorder=4, label="Midpoint")

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\mathrm{Var}[T]$ [K$^2$]")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/double_loop_MCS_T_variance_vs_t.png", transparent=True)
plt.close(fig)
