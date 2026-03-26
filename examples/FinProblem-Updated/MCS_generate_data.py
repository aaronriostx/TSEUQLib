import numpy as np
import matplotlib.pyplot as plt
import fin_problem as fp

# Define model parameters as probability distributions
SEED = 2026
num_of_samples = 1000000
COV = 0.10
k = fp.sample("normal",    num_of_samples, {"mean": 7.1, "std": 7.1*COV}, seed=SEED)
Cp = fp.sample("normal",    num_of_samples, {"mean": 5.800e8, "std": 5.800e8*COV}, seed=SEED)
rho = fp.sample("normal",    num_of_samples, {"mean": 4.430e-9 , "std": 4.430e-9*COV}, seed=SEED)
hU = fp.sample("normal",    num_of_samples, {"mean": 0.114, "std": 0.114*COV}, seed=SEED)
Tinf = fp.sample("normal",    num_of_samples, {"mean": 283.0, "std": 283.0*COV}, seed=SEED)
Tw = fp.sample("normal",    num_of_samples, {"mean": 389.0, "std": 389.0*COV}, seed=SEED)
b = fp.sample("normal",    num_of_samples, {"mean": 51.0, "std": 7.1*COV}, seed=SEED)
T0 = Tinf # Define initial temp.

# Deterministic model parameters
d     = 4.75 # Deterministic fin thickness
L     = 100.0 # Fin width

# Fin location
x = 0

# Time points
t = np.linspace(1, 450, num=450, endpoint=True)

# Initialize array to store data
T_data = np.zeros((num_of_samples, len(t)))
theta_data = np.zeros((num_of_samples, len(t)))
theta_tau_data = np.zeros((num_of_samples, len(t)))
theta_ss_data = np.zeros((num_of_samples, len(t)))

for i in range(num_of_samples):
    # Solve step
    T, theta, theta_tau, theta_ss = fp.analytic_solution(x, t, b[i], d, L, rho[i], Cp[i], k[i], hU[i], T0[i], Tw[i], Tinf[i])

    # Save to arrays
    T_data[i, :] = T
    theta_data[i, :] = theta
    theta_tau_data[i, :] = theta_tau
    theta_ss_data[i, :] = theta_ss

# Export as .npz files
np.savez("MCS_data.npz", t=t, T=T_data, theta=theta_data, theta_tau=theta_tau_data, theta_ss=theta_ss_data)