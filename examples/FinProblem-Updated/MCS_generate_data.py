import numpy as np
import matplotlib.pyplot as plt
import fin_problem as fp

# Define model parameters as probability distributions
SEED = 2026
num_of_samples = 50000

COV = [0.10, 0.03, 0.03, 0.10, 0.001, 0.05, 0.01]
k = fp.sample("normal",    num_of_samples, {"mean": 7.1, "std": 7.1*COV[0]}, seed=SEED)
Cp = fp.sample("normal",    num_of_samples, {"mean": 5.800e8, "std": 5.800e8*COV[1]}, seed=SEED+1)
rho = fp.sample("normal",    num_of_samples, {"mean": 4.430e-9 , "std": 4.430e-9*COV[2]}, seed=SEED+2)
hU = fp.sample("normal",    num_of_samples, {"mean": 0.114, "std": 0.114*COV[3]}, seed=SEED+3)
Tinf = fp.sample("normal",    num_of_samples, {"mean": 283.0, "std": 283.0*COV[4]}, seed=SEED+4)
Tw = fp.sample("normal",    num_of_samples, {"mean": 389.0, "std": 389.0*COV[5]}, seed=SEED+5)
b = fp.sample("normal",    num_of_samples, {"mean": 51, "std": 51*COV[6]}, seed=SEED+6)
T0 = Tinf # Define initial temp.

# rv_mean     = np.array([7.1, 580, 4430, 114.0, 283.0, 389,  0.051])
# rv_stdev    = rv_mean*np.array([0.10, 0.03, 0.03, 0.10, 0.001, 0.05, 0.01])

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
np.savez("MCS_data.npz", 
         k=k, Cp=Cp, rho=rho, hU=hU, Tinf=Tinf, Tw=Tw, b=b, T0=T0,
         t=t, T=T_data, theta=theta_data, theta_tau=theta_tau_data, theta_ss=theta_ss_data)