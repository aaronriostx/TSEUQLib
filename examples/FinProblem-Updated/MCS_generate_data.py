import numpy as np
import matplotlib.pyplot as plt
import fin_problem as fp
from fin_params import (
    num_of_samples, k, Cp, rho, hU, Tinf, Tw, b, T0,
    d, L, x, t,
)

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