import numpy as np
import fin_problem as fp

SEED = 2026
num_of_samples = 50000

# Stochastic parameters (normal distributions)
COV  = np.array([0.10, 0.03, 0.03])
MEAN = np.array([7.1, 5.800e8, 4.430e-9])
STD  = COV * MEAN

k   = fp.sample("normal", num_of_samples, {"mean": MEAN[0], "std": STD[0]}, seed=SEED)
Cp  = fp.sample("normal", num_of_samples, {"mean": MEAN[1], "std": STD[1]}, seed=SEED+1)
rho = fp.sample("normal", num_of_samples, {"mean": MEAN[2], "std": STD[2]}, seed=SEED+2)

# Stochastic parameters with epistemic COV
MEAN_hU = 0.114
COV_hU = np.array([0.9, 0.11])
STD_hU  = COV_hU * MEAN_hU
hU  = fp.sample("normal", num_of_samples, {"mean": MEAN_hU, "std": STD_hU}, seed=SEED+3)

# Epistemic parameters (intervals [lower, upper])
Tinf = (282.7, 283.3)
Tw   = (369.6, 408.5)
b    = (50.49, 51.51)
T0 = Tinf

# Deterministic parameters
d = 4.75
L = 100.0

x = 0

t = np.linspace(1, 450, num=450, endpoint=True)
