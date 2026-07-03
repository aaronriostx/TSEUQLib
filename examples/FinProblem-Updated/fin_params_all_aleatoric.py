import numpy as np
import fin_problem as fp

SEED = 2026
num_of_samples = 50000

# Stochastic parameters (normal distributions)
COV  = np.array([0.10, 0.03, 0.03, 0.10])
MEAN = np.array([7.1, 5.800e8, 4.430e-9, 0.114])
STD  = COV * MEAN

k   = fp.sample("normal", num_of_samples, {"mean": MEAN[0], "std": STD[0]}, seed=SEED)
Cp  = fp.sample("normal", num_of_samples, {"mean": MEAN[1], "std": STD[1]}, seed=SEED+1)
rho = fp.sample("normal", num_of_samples, {"mean": MEAN[2], "std": STD[2]}, seed=SEED+2)
hU  = fp.sample("normal", num_of_samples, {"mean": MEAN[3], "std": STD[3]}, seed=SEED+3)

# Now prescribing these parameters as uniform distributions
LOW = np.array([282.7, 369.6, 50.49])
HIGH = np.array([283.3, 408.5, 51.51])

Tinf = fp.sample("uniform", num_of_samples, {"low": LOW[0], "high": HIGH[0]}, seed=SEED+4)
Tw   = fp.sample("uniform", num_of_samples, {"low": LOW[1], "high": HIGH[1]}, seed=SEED+5)
b    = fp.sample("uniform", num_of_samples, {"low": LOW[2], "high": HIGH[2]}, seed=SEED+6)
T0 = Tinf

# Deterministic parameters
d = 4.75
L = 100.0

x = 0

t = np.linspace(1, 450, num=450, endpoint=True)
