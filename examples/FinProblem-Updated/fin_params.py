import numpy as np
import fin_problem as fp

SEED = 2026
num_of_samples = 50000

COV = np.array([0.10, 0.03, 0.03, 0.10, 0.001, 0.05, 0.01])
MEAN = np.array([7.1, 5.800e8, 4.430e-9, 0.114, 283.0, 389.0, 51])
STD = COV*MEAN

k    = fp.sample("normal", num_of_samples, {"mean": MEAN[0],      "std": STD[0]},      seed=SEED)
Cp   = fp.sample("normal", num_of_samples, {"mean": MEAN[1],  "std": STD[1]},  seed=SEED+1)
rho  = fp.sample("normal", num_of_samples, {"mean": MEAN[2], "std": STD[2]}, seed=SEED+2)
hU   = fp.sample("normal", num_of_samples, {"mean": MEAN[3],    "std": STD[3]},    seed=SEED+3)
Tinf = fp.sample("normal", num_of_samples, {"mean": MEAN[4],    "std": STD[4]},    seed=SEED+4)
Tw   = fp.sample("normal", num_of_samples, {"mean": MEAN[5],    "std": STD[5]},    seed=SEED+5)
b    = fp.sample("normal", num_of_samples, {"mean": MEAN[6],       "std": STD[6]},        seed=SEED+6)
T0   = Tinf

d = 4.75
L = 100.0

x = 0

t = np.linspace(1, 450, num=450, endpoint=True)
