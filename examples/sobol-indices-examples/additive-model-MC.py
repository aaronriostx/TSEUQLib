"""
First-order (main-effect) Sobol' sensitivity indices via Monte Carlo,
using the Saltelli sampling scheme -- no SALib, just numpy.

Test function: 
    Y = X1 + 2*X2

with X1, X2 ~ Normal(0, 1)

Original exmaple from: https://uqpyproject.readthedocs.io/en/latest/auto_examples/sensitivity/comparison/additive.html#sphx-glr-auto-examples-sensitivity-comparison-additive-py
"""

import numpy as np

rng = np.random.default_rng(seed=42)

k = 2          # number of inputs
N = 50000     # base sample size; total model evaluations = N * (k + 2)


def additive_fcn(X):
    x1, x2 = X[:, 0], X[:, 1]
    return x1 + 2*x2


def sample(n, k):
    return rng.normal(0, 1, size=(n, k))


# 1. Two independent base samples
A = sample(N, k) # A: Nxk matrix of random samples
B = sample(N, k) # B: Another Nxk matrix of random samples

# 2. Build the k "recombined" matrices AB^i: A with column i swapped for B's column i
AB = np.empty((k, N, k))
for i in range(k):
    AB_i = A.copy() # Copy A (Nxk matrix)
    AB_i[:, i] = B[:, i] # Replace i-th column with B's i-th column (Nxk matrix)
    AB[i] = AB_i # Store as k-th entry for AB (kxNxk)

print(AB.shape)

# 3. Evaluate the model at every row -> N*(k+2) total evaluations
fA = additive_fcn(A) # Returns vector of length, N
fB = additive_fcn(B) # Returns vector of length, N
fAB = np.array([additive_fcn(AB[i]) for i in range(k)])   # matrix shape (k, N)

print(fA.shape)
print(fB.shape)
print(fAB.shape)

# 4. Total variance, estimated from the pooled samples
var_Y = np.concatenate([fA, fB]).var() # Combined fA and fB and taking the variance

# 5. Saltelli (2010) estimator for each first-order index
S1_saltelli = np.array([
    np.mean(fB * (fAB[i] - fA)) / var_Y
    for i in range(k)
])

print(f"Model evaluations used: {N * (k + 2)}\n")
print(f"{'input':<8}{'Saltelli':>12}")
for i in range(k):
    print(f"X{i+1:<7}{S1_saltelli[i]:>12.4f}")