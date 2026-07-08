"""
First-order (main-effect) Sobol' sensitivity indices via Monte Carlo,
using the Saltelli sampling scheme -- no SALib, just numpy.

Test function: the Ishigami function, the standard benchmark for Sobol
analysis because its indices are known in closed form, so we can check
the Monte Carlo estimate against the true answer.

    Y = sin(X1) + a * sin(X2)^2 + b * X3^4 * sin(X1)

with X1, X2, X3 ~ Uniform(-pi, pi), a = 7, b = 0.1
"""

import numpy as np

rng = np.random.default_rng(seed=42)

a, b = 7.0, 0.1
k = 3          # number of inputs
N = 50000     # base sample size; total model evaluations = N * (k + 2)


def ishigami(X):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * x3 ** 4 * np.sin(x1)


def sample(n, k):
    return rng.uniform(-np.pi, np.pi, size=(n, k))


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
fA = ishigami(A) # Returns vector of length, N
fB = ishigami(B) # Returns vector of length, N
fAB = np.array([ishigami(AB[i]) for i in range(k)])   # matrix shape (k, N)

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

# Analytical answer for the Ishigami function, for comparison
var_analytical = a**2 / 8 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 18 + 0.5
V1 = 0.5 * (1 + b * np.pi**4 / 5) ** 2
V2 = a**2 / 8
V3 = 0.0
S1_true = np.array([V1, V2, V3]) / var_analytical

print(f"Model evaluations used: {N * (k + 2)}\n")
print(f"{'input':<8}{'Saltelli':>12}{'analytical':>14}")
for i in range(k):
    print(f"X{i+1:<7}{S1_saltelli[i]:>12.4f}{S1_true[i]:>14.4f}")

# Crude Monte Carlo
# Fix every input except Xi at its mean (0, for Uniform(-pi, pi)) and let
# only Xi vary; the variance of Y over that ensemble estimates Var_i.
# NOTE: this only recovers the true first-order variance when the model is
# purely additive. Ishigami has an X1-X3 interaction (b * X3^4 * sin(X1)),
# so fixing X3 at a single value rather than integrating it out biases the
# estimate -- expect this to disagree with Saltelli/analytical, especially
# for X1.
S1_MC = np.empty(k)
for i in range(k):
    X = sample(N, k)
    for j in range(k):
        if j != i:
            X[:, j] = 0
    S1_MC[i] = ishigami(X).var() / var_Y

print(f"\n{'input':<8}{'Crude MC':>12}")
for i in range(k):
    print(f"X{i+1:<7}{S1_MC[i]:>12.4f}")