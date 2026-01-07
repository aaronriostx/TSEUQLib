# Compute central moments of the Taylor series expansion of a 1D polynomial

# path to tseuqlib
import sys
sys.path.append('../')

# tseuqlib
import tseuqlib as tuq
import tseuqlib.rv_moments as rv
import tseuqlib.oti_moments as moti

# pyoti
import pyoti.sparse as oti
import pyoti.core as coti

import numpy as np
# import scipy.stats as stats
import matplotlib.pyplot as plt
# from scipy.special  import gamma, factorial2, binom


# true model
def func(x, alg=oti):
    # return alg.sin(x**2)
    return 1 + 2*x + 3*x**2 + 5*x**3 # +7*x**4

# Random variable parameters
rv_pdf_name = np.array(['N']) # PDF of each variable
rv_a = np.array([1]) # shape parameter of each variable
rv_b = np.array([0.5]) # scale parameter of each variable
n_dim = len(rv_a) # number of random variables (dimensions)

# store random variable parameters in rv_params as rv_params = [rv_pdf_name, rv_a, rv_b, rv_mean, rv_stdev]
rv_params = rv.rv_mean_from_ab([rv_pdf_name, rv_a, rv_b])

# order of Taylor series expansion
tse_order_max = 3


# random variable(s) with nominal values at the mean and perturbed with OTI imaginary directions
x = rv_params[3][0] + oti.e(1, order=tse_order_max)

# construct Taylor series expansion, y
y = func(x)


# generate moments of the random variables
rv_moments_order = 12 #TODO: generalize for moments past 12th order

# compute second through twelfth central moments of the random variables
pdfInp, lb, ub, meanVal, mx02, mx03, mx04, mx05, mx06, mx07, mx08, mx09, mx10, mx11, mx12 = \
rv.rv_central_moments(rv_params, rv_moments_order)

rv_mu = [[0]*n_dim, mx02, mx03, mx04, mx05, mx06, mx07, mx08, mx09, mx10, mx11, mx12]

rv_mu_joint = rv.joint_central_moments(rv_mu, rv_moments_order)

# compute moments of OTI number
mu1_oti = moti.oti_expectation(y, rv_mu_joint)
mu2_oti = moti.oti_central_moment(y, 2, rv_mu_joint)
mu3_oti = moti.oti_central_moment(y, 3, rv_mu_joint)
mu4_oti = moti.oti_central_moment(y, 4, rv_mu_joint)
# mu5_oti = moti.oti_central_moment(y, 5, rv_mu_joint)

print('Central moments of OTI number:')
print('Mean:', mu1_oti)
print('Var: ', mu2_oti)
print('mu3: ', mu3_oti)
print('mu4: ', mu4_oti)
#print('mu5: ', mu5_oti)
print()



# closed-form central moments of the Taylor series expansion
tse_mu = tuq.tse_moments.tse_central_moment([y], n_dim, tse_order_max, rv_mu)

# central moments of the third Taylor series
mu1_tse = tse_mu.tseEvThird()[0]
mu2_tse = tse_mu.tseVarThird()[0][0]
mu3_tse = tse_mu.tseTcmThird()[0]
mu4_tse = tse_mu.tseFcmThird()[0]

print('Exact central moments of Taylor series:')
print('Mean:', mu1_tse)
print('Var: ', mu2_tse)
print('mu3: ', mu3_tse)
print('mu4: ', mu4_tse)
print()



# relative error 1-(oti_moment/tse_moment)
rel_err_mu1 = 1-(mu1_oti/mu1_tse)
rel_err_mu2 = 1-(mu2_oti/mu2_tse)
rel_err_mu3 = 1-(mu3_oti/mu3_tse)
rel_err_mu4 = 1-(mu4_oti/mu4_tse)

print('Relative error 1-(oti_moment/tse_moment):')
print('rel_err_mu1 =', rel_err_mu1)
print('rel_err_mu2 =', rel_err_mu2)
print('rel_err_mu3 =', rel_err_mu3)
print('rel_err_mu4 =', rel_err_mu4)