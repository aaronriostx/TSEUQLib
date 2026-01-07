import pyoti.sparse as oti

# Set pyoti to print all coefficients.
oti.set_printoptions(terms_print=-1)


import numpy as np
np.set_printoptions(linewidth=120)

# path to tseuqlib
import sys
sys.path.append('../')

# tseuqlib
import tseuqlib as tuq
import tseuqlib.rv_moments as rv
import tseuqlib.oti_moments as moti


# This example implements the HYPAD-UQ method described in 
# Dr. Matthew Balcer's PhD thesis, 2023, Section 2.1.7.2, Pg. 37.

# Second order polynomial:

# =================================================================================
def funct(x):
    """
    Equation 2.92 in 

    INPUTS:
    - xi: OTI number.
    """

    return ( 1 + 2 * x[0] + 3 * x[1] + 5 * x[2] + 7 * x[3] )**2

# ---------------------------------------------------------------------------------


print(" ---------------------------------------------------")
print(" This example file runs a Verification example from ")
print(" Dr. Matthew Balcer's PhD Thesis (2023). It follows ")
print(" verification example in Section 2.1.7.2, Pg. 37.   ")
print(" ---------------------------------------------------")

# Extracted from Table 2.1. (Pg. 35)



# Random variable parameters
rv_pdf_name = np.array(['K','K','K','K']) # PDF of each variable
rv_a = np.array([1,1,2,2]) # shape parameter of each variable
rv_b = np.array([3,8,6,4]) # scale parameter of each variable
n_dim = len(rv_a) # number of random variables (dimensions)

# store random variable parameters in rv_params as rv_params = [rv_pdf_name, rv_a, rv_b, rv_mean, rv_stdev]
rv_params = rv.rv_mean_from_ab([rv_pdf_name, rv_a, rv_b])


print(rv_params)

# order of Taylor series expansion
tse_order_max = 3


# random variable(s) with nominal values at the mean and perturbed with OTI imaginary directions
x = [0]*n_dim
for d in range(n_dim):
    x[d] = rv_params[3][d] + oti.e(int(d+1), order=tse_order_max)


# generate moments of the random variables
rv_moments_order = 12 #TODO: generalize for moments past 12th order

# compute second through twelfth central moments of the random variables
pdfInp, lb, ub, meanVal, mx02, mx03, mx04, mx05, mx06, mx07, mx08, mx09, mx10, mx11, mx12 = \
rv.rv_central_moments(rv_params, rv_moments_order)

rv_mu = [[0]*n_dim, mx02, mx03, mx04, mx05, mx06, mx07, mx08, mx09, mx10, mx11, mx12]

# Build the joint distribution from uncorrelated variables.
rv_mu_joint = moti.build_joint(rv_mu)

print(" ---------------------------------------------------")
print(" A total of 4 variables are considered with a       ")
print(" Kumaraswamy distribution. ")
print()
print(" mean values: ", meanVal)
print(" central moments: ",)
# Print moment list.
for i in range(len(rv_mu)):
    print(f' - mu_{i+1}:', rv_mu[i])
# end for
print()
print(" Joint distribution built from the moments: (uncomment lines 79-81)")
# # for i in range(len(rv_mu_joint)):
# #     print(rv_mu_joint[i])
# # # end for
print()
print(" ---------------------------------------------------")
print()


# OTI evaluation:


y =  funct(x)
print(' Function evaluated at OTI-perturbed inputs:')
print(y)

print(" ---------------------------------------------------")
print()


# Expectation and Variance estimates

mu_y   = moti.oti_expectation(y,rv_mu_joint)
Vy     = moti.oti_central_moment(y,2,rv_mu_joint)

# Third and fourth moment estimates.
muc_y3 = moti.oti_central_moment(y,3,rv_mu_joint)
muc_y4 = moti.oti_central_moment(y,4,rv_mu_joint)

print( " Stats. results from OTI. Compare to Table 2.3 Pg 39 ")
print(f" Expected Value:   {mu_y:10.3f}")
print(f" Total Variance:   {Vy:10.3f}")
print(f" central_moment 3: {muc_y3:10.3f}")
print(f" central_moment 4: {muc_y4:10.3f}")
print()

print(" ---------------------------------------------------")
print()



# HDMR representation

# f0 is the expected value.
f0 = mu_y

# TODO: Automate HDMR generation.
# - Need to think of a datastructure that can hold this
# - How to operate the subtractions.

# HDMRs: 
# Use selective expectation to take variables out of the consideration 
#
f1 = moti.oti_selective_expectation(y,rv_mu_joint,[1]) - f0
f2 = moti.oti_selective_expectation(y,rv_mu_joint,[2]) - f0
f3 = moti.oti_selective_expectation(y,rv_mu_joint,[3]) - f0
f4 = moti.oti_selective_expectation(y,rv_mu_joint,[4]) - f0


f12 = moti.oti_selective_expectation(y,rv_mu_joint,[1,2]) - f1 - f2 - f0
f13 = moti.oti_selective_expectation(y,rv_mu_joint,[1,3]) - f1 - f3 - f0
f23 = moti.oti_selective_expectation(y,rv_mu_joint,[2,3]) - f2 - f3 - f0
f14 = moti.oti_selective_expectation(y,rv_mu_joint,[1,4]) - f1 - f4 - f0
f24 = moti.oti_selective_expectation(y,rv_mu_joint,[2,4]) - f2 - f4 - f0
f34 = moti.oti_selective_expectation(y,rv_mu_joint,[3,4]) - f3 - f4 - f0
f123 = moti.oti_selective_expectation(y,rv_mu_joint,[1,2,3]) - f12 - f13 - f23 - f1 - f2 - f3 - f0
f124 = moti.oti_selective_expectation(y,rv_mu_joint,[1,2,4]) - f12 - f14 - f24 - f1 - f2 - f4 - f0
f134 = moti.oti_selective_expectation(y,rv_mu_joint,[1,3,4]) - f13 - f14 - f34 - f1 - f3 - f4 - f0
f234 = moti.oti_selective_expectation(y,rv_mu_joint,[2,3,4]) - f23 - f24 - f34 - f2 - f3 - f4 - f0
f1234 = moti.oti_selective_expectation(y,rv_mu_joint,[1,2,3,4]) + (- f123 - f124 - f134 - f234
                                                       - f12 - f13 - f23 - f14 - f24 - f34 
                                                       - f1 - f2 - f3 - f4 
                                                       - f0)




V1 = moti.oti_central_moment(f1,2,rv_mu_joint)
V2 = moti.oti_central_moment(f2,2,rv_mu_joint)
V3 = moti.oti_central_moment(f3,2,rv_mu_joint)
V4 = moti.oti_central_moment(f4,2,rv_mu_joint)

V12 = moti.oti_central_moment(f12,2,rv_mu_joint)
V13 = moti.oti_central_moment(f13,2,rv_mu_joint)
V23 = moti.oti_central_moment(f23,2,rv_mu_joint)
V14 = moti.oti_central_moment(f14,2,rv_mu_joint)
V24 = moti.oti_central_moment(f24,2,rv_mu_joint)
V34 = moti.oti_central_moment(f34,2,rv_mu_joint)

V123 = moti.oti_central_moment(f123,2,rv_mu_joint)
V124 = moti.oti_central_moment(f124,2,rv_mu_joint)
V134 = moti.oti_central_moment(f134,2,rv_mu_joint)
V234 = moti.oti_central_moment(f234,2,rv_mu_joint)

V1234 = moti.oti_central_moment(f1234,2,rv_mu_joint)

print("Partial Variances from HDMR built from OTIs.")
print("Compare to Table 2.3 Pg 39.")

print()
print(f' V1:    {V1:10.5f}')
print(f' V2:    {V2:10.5f}')
print(f' V3:    {V3:10.5f}')
print(f' V4:    {V4:10.5f}')

print()
print(f' V12:   {V12:10.5f}')
print(f' V13:   {V13:10.5f}')
print(f' V14:   {V14:10.5f}')
print(f' V23:   {V23:10.5f}')
print(f' V24:   {V24:10.5f}')
print(f' V34:   {V34:10.5f}')
print()
print(f' V123:  {V123:10.5f}')
print(f' V124:  {V124:10.5f}')
print(f' V134:  {V134:10.5f}')
print(f' V234:  {V234:10.5f}')
print()
print(f' V1234: {V1234:10.5f}')
print()
# Verify that partial variances add the total variance:
Vi_sum = np.sum([V1,   V2,   V3,   V4,
                 V12,  V13,  V23,  V14,  V24,  V34,
                 V123, V124, V134, V234,
                 V1234])
print(' Sum of Partial variances:',Vi_sum)
print(' Relative error:',(Vi_sum-Vy)/Vy)

print()
print(" ---------------------------------------------------")
print()

print(" HDMRs obtained by OTI look different than the ones in the Thesis.")
print(" This is because the OTi form has the TSE coeffs, while the Thesis")
print(" has the Polynomial form.")
print(" The best part is that these forms can be easily transformed from")
print(" one to the other:")


print()
# Define transformation from OTI to Function Polynomial.
dirs  = [1,2,3,4]
delts = [ oti.e(1,order=tse_order_max)-mu_x1,
          oti.e(2,order=tse_order_max)-mu_x2,
          oti.e(3,order=tse_order_max)-mu_x3,
          oti.e(4,order=tse_order_max)-mu_x4 ]

print(" > OTI form obtained from the HDMR for f1:")
print(f1)
print()
print(" > Transformed HDMR form obtained from the OTI-form of f1")
print("    (compare to eqs. 2.94-2.103): ")
print(f1.rom_eval_object(dirs,delts))
print()
print(" > Transformed HDMR f2 (eq 2.95)")
print(f2.rom_eval_object(dirs,delts))
print()
print(" > Transformed HDMR f4 (eq 2.97)")
print(f4.rom_eval_object(dirs,delts))
print()
print(" > Transformed HDMR f13 (eq 2.99)")
print(f13.rom_eval_object(dirs,delts))

