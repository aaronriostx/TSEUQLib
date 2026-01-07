## poly4d_oti_partVar_o2_sens.py

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
import tseuqlib.oti_util as uoti

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
    
    return ( 1 + 2 * x[0] + 3 * x[1] + 5 * x[2] + 7 * x[3] )**3

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
# +oti.e(5, order=1)
n_dim = len(rv_a) # number of random variables (dimensions)

# store random variable parameters in rv_params as rv_params = [rv_pdf_name, rv_a, rv_b, rv_mean, rv_stdev]
rv_params = rv.rv_mean_from_ab([rv_pdf_name, rv_a, rv_b])


print(rv_params)

# order of Taylor series expansion
tse_order_max = 5


# random variable(s) with nominal values at the mean and perturbed with OTI imaginary directions
x = [0]*n_dim
for d in range(n_dim):
    x[d] = rv_params[3][d] + oti.e(int(d+1), order=tse_order_max)


# generate moments of the random variables
rv_moments_order = int(tse_order_max*4) #TODO: generalize for moments past 12th order

# compute central moments of the random variables
# rv moments object
rv_moments = rv.rv_central_moments(rv_params, rv_moments_order)
# compute moments
rv_moments.compute_central_moments()
print(rv_moments.rv_mu)

# Build the joint distribution from uncorrelated variables.
rv_mu_joint = uoti.build_rv_joint_moments(rv_moments.rv_mu)

print(" ---------------------------------------------------")
print(" A total of 4 variables are considered with a       ")
print(" Kumaraswamy distribution. ")
print()
print(" mean values: ", rv_moments.rv_mean)
print(" central moments: ",)
# Print moment list.
for i in range(len(rv_moments.rv_mu)):
    print(f' - mu_{i+1}:', rv_moments.rv_mu[i])
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
print('tse =', y)

print(" ---------------------------------------------------")
print()


# create oti_moments object
oti_mu = moti.tse_uq(n_dim)


# Expectation
mu_y   = oti_mu.expectation(y, rv_mu_joint)
print('mu_y =', mu_y.real)
# Extract sensitivities of central moments wrt rv parameters
try:
    dev_db1 = mu_y.get_deriv([5])
    print('dev_db1 =', dev_db1)
except:
    pass
# input('i')

# variance
mu2     = oti_mu.central_moment(y, 2, rv_mu_joint)
print('mu2 =', mu2.real)
try:
    dmu2_db1 = mu2.get_deriv([5])
    print('dmu2_db1 =', dmu2_db1)
except:
    pass
# input('i')

# Third central moment
mu3 = oti_mu.central_moment(y, 3, rv_mu_joint)
print('mu3 =', mu3.real)
try:
    dmu3_db1 = mu3.get_deriv([5])
    print('dmu3_db1 =', dmu3_db1)
except:
    pass
# input('i')

# Fourth central moment
mu4 = oti_mu.central_moment(y, 4, rv_mu_joint)
print('mu4 =', mu4.real)
try:
    dmu4_db1 = mu4.get_deriv([5])
    print('dmu4_db1 =', dmu4_db1)
except:
    pass
# input('i')

print( " Stats. results from OTI. Compare to Table 2.3 Pg 39 ")
print(" Expected Value:   {0}".format(mu_y.real))
print(" Total variance:   {0}".format(mu2.real))
print(" Third moment:     {0}".format(mu3.real))
print(" Fourth moment:    {0}".format(mu4.real))
print()

print(" ---------------------------------------------------")
print()

# Compute Sobol
sobol, Vy_hdmr = oti_mu.sobol_indices(y, rv_mu_joint)
print('Vy_hdmr =', Vy_hdmr)
# print(sobol)
for i in range(len(sobol)):
    print(f" -> Order --  {i}:")
    print('sobol:   ', [sobol[i][j].real for j in range(len(sobol[i]))] )
    print('Vi:      ', [Vy_hdmr[i][j].real for j in range(len(sobol[i]))] )
    #if i>0:
    #    print('derivs: ', [sobol[i][j].get_deriv([5]) for j in range(len(sobol[i]))] )


# get error of the Taylor series expansion, using the remainder terms


k_list=[1,2,3,4,5]

y_ev_rem  = np.zeros((tse_order_max, len(k_list)))
y_mu2_rem = np.zeros((tse_order_max, len(k_list)))
y_mu3_rem = np.zeros((tse_order_max, len(k_list)))
y_mu4_rem = np.zeros((tse_order_max, len(k_list)))
y_Si_rem = np.zeros((tse_order_max+1,n_dim))


Ef = 310.725431605432
mu2f = float(400885535046969013/7761123995625)
mu3f = 16706254.3029196
mu4f = 15315376784.8466


for i, i_order in enumerate(range(1,tse_order_max+1)):
    for ck, k in enumerate(k_list):
        y_ev_rem[i, ck] = oti_mu.expectation(y.truncate_order(i_order+k), rv_mu_joint) - oti_mu.expectation(y.truncate_order(i_order), rv_mu_joint)
        y_mu2_rem[i, ck] = oti_mu.central_moment(y.truncate_order(i_order+k), 2, rv_mu_joint) - oti_mu.central_moment(y.truncate_order(i_order), 2, rv_mu_joint)
        y_mu3_rem[i, ck] = oti_mu.central_moment(y.truncate_order(i_order+k), 3, rv_mu_joint) - oti_mu.central_moment(y.truncate_order(i_order), 3, rv_mu_joint)
        y_mu4_rem[i, ck] = oti_mu.central_moment(y.truncate_order(i_order+k), 4, rv_mu_joint) - oti_mu.central_moment(y.truncate_order(i_order), 4, rv_mu_joint)

print()
print('y_ev_rem = {0}, sum = {1}'.format(y_ev_rem, np.sum(y_ev_rem)))
print('y_mu2_rem = {0}, sum = {1}'.format(y_mu2_rem, np.sum(y_mu2_rem)))
print('y_mu3_rem = {0}, sum = {1}'.format(y_mu3_rem, np.sum(y_mu3_rem)))
print('y_mu4_rem = {0}, sum = {1}'.format(y_mu4_rem, np.sum(y_mu4_rem)))

# compute true error

y_ev_err_true  = np.zeros((tse_order_max))
y_mu2_err_true = np.zeros((tse_order_max))
y_mu3_err_true = np.zeros((tse_order_max))
y_mu4_err_true = np.zeros((tse_order_max))

for i, i_order in enumerate(range(1,tse_order_max+1)):
    y_ev_err_true[i] = Ef - oti_mu.expectation(y.truncate_order(i_order), rv_mu_joint) 
    y_mu2_err_true[i] = mu2f - oti_mu.central_moment(y.truncate_order(i_order), 2, rv_mu_joint) 
    y_mu3_err_true[i] = mu3f - oti_mu.central_moment(y.truncate_order(i_order), 3, rv_mu_joint) 
    y_mu4_err_true[i] = mu4f - oti_mu.central_moment(y.truncate_order(i_order), 4, rv_mu_joint) 


y_ev_rem_rel_err  = np.zeros((tse_order_max, len(k_list)))
y_mu2_rem_rel_err = np.zeros((tse_order_max, len(k_list)))
y_mu3_rem_rel_err = np.zeros((tse_order_max, len(k_list)))
y_mu4_rem_rel_err = np.zeros((tse_order_max, len(k_list)))


for i, i_order in enumerate(range(1,tse_order_max+1)):
    for ck, k in enumerate(k_list):
        y_ev_rem_rel_err[i,ck] = 1-(y_ev_rem[i,ck]/y_ev_err_true[i])
        y_mu2_rem_rel_err[i,ck] = 1-(y_mu2_rem[i,ck]/y_mu2_err_true[i])
        y_mu3_rem_rel_err[i,ck] = 1-(y_mu3_rem[i,ck]/y_mu3_err_true[i])
        y_mu4_rem_rel_err[i,ck] = 1-(y_mu4_rem[i,ck]/y_mu4_err_true[i])


print()
print('rel err. of Ef err = ', y_ev_rem_rel_err)
print('rel err. of mu2 err = ', y_mu2_rem_rel_err)
print('rel err. of mu3 err = ', y_mu3_rem_rel_err)
print('rel err. of mu4 err = ', y_mu4_rem_rel_err)


input('i')


