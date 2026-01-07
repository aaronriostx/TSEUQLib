## poly4d_oti_partVar_o2_sens.py

import pyoti.sparse as oti

# Set pyoti to print all coefficients.
oti.set_printoptions(terms_print=-1)


import numpy as np
np.set_printoptions(precision=16, linewidth=500)

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


def polyfd(h):
    extra_bases = []

    # Random variable parameters
    rv_pdf_name = np.array(['N']*4) # PDF of each variable
    rv_a = np.array([0.01+h,0.01,0.02,0.02]) # shape parameter of each variable
    rv_b = np.array([0.003,0.008,0.006,0.004]) # scale parameter of each variable
    # +oti.e(5, order=1)
    n_dim = len(rv_a) # number of random variables (dimensions)
    
    # store random variable parameters in rv_params as rv_params = [rv_pdf_name, rv_a, rv_b, rv_mean, rv_stdev]
    rv_params = rv.rv_mean_from_ab([rv_pdf_name, rv_a, rv_b])
    
    
    print(rv_params)
    
    # order of Taylor series expansion
    tse_order_max = 4
    
    
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
    
    
    # OTI evaluation:
    
    
    y =  funct(x)
    
    
    # create oti_moments object
    oti_mu = moti.tse_uq(n_dim, rv_mu_joint, extra_bases=extra_bases)
    
    
    # non-central moment
    # mu2c     = oti_mu.central_moment(y, 2, Ey=2)
    # print('mu2c =', mu2c.real)
    # print(float(563263737029933/3832653825))
    # input('i')
    
    # Expectation
    mu_y   = oti_mu.expectation(y)
    print('mu_y =', mu_y.real)
    # Extract sensitivities of central moments wrt rv parameters
    try:
        dev_db1 = mu_y.get_deriv([5])
        print('dev_db1 =', dev_db1)
    except:
        pass
    # input('i')
    
    # variance
    mu2     = oti_mu.central_moment(y, 2)
    print('mu2 =', mu2.real)
    try:
        dmu2_db1 = mu2.get_deriv([5])
        print('dmu2_db1 =', dmu2_db1)
    except:
        pass
    # input('i')
    
    # Third central moment
    mu3 = oti_mu.central_moment(y, 3)
    print('mu3 =', mu3.real)
    try:
        dmu3_db1 = mu3.get_deriv([5])
        print('dmu3_db1 =', dmu3_db1)
    except:
        pass
    # input('i')
    
    # Fourth central moment
    mu4 = oti_mu.central_moment(y, 4)
    print('mu4 =', mu4.real)
    try:
        dmu4_db1 = mu4.get_deriv([5])
        print('dmu4_db1 =', dmu4_db1)
    except:
        pass
    # input('i')
    
    
    # Compute Sobol
    sobol, Vy_hdmr = oti_mu.sobol_indices(y)
    
    shap = oti_mu.shapley_values_v2(y)

    ## remainder term
    l_list = [1,2,3,4]
    ev_rem = np.zeros((tse_order_max, len(l_list)))
    mu2_rem = np.zeros((tse_order_max, len(l_list)))
    mu3_rem = np.zeros((tse_order_max, len(l_list)))
    mu4_rem = np.zeros((tse_order_max, len(l_list)))
    Vi_rem = np.zeros((tse_order_max, len(l_list), n_dim))
    Vij_rem = np.zeros((tse_order_max, len(l_list), len(sobol[2])))
    Vijk_rem = np.zeros((tse_order_max, len(l_list), len(sobol[3])))
    '''
    for n in range(tse_order_max):
        for il, l in enumerate(l_list):
            ev_rem[n,il] = oti_mu.expectation_remainder(y, n, l)
            mu2_rem[n,il] = oti_mu.central_moment_remainder(y, n, l, 2)
            mu3_rem[n,il] = oti_mu.central_moment_remainder(y, n, l, 3)
            mu4_rem[n,il] = oti_mu.central_moment_remainder(y, n, l, 4)
            Si_rem_tmp, Vi_rem_tmp = oti_mu.sobol_indices_remainder(y, n, l)
            Vi_rem[n,il,:] = Vi_rem_tmp[1]
            Vij_rem[n,il,:] = Vi_rem_tmp[2]
            Vijk_rem[n,il,:] = Vi_rem_tmp[3]
    '''
    return mu_y, mu2, mu3, mu4, sobol, Vy_hdmr, shap, ev_rem, mu2_rem, mu3_rem, mu4_rem, Vi_rem, Vij_rem, Vijk_rem


mu_y0, mu20, mu30, mu40, Si0, Vi0, Shap0, ev_rem0, mu2_rem0, mu3_rem0, mu4_rem0, Vi_rem0, Vij_rem0, Vijk_rem0 = polyfd(0)

tse_order_max = 4

h_list = np.logspace(-20, -1, num=20)
nh = len(h_list)
n_dim = 4

mu_y = np.zeros((nh,4))
mu2 = np.zeros((nh,4))
mu3 = np.zeros((nh,4))
mu4 = np.zeros((nh,4))
Si = np.zeros((nh,4,n_dim))
Vi = np.zeros((nh,4,n_dim))
Sij = np.zeros((nh,4,len(Si0[2])))
Vij = np.zeros((nh,4,len(Si0[2])))
Sijk = np.zeros((nh,4,len(Si0[3])))
Vijk = np.zeros((nh,4,len(Si0[3])))
Shap = np.zeros((nh,4,n_dim))
## remainder term
l_list = [1,2,3,4]
ev_rem = np.zeros((nh, 4, tse_order_max, len(l_list)))
mu2_rem = np.zeros((nh, 4, tse_order_max, len(l_list)))
mu3_rem = np.zeros((nh, 4, tse_order_max, len(l_list)))
mu4_rem = np.zeros((nh, 4, tse_order_max, len(l_list)))
Vi_rem = np.zeros((nh, 4, tse_order_max, len(l_list), n_dim))
Vij_rem = np.zeros((nh, 4, tse_order_max, len(l_list), len(Si0[2])))
Vijk_rem = np.zeros((nh, 4, tse_order_max, len(l_list), len(Si0[3])))


mu_y_fd = np.zeros(nh)
mu2_fd = np.zeros(nh)
mu3_fd = np.zeros(nh)
mu4_fd = np.zeros(nh)
Si_fd = np.zeros((nh, n_dim))
Vi_fd = np.zeros((nh, n_dim))
Sij_fd = np.zeros((nh, len(Si0[2])))
Vij_fd = np.zeros((nh, len(Si0[2])))
Sijk_fd = np.zeros((nh, len(Si0[3])))
Vijk_fd = np.zeros((nh, len(Si0[3])))
Shap_fd = np.zeros((nh, n_dim))
ev_rem_fd = np.zeros((nh, tse_order_max, len(l_list)))
mu2_rem_fd = np.zeros((nh, tse_order_max, len(l_list)))
mu3_rem_fd = np.zeros((nh, tse_order_max, len(l_list)))
mu4_rem_fd = np.zeros((nh, tse_order_max, len(l_list)))
Vi_rem_fd = np.zeros((nh, tse_order_max, len(l_list), n_dim))
Vij_rem_fd = np.zeros((nh, tse_order_max, len(l_list), len(Si0[2])))
Vijk_rem_fd = np.zeros((nh, tse_order_max, len(l_list), len(Si0[3])))

for cj, h in enumerate(h_list):
    mu_y[cj,0], mu2[cj,0], mu3[cj,0], mu4[cj,0], Si_tmp, Vi_tmp, Shap[cj,0,:], ev_rem[cj,0,:], mu2_rem[cj,0,:], mu3_rem[cj,0,:], mu4_rem[cj,0,:], Vi_rem[cj,0,:,:], Vij_rem[cj,0,:,:], Vijk_rem[cj,0,:,:] = polyfd(h)
    print(Si_tmp[1][:])
    Si[cj,0,:] = Si_tmp[1][:]
    Vi[cj,0,:] = Vi_tmp[1][:]
    Sij[cj,0,:] = Si_tmp[2][:]
    Vij[cj,0,:] = Vi_tmp[2][:]
    Sijk[cj,0,:] = Si_tmp[3][:]
    Vijk[cj,0,:] = Vi_tmp[3][:]

    mu_y[cj,1], mu2[cj,1], mu3[cj,1], mu4[cj,1], Si_tmp, Vi_tmp, Shap[cj,1,:], ev_rem[cj,1,:], mu2_rem[cj,1,:], mu3_rem[cj,1,:], mu4_rem[cj,1,:], Vi_rem[cj,1,:,:], Vij_rem[cj,1,:,:], Vijk_rem[cj,1,:,:] = polyfd(-h)
    Si[cj,1,:] = Si_tmp[1][:]
    Vi[cj,1,:] = Vi_tmp[1][:]
    Sij[cj,1,:] = Si_tmp[2][:]
    Vij[cj,1,:] = Vi_tmp[2][:]
    Sijk[cj,1,:] = Si_tmp[3][:]
    Vijk[cj,1,:] = Vi_tmp[3][:]

    mu_y[cj,2], mu2[cj,2], mu3[cj,2], mu4[cj,2], Si_tmp, Vi_tmp, Shap[cj,2,:], ev_rem[cj,2,:], mu2_rem[cj,2,:], mu3_rem[cj,2,:], mu4_rem[cj,2,:], Vi_rem[cj,2,:,:], Vij_rem[cj,2,:,:], Vijk_rem[cj,2,:,:] = polyfd(2*h)
    Si[cj,2,:] = Si_tmp[1][:]
    Vi[cj,2,:] = Vi_tmp[1][:]
    Sij[cj,2,:] = Si_tmp[2][:]
    Vij[cj,2,:] = Vi_tmp[2][:]
    Sijk[cj,2,:] = Si_tmp[3][:]
    Vijk[cj,2,:] = Vi_tmp[3][:]

    mu_y[cj,3], mu2[cj,3], mu3[cj,3], mu4[cj,3], Si_tmp, Vi_tmp, Shap[cj,3,:], ev_rem[cj,3,:], mu2_rem[cj,3,:], mu3_rem[cj,3,:], mu4_rem[cj,3,:], Vi_rem[cj,3,:,:], Vij_rem[cj,3,:,:], Vijk_rem[cj,3,:,:] = polyfd(-2*h)
    Si[cj,3,:] = Si_tmp[1][:]
    Vi[cj,3,:] = Vi_tmp[1][:]
    Sij[cj,3,:] = Si_tmp[2][:]
    Vij[cj,3,:] = Vi_tmp[2][:]
    Sijk[cj,3,:] = Si_tmp[3][:]
    Vijk[cj,3,:] = Vi_tmp[3][:]
    

    print()
    print(h)
    print(Vi[cj,0,:])
    print(Vi[cj,1,:])
    input('i')

    mu_y_fd[cj] = (mu_y[cj,0]-mu_y[cj,1])/(2*h)
    mu2_fd[cj] = (mu2[cj,0]-mu2[cj,1])/(2*h)
    mu3_fd[cj] = (-mu3[cj,2] + 8*mu3[cj,0] - 8*mu3[cj,1] + mu3[cj,3])/(12*h) # (mu3[cj,0]-mu3[cj,1])/(2*h)
    mu4_fd[cj] = (mu4[cj,0]-mu4[cj,1])/(2*h)
    Si_fd[cj,:] = [(Si[cj,0,i]-Si[cj,1,i])/(2*h) for i in range(n_dim)]
    Vi_fd[cj,:] = [(Vi[cj,0,i]-Vi[cj,1,i])/(2*h) for i in range(n_dim)]
    Sij_fd[cj,:] = [(Sij[cj,0,i]-Sij[cj,1,i])/(2*h) for i in range(len(Si0[2]))]
    Vij_fd[cj,:] = [(Vij[cj,0,i]-Vij[cj,1,i])/(2*h) for i in range(len(Vi0[2]))]
    Sijk_fd[cj,:] = [(Sijk[cj,0,i]-Sijk[cj,1,i])/(2*h) for i in range(len(Si0[3]))]
    Vijk_fd[cj,:] = [(Vijk[cj,0,i]-Vijk[cj,1,i])/(2*h) for i in range(len(Vi0[3]))]
    Shap_fd[cj,:] = [(Shap[cj,0,i]-Shap[cj,1,i])/(2*h) for i in range(n_dim)]
    ev_rem_fd[cj,:] = [(ev_rem[cj,0,i]-ev_rem[cj,1,i])/(2*h) for i in range(len(l_list))]

    

    










print('nominal values:')
print(mu_y0)
print(mu20)
print(mu30)
print(mu40)
print('Vi0', Vi0)
#print(Shap0)
print()



print('mu_y_fd =', mu_y_fd[13])
print('mu2_fd =', mu2_fd[13])
print('mu3_fd =', mu3_fd[13])
print('mu4_fd =', mu4_fd[13])
print('Si_fd =', Si_fd[13,:])
print('Vi_fd =', Vi_fd)
print('Sij_fd =', Sij_fd[13,:])
print('Vij_fd =', Vij_fd[13,:])
print('Sijk_fd =', Sijk_fd[13,:])
print('Vijk_fd =', Vijk_fd[13,:])
print('Shap_fd =', Shap_fd[13,:])
print('ev_rem_fd =', ev_rem_fd[13,:])

