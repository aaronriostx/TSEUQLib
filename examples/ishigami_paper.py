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

import time

def funct(x): 
    a = 7
    b = 0.1
    return oti.sin(x[0]) + a*oti.sin(x[1])**2 + b*x[2]**4*oti.sin(x[0])


# ---------------------------------------------------------------------------------
t1 = time.time()

# Random variable parameters
rv_pdf_name = np.array(['U']*3) # PDF of each variable
rv_a = np.array([-np.pi]*3) # shape parameter of each variable
rv_b = np.array([np.pi]*3) # scale parameter of each variable

n_dim = len(rv_a) # number of random variables (dimensions)

# store random variable parameters in rv_params as rv_params = [rv_pdf_name, rv_a, rv_b, rv_mean, rv_stdev]
rv_params = rv.rv_mean_from_ab([rv_pdf_name, rv_a, rv_b])


# print(rv_params)

# order of Taylor series expansion
tse_order_max = 6


# random variable(s) with nominal values at the mean and perturbed with OTI imaginary directions
x = [0]*n_dim
for d in range(n_dim):
    x[d] = rv_params[3][d] + oti.e(int(d+1), order=tse_order_max)
#print('x =', x)


# create OTI number
y =  funct(x)
print(' Function evaluated at OTI-perturbed inputs:')
print('tse =', y)
print(" ---------------------------------------------------")
print()


# generate moments of the random variables
rv_moments_order = int(tse_order_max*2) #TODO: generalize for moments past 12th order

# compute central moments of the random variables
# rv moments object
rv_moments = rv.rv_central_moments(rv_params, rv_moments_order)
# compute moments
rv_moments.compute_central_moments()
print(rv_moments.rv_mu)



print(" ---------------------------------------------------")
print()
print(" mean values: ", rv_moments.rv_mean)
print(" central moments: ",)
# Print moment list.
for i in range(len(rv_moments.rv_mu)):
    print(f' - mu_{i+1}:', rv_moments.rv_mu[i])
#print()
#print(" Joint distribution built from the moments: (uncomment lines 79-81)")
# # for i in range(len(rv_mu_joint)):
# #     print(rv_mu_joint[i])
# # # end for
print()
print(" ---------------------------------------------------")
print()



# Build the joint distribution from uncorrelated variables.
rv_mu_joint = uoti.build_rv_joint_moments(rv_moments.rv_mu)


# create oti_moments object
oti_mu = moti.tse_uq(n_dim, rv_mu_joint)


# Expectation
mu_y   = oti_mu.expectation(y)
print('mu_y =', mu_y)

# variance
mu2     = oti_mu.central_moment(y, 2)
print('mu2 =', mu2)

# Third central moment
#mu3 = oti_mu.central_moment(y, 3)
#print('mu3 =', mu3)

# Fourth central moment
#mu4 = oti_mu.central_moment(y, 4)
#print('mu4 =', mu4)

'''
print( " Stats. results from OTI. Compare to Table 2.3 Pg 39 ")
print(f" Expected Value:   {mu_y.real:10.3f}")
print(f" Total Variance:   {mu2.real:10.3f}")
print(f" central_moment 3: {mu3.real:10.3f}")
print(f" central_moment 4: {mu4.real:10.3f}")
print()
'''
print(" ---------------------------------------------------")
print()

# Compute Sobol
sobol, Vy_hdmr, f_hdmr = oti_mu.sobol_indices(y)

Shap = oti_mu.shapley_values_v2(y)

t2 = time.time()
print("    time =", t2-t1)

print('sobol =')
for i in range(len(sobol)):
    print(f" -> Order --  {i}:")
    print('real sob:   ', [sobol[i][j].real for j in range(len(sobol[i]))] )
    print('real Vi :   ', [Vy_hdmr[i][j].real for j in range(len(Vy_hdmr[i]))] )
Vi = [Vy_hdmr[1][j] for j in range(len(Vy_hdmr[1]))]
Vij = [Vy_hdmr[2][j] for j in range(len(Vy_hdmr[2]))]
Vijk = [Vy_hdmr[3][j] for j in range(len(Vy_hdmr[3]))]
Si = [sobol[1][j] for j in range(len(Vy_hdmr[1]))]
Sij = [sobol[2][j] for j in range(len(Vy_hdmr[2]))]
Sijk = [sobol[3][j] for j in range(len(Vy_hdmr[3]))]


# analytical solutions of metrics (precomputed for the example in Section 3.1)
a = 7
b = 0.1
ev_an  = a/2
mu2_an = 0.5 + (a**2/8) + ((b*np.pi**4)/5) + ((b**2*np.pi**8)/18)

Vi_an = [(0.5 + ((b*np.pi**4)/5) + ((b**2*np.pi**8)/50)), (a**2/8), 0+1e-16]
Vij_an = [0+1e-16, ((8*b**2*np.pi**8)/225), 0+1e-16]
Vijk_an = [0+1e-16]

Si_an = np.array(Vi_an)/mu2_an
Sij_an = np.array(Vij_an)/mu2_an
Sijk_an = np.array(Vijk_an)/mu2_an

Shap_an = [0]

# relative errors
mu_err = 1 - (ev_an/mu_y)**-1
mu2_err = 1 - (mu2_an/mu2)**-1
Vi_err = [1 - (Vi_an[i]/Vi[i])**-1 for i in range(len(Vi))]
Si_err = [1 - (Si_an[i]/Si[i])**-1 for i in range(len(Si))]
Vij_err = [1 - (Vij_an[i]/Vij[i])**-1 for i in range(len(Vij))]
Sij_err = [1 - (Sij_an[i]/Sij[i])**-1 for i in range(len(Sij))]
Vijk_err = [1 - (Vijk_an[i]/Vijk[i])**-1 for i in range(len(Vijk))]
Sijk_err = [1 - (Sijk_an[i]/Sijk[i])**-1 for i in range(len(Sijk))]
#Shap_err = [1 - (Shap_db1[i]/Shap_fd[i]) for i in range(len(Shap))]

print()
print('mu_err =', mu_err)
print('mu2_err =', mu2_err)
print('Vi_err =', Vi_err)
print('Si_err =', Si_err)
print('Vij_err =', Vij_err)
print('Sij_err =', Sij_err)
print('Vijk_err =', Vijk_err)
print('Sijk_err =', Sijk_err)
#print('Shap_err =', Shap_err)



# input('i')


## remainder term
m_list = [2]
rem_orders = np.arange(1,tse_order_max-1)
y_ev_rem = np.zeros((len(rem_orders), len(m_list)))
y_mu2_rem = np.zeros((len(rem_orders), len(m_list)))
y_Vi_rem = np.zeros((len(rem_orders), len(m_list), n_dim))
y_Vij_rem = np.zeros((len(rem_orders), len(m_list), len(sobol[2])))
y_Vijk_rem = np.zeros((len(rem_orders), len(m_list), len(sobol[3])))

t1 = time.time()

for ni,n in enumerate(rem_orders):
    for im, m in enumerate(m_list):
        y_ev_rem[ni,im] = oti_mu.expectation_remainder(y, n, m)
        y_mu2_rem[ni,im] = oti_mu.central_moment_remainder(y, n, m, 2)
        Si_rem, Vi_rem = oti_mu.sobol_indices_remainder(y, n, m)
        y_Vi_rem[ni,im,:] = Vi_rem[1]
        y_Vij_rem[ni,im,:] = Vi_rem[2]
        y_Vijk_rem[ni,im,:] = Vi_rem[3]

t2 = time.time()
print("    time to compute remainder terms =", t2-t1)

'''
print('y_ev_rem =', y_ev_rem)
print('y_mu2_rem =', y_mu2_rem)
print('y_Vi_rem =', y_Vi_rem)
print('y_Vij_rem =', y_Vij_rem)
print('y_Vijk_rem =', y_Vijk_rem)
'''


Ef_rel_err = np.zeros((len(rem_orders), len(m_list)))
mu2_rel_err = np.zeros((len(rem_orders), len(m_list)))
# Vi_rel_err = np.zeros_like(y_Si_rem)
Vi_rel_err = np.zeros((len(rem_orders), len(m_list), n_dim))
Vij_rel_err = np.zeros((len(rem_orders), len(m_list), len(sobol[2])))
Vijk_rel_err = np.zeros((len(rem_orders), len(m_list), len(sobol[3])))

# true errors of taylor series
ev_err = np.zeros(len(rem_orders))
mu2_err = np.zeros(len(rem_orders))
Vi_err = np.zeros((len(rem_orders),len(Vi)))
Vij_err = np.zeros((len(rem_orders),len(Vij)))
Vijk_err = np.zeros((len(rem_orders),len(Vijk)))

t1 = time.time()
for ni, n in enumerate(rem_orders):
    ev_err[ni] = ev_an - oti_mu.expectation(y.truncate_order(n+1))
    mu2_err[ni] = mu2_an - oti_mu.central_moment(y.truncate_order(n+1),2)
        
    Si_rem, Vi_rem, f_hdmr = oti_mu.sobol_indices(y.truncate_order(n+1))
    Vi_err[ni,:] = (np.array(Vi_an) - np.array(Vi_rem[1]))
    Vij_err[ni,:] = np.array(Vij_an) - np.array(Vi_rem[2])
    Vijk_err[ni,:] = np.array(Vijk_an) - np.array(Vi_rem[3])

t2 = time.time()
print("    time to compute metrics =", t2-t1)

for ni, n in enumerate(rem_orders):
    for ck, k in enumerate(m_list):
        Ef_rel_err[ni,ck] = 1-(y_ev_rem[ni,ck]/(ev_err[ni]))
        mu2_rel_err[ni,ck] = 1-(y_mu2_rem[ni,ck]/(mu2_err[ni]))
        Vi_rel_err[ni,ck,:] = 1-(np.array(y_Vi_rem[ni,ck,:])/(Vi_err[ni,:]))
        Vij_rel_err[ni,ck,:] = 1-(np.array(y_Vij_rem[ni,ck,:])/(Vij_err[ni,:]))
        Vijk_rel_err[ni,ck,:] = 1-(np.array(y_Vijk_rem[ni,ck,:])/(Vijk_err[ni,:]))


'''
print()
print('Ef_rel_err =', Ef_rel_err)
print('mu2_rel_err =', mu2_rel_err)
print('Vi_rel_err =', Vi_rel_err)
print('Vij_rel_err =', Vij_rel_err)
print('Vijk_rel_err =', Vijk_rel_err)
'''
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# plot remainder error 

# m=2
plt.figure(figsize=(9,9))
lw=5
plt.semilogy(rem_orders, np.abs(y_ev_rem[:,0]), label=r'$E[f]$', lw=lw)
plt.semilogy(rem_orders, np.abs(y_mu2_rem[:,0]), label=r'$V[f]$', lw=lw)
plt.semilogy(rem_orders, np.abs(y_Vi_rem[:,0,0]), label=r'$V_{{1}}$', lw=lw)
plt.semilogy(rem_orders, np.abs(y_Vi_rem[:,0,1]), label=r'$V_{{2}}$', lw=lw)
plt.semilogy(rem_orders, np.abs(y_Vij_rem[:,0,1]), label=r'$V_{{13}}$', lw=lw)
plt.xlabel(r'$n$', size=20)
plt.ylabel(r'$\left|\varepsilon_{{2}}\left[\xi\left[Y_{{n}}\left(\boldsymbol{{x}}\right)\right]\right]\right|$',size=20)
ax = plt.gca()
ax.tick_params('x', labelsize=16)
ax.tick_params('y', labelsize=16)
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(np.arange(min(rem_orders), max(rem_orders)+1, 1))
plt.legend(fontsize=20, loc='upper right')
plt.axhline(0, color='k', lw=2)
plt.savefig('plots/rem_m2.png', format='PNG', transparent=True, dpi=300)
#plt.show()


plt.figure(figsize=(9,9))
lw=5
plt.semilogy(rem_orders, np.abs(Ef_rel_err[:,0]), label=r'$E[f]$', lw=lw)
plt.semilogy(rem_orders, np.abs(mu2_rel_err[:,0]), label=r'$V[f]$', lw=lw)
plt.semilogy(rem_orders, np.abs(Vi_rel_err[:,0,0]), label=r'$V_{{1}}$', lw=lw)
plt.semilogy(rem_orders, np.abs(Vi_rel_err[:,0,1]), label=r'$V_{{2}}$', lw=lw)
plt.semilogy(rem_orders, np.abs(Vij_rel_err[:,0,1]), label=r'$V_{{13}}$', lw=lw)
plt.xlabel(r'$n$', size=24)
plt.ylabel(r'Absolute Relative Error of $\varepsilon_{{2}}\left[\xi\left[Y_{{n}}\left(\boldsymbol{{x}}\right)\right]\right]$', size=24)
ax = plt.gca()
ax.tick_params('x', labelsize=16)
ax.tick_params('y', labelsize=16)
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(np.arange(min(rem_orders), max(rem_orders)+1, 1))
plt.legend(fontsize=20, loc='upper right')
plt.axhline(0, color='k', lw=5)
plt.savefig('plots/rel_err_rem_m2.png', format='PNG', transparent=True, dpi=300)

# true error
plt.figure(figsize=(9,9))
lw=5
plt.semilogy(rem_orders, np.abs(ev_err), label=r'$E[f]$', lw=lw)
plt.semilogy(rem_orders, np.abs(mu2_err), label=r'$V[f]$', lw=lw)
plt.semilogy(rem_orders, np.abs(Vi_err[:,0]), label=r'$V_{{1}}$', lw=lw)
plt.semilogy(rem_orders, np.abs(Vi_err[:,1]), label=r'$V_{{2}}$', lw=lw)
plt.semilogy(rem_orders, np.abs(Vij_err[:,1]), label=r'$V_{{13}}$', lw=lw)
plt.xlabel(r'$n$', size=24)
plt.ylabel(r'Absolute Error of $Y_{{n}}(\boldsymbol{{x}})$',size=24)
ax = plt.gca()
ax.tick_params('x', labelsize=16)
ax.tick_params('y', labelsize=16)
ax.set_xticks(np.arange(min(rem_orders), max(rem_orders)+1, 1))
plt.legend(fontsize=20, loc='upper right')
plt.axhline(0, color='k', lw=2)
plt.savefig('plots/true_err.png', format='PNG', transparent=True, dpi=300)
#plt.show()


