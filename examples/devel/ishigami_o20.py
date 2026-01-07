import pyoti.sparse as oti
from pyoti.sparse import sin
oti.set_printoptions(terms_print=-1)

import numpy as np

import matplotlib.pyplot as plt

import pyvista as pv
import sys

sys.path.append('../')

# tseuqlib
import tseuqlib as tuq
import tseuqlib.rv_moments as rv
import tseuqlib.oti_moments as moti


import time

def ishigami(x,a,b, alg=oti):
    """
    Ishigami function. See https://uqworld.org/t/ishigami-function/55

    INPUTS:
    - x: List of inputs.
    - a: Ishigami function param.
    - b: Ishigami function parameter.
    - alg: (Optional), algebra backend to use.
    """
    
    return alg.sin(x[0]) + a*(alg.sin(x[1]))**2 + b*(x[2]**4)*alg.sin(x[0])

# end function

def mgf_uniform( a, b, e, order):
    '''
    Moment Generating function for Uniform Distribution.
    - x: List of inputs.
    - a: Ishigami function param.
    - b: Ishigami function parameter.
    - alg: (Optional), algebra backend to use.

    '''
    # Notice important things:
    # 1. Real part is zero.
    # 2. The order is increased by 1 because of l'Hopital's rule.
    t = oti.e(e, order=order+1) 

    # Evaluate the Moment generating function 
    num = (oti.exp(t*b)-oti.exp(t*a))
    den = (t*(b-a))
    
    # The next line essentially applies L'Hopital's rule.
    MGF = num.extract_im(e)/den.extract_im(e)
    
    return MGF

# end function



# Ishigami function parameters:
# ( see https://uqworld.org/t/ishigami-function/55 )
a = 7
b = 0.05




# Build distributions:

# Random variable parameters
rv_pdf_name = np.array(['U','U','U']) # PDF of each variable
rv_a = np.array([-np.pi,-np.pi,-np.pi]) # shape parameter of each variable
rv_b = np.array([ np.pi, np.pi, np.pi]) # scale parameter of each variable
n_dim = len(rv_a) # number of random variables (dimensions)

# store random variable parameters in rv_params as rv_params = [rv_pdf_name, rv_a, rv_b, rv_mean, rv_stdev]
rv_params = rv.rv_mean_from_ab([rv_pdf_name, rv_a, rv_b])


# Upto 50th order second order moments.
#
tse_order_max = 25

# random variable(s) with nominal values at the mean and perturbed with OTI imaginary directions
x = [0]*n_dim
for d in range(n_dim):
    x[d] = rv_params[3][d] + oti.e(int(d+1), order=tse_order_max)


# generate moments of the random variables
rv_moments_order = 2*tse_order_max +1#TODO: generalize for moments past 12th order

# compute second through twelfth central moments of the random variables
# pdfInp, lb, ub, meanVal, mx02, mx03, mx04, mx05, mx06, mx07, mx08, mx09, mx10, mx11, mx12
# rv_moments_structure = \
# rv.rv_central_moments(rv_params, rv_moments_order)


mgf = mgf_uniform( rv_a[0], rv_b[0], 1, rv_moments_order)
rv_mu = []

for i in range(1,rv_moments_order):

    rv_mu.append(
        n_dim*[mgf.get_deriv([[1,i]])]
    )



print(len(rv_mu))


# Build the joint distribution from uncorrelated variables.
rv_mu_joint = moti.build_rv_joint_moments(rv_mu)

t_start = time.time()

y    = ishigami(x,a,b)

mu_y = moti.oti_expectation(y,rv_mu_joint)
Vy   = moti.oti_central_moment(y,2,rv_mu_joint)
# Compute Sobol
sobol = moti.oti_s9obol_indices(y,rv_mu_joint )

t_end = time.time()


mu_y_an = a/2
Vy_an   = (a**2)/8 + b*(np.pi**4)/5 + b**2*(np.pi**8)/18 + 1/2


sobol_an   = [[1]]
sobol_an.append([ (0.5*(1 + b*(np.pi**4)/5)**2)/Vy_an,
                  ((a**2)/8)/Vy_an,
                  0.0
                   ])

print( " Stats. results from OTI. Compare to Ishigami ")
print(f" Expected Value:   {mu_y:10.6e} {mu_y_an:10.6e} ")
print(f" Total Variance:   {Vy:10.6e} {Vy_an:10.6e}  ")
print(f" Sobol Indices: ")
for i in range(len(sobol)):
    print(f" -> Order --  {i} ",sobol[i])

    if i<len(sobol_an):
        print(f" -> Order(an) {i} ",sobol_an[i])

        
print(f" CPU time:   {(t_end-t_start):10.6f} s (OTI based)")

print()
# Sampling based error estimation method.

# t = np.linspace(rv_a[0],rv_b[0],100)
# X1, X2, X3 = np.meshgrid(t,t,t)

# # Measuring the error from the TSE sample using the two last components.
# err_est = y.get_order_im(tse_order_max) + y.get_order_im(tse_order_max-1)

# TSEERR = np.abs(err_est.rom_eval_object([1,2,3],[X1,X2,X3]))

# err = np.sum(TSEERR)


# print(f' Total Error estimate from TSE:{err}')

print()

# 
nsamples = int(1e7)

X = [0]*n_dim
for i in range(n_dim):
    X[i] = np.random.uniform( rv_a[i], rv_b[i], nsamples )
# end for


t_start = time.time()
Y = ishigami( X, a, b, alg=np )
mu_y_mcs = np.mean(Y)
V_y_mcs = np.var(Y)
t_end = time.time()
print( " Stats. results from MCS to Ishigami ")
print(f" Expected Value:   {mu_y_mcs:10.6e}")
print(f" Total Variance:   {V_y_mcs:10.6e}")
print(f" CPU time:   {(t_end-t_start):10.6f} s (MCS)")
# nbins = 1000

# for i in range(n_dim):
    
#   plt.figure()
#   plt.hist( X[i].ravel(), nbins, density=True )
#   plt.xlabel( f'x{i+1}' )

# # end for 

# plt.figure()
# plt.hist(Y.ravel(),nbins,density=True)
# plt.xlabel('y')
# plt.show()










# Ytse = y.rom_eval_object([1,2,3],X)

# # Compute actual error
# ERR = np.abs( Y - Ytse )
# err = np.sum( ERR )

# print(f' Actual error from Ishigami vs TSE:{err}')

# grid = pv.ImageData()

# grid.dimensions = X1.shape
# grid.origin = (-np.pi,-np.pi,-np.pi,)
# grid.spacing = (1,1,1)

# grid.point_data['TSE_error'] = TSEERR.flatten(order='F')
# grid.point_data['error'] = ERR.flatten(order='F')


# grid.save('error.vtk')

# grid.plot()
# # plt.figure()
