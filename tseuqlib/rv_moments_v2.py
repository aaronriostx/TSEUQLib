#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:52:45 2024

@author: sam
"""

from .oti_util import gen_OTI_indices
import pyoti.core as coti
from scipy.special import factorial2
import numpy as np
import math
import scipy.stats as stats

def get_pdf_params(rv_pdf_name, rv_mean, rv_stdev):
    
    """Compute the parameters for probability distributions.

    This function computes the parameters (rv_a and rv_b) for various probability distributions
    based on the distribution names (rv_pdf_name), mean values (rv_mean), and standard deviations
    (rv_stdev).

    Parameters
    ----------
    rv_pdf_name : list of str
        List of distribution names for each variable.
        Possible values:
            - 'N' for normal (Gaussian) distribution.
            - 'U' for uniform distribution.
            - 'LN' for log-normal distribution.
            - 'B' for beta distribution.
            - 'T' for triangle (symmetric) distribution.
    rv_mean : array_like
        Mean values of the variables.
    rv_stdev : array_like
        Standard deviations of the variables.

    Returns
    -------
    rv_a : ndarray
        Generic PDF shape parameter (a) for each variable.
    rv_b : ndarray
        Generic PDF scale parameter (b) for each variable.

    Notes
    -----
    For each distribution type, the function computes the appropriate parameters rv_a and rv_b
    based on the mean and standard deviation of the variables.

    Examples
    --------
    >>> rv_pdf_name = ['N', 'U', 'LN', 'B', 'T']
    >>> rv_mean = [0, 0, 1, 0.5, 0]
    >>> rv_stdev = [1, 1, 0.5, 0.1, 0.2]
    >>> get_pdf_params(rv_pdf_name, rv_mean, rv_stdev)
    (array([...]), array([...]))
    """
    
    # convert mean and standard deviation to inputs for distribution
    nVar = len(rv_mean)
    rv_a = np.zeros(nVar)
    rv_b = np.zeros(nVar)
    
    c_var=0
    for i_pdf_name in rv_pdf_name:
        if i_pdf_name=='N': # normal (Gaussian)
            rv_a[c_var] = rv_mean[c_var]
            rv_b[c_var] = rv_stdev[c_var]
            
        elif i_pdf_name=='U': # uniform
            rv_a[c_var] = rv_mean[c_var] - np.sqrt(3)*rv_stdev[c_var] # 0.5*(x+y)-a
            rv_b[c_var] = rv_mean[c_var] + np.sqrt(3)*rv_stdev[c_var] # (1/sqrt(12))*(y-x)-b

        elif i_pdf_name=='LN': # log-normal
            rv_a[c_var] = np.log(rv_mean[c_var]**2/(np.sqrt(rv_mean[c_var]**2 + rv_stdev[c_var]**2))) # exp(x+0.5*y^2)-a # rv_mean[c_var]  # 0.5*(2*np.log(rv_mean[c_var])-np.log((rv_mean[c_var]**2+rv_stdev[c_var]**2)/rv_mean[c_var]**2)) # exp(x+0.5*y^2)-a
            rv_b[c_var] = np.log((rv_mean[c_var]**2+rv_stdev[c_var]**2)/rv_mean[c_var]**2) # sqrt((exp(y^2)-1)*exp(2*x+y^2))-b # rv_stdev[c_var] # np.abs(np.sqrt(np.log((rv_mean[c_var]**2+rv_stdev[c_var]**2)/rv_mean[c_var]**2))) # sqrt((exp(y^2)-1)*exp(2*x+y^2))-b
        
        elif i_pdf_name=='B': # beta 
            rv_a[c_var] = (rv_mean[c_var]**2-rv_mean[c_var]**3-rv_mean[c_var]*rv_stdev[c_var])/rv_stdev[c_var]
            rv_b[c_var] = ((-1+rv_mean[c_var])*(-rv_mean[c_var]+rv_mean[c_var]**2+rv_stdev[c_var]))/rv_stdev[c_var]

        elif i_pdf_name=='T': # triangle (symmetric) 
            rv_a[c_var] = rv_mean[c_var] - np.sqrt(6)*rv_stdev[c_var]
            ub = rv_mean[c_var] + np.sqrt(6)*rv_stdev[c_var]
            rv_b[c_var] = (ub - rv_a[c_var])

        c_var+=1

    return rv_a, rv_b


def build_joint(mu_ind):
    """
    Build the joint distribution moment structure from 
    independent (uncorrelated) random parameters.
    
    INPUTS:
    - mu_ind: List of size m, with the central moments of each input distribution.
          mu_ind[j][i] is the (j+1)'th order moment of the (i+1)'th variable.
    OUTPUT:
    - List of lists with the structure of the central moments.

    mu_joint = [
                 [mu1.1, mu1.2], #< First order central moments ( Must be all zero)
                 [mu2.11, mu2.12, mu2.22], #< Second order central moments. (covars. of the joint distr.)
                 [mu3.111, mu3.112, mu3.122, mu3.322], #< Third order ctr. mnts. of the joint distribution
                 ...
                ]

    """

    # Direction helper from OTI.
    h     = coti.get_dHelp()
    
    order = len(mu_ind) # Maximum order
    nvars = len(mu_ind[0])# number of variables
    
    mu = []
    
    for ordi in range(1,order+1):
        
        # Get number of terms per order and imdirs.
        nimord = coti.ndir_order(nvars,ordi)
        mu_i = [1.0]*nimord
        
        for idx in range(nimord):
            
            # Get bases and exponents of directions.
            bases, exps = h.get_base_exp(idx,ordi)

            for i in range(bases.size):
            
                mu_i[idx]*=mu_ind[exps[i]-1][bases[i]-1]
            
            # end for 

        # end for 

        mu.append(mu_i)
        
    # end for 
    
    return mu

def generate_moments(dists,shape, scale,means, order_of_moments):
    
    moments = []
    dim = len(dists)
    raw_moments = [[] for i in range(order_of_moments+1)]
    raw_moments[0] = [1 for i in range(dim)]
    for j in range(1, order_of_moments+1):
        moment = []
        for i in range(0,len(dists)):
            if dists[i] == 'U':
                if j%2 == 1:
                    moment.append(0)
                else:
                    a = shape[i]
                    b = scale[i]
                    moment.append(((a-b)**j + (b - a)**j)/(2**(j+1)*(j+1)))
            elif dists[i] == 'N':
                if j%2 == 1:
                    moment.append(0)
                else:
                    sigma = scale[i]
                    moment.append(sigma**j*factorial2((j-1), exact = False))
            elif dists[i] == 'LN':
                    mu = shape[i]
                    sigma_2 = scale[i]
                    raw_moments[j].append(np.exp(j*mu + .5*j**2*sigma_2))
                    central_moment = 0
                    for k in range(0,j+1):
                        central_moment = central_moment + math.comb(j,k)*(-1)**k*raw_moments[j- k][i]*means[i]**k
                    moment.append(central_moment)
            elif dists[i] == 'T':
                    loc = shape[i]
                    scal = scale[i]
                    c = .5
                    dist = stats.triang(c, loc = loc, scale = scal)
                    raw_moments[j].append(dist.moment(j))
                    central_moment = 0
                    for k in range(0,j+1):
                        central_moment = central_moment + math.comb(j,k)*(-1)**k*raw_moments[j- k][i]*means[i]**k
                    moment.append(central_moment)
                
        moments.append(moment)
    
    joint_central_mom =  build_joint(moments)
    return joint_central_mom