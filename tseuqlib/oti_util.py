import pyoti.sparse as oti 
import pyoti.core   as coti

import numpy as np


# @profile
def build_rv_joint_moments(mu_ind):
    """
    @brief Build the joint distribution moment structure from 
    independent (uncorrelated) random parameters.
    
    
    @param[in] mu_ind: List of size m, with the central moments of each input distribution.
          mu_ind[j][i] is the (j+1)'th order moment of the (i+1)'th variable.
    
    @param[out] List of lists with the structure of the central moments.

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
            
        mu.append(mu_i)
        
    # end for 
    
    return mu


def gen_OTI_basis_vector(nvars, order):
    
    '''
    Generate list of bases in an OTI number.
    '''
    
    dH = coti.get_dHelp()
    nterms_total = coti.ndir_total(nvars,order)
    
    X=[0]*nterms_total
    X[0] = oti.number(1)
    
    k=1
    for ordi in range(1,order+1):
        nterms = coti.ndir_order(nvars,ordi)
        for idx in range(nterms):
            deriv = dH.get_fulldir(idx,ordi)
            X[k] = oti.e(deriv)
            k+=1
    return X


def gen_OTI_indices(nvars, order):
    
    '''
    Generate list of lists that contain indices of bases in an OTI number
    '''
    
    dH = coti.get_dHelp()
    nterms_total = coti.ndir_total(nvars,order)
    
    ind = [0]*order
    
    for ordi in range(1,order+1):
        nterms = coti.ndir_order(nvars,ordi)
        i = [0]*nterms
        for idx in range(nterms):
            i[idx] = convert_index_to_exponent_form(dH.get_fulldir(idx,ordi))
        ind[ordi-1] = i
    return ind

def convert_index_to_exponent_form(lst):
    
    compressed = []
    current_num = None
    count = 0
    
    for num in lst:
        if num != current_num:
            if current_num is not None:
                compressed.append([current_num, count])
            current_num = num
            count = 1
        else:
            count += 1

    if current_num is not None:
        compressed.append([current_num, count])

    return compressed

def tse_remainder(y, n, k):

    '''
    Estimate of the remainder term in a Taylor series expansion

    Args:
        - y (oti number): Taylor series expansion (OTI number)
        - n (int): order of Taylor series expansion to be evaluated
        - k (int): number of additional orders

    Returns:
        - remainder (oti number): remainder term = y_{n+k} - y_{n}
    '''
    
    order = y.order
    # check if remainder term can be computed
    if k<0:
        raise ValueError('k must be greater than 0 in oti_util.tse_error()')
    if k>order:
        raise ValueError('n+k is greater than order of Taylor series in oti_util.tse_error()')
    
    return y.truncate_order(n+k) - y.truncate_order(n)


