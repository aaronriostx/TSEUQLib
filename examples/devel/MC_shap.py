from modules.lhs import create_latin_hypercube_samples as lhs
import numpy as np
import pyoti.sparse as oti
import pyoti.core as coti


def MC_shap(n_samples, dim):
    def func(x,alg=oti):
        y = 1
        for i in range(0,x.shape[1]):
            y = y + ((.001*i+1)*x[:,i])
        return y
    n = n_samples
    d = dim
    x = lhs(n,d).T
    x = 2*(x) - .5
    y = lhs(n,d).T
    y = 2*(y) - .5
    pm = np.argsort(lhs(n,d).T, axis =1)
    
    z = x
    fz1 = func(z)
    fx = fz1
    phi1 = np.zeros((1,d))
    phi2 = np.zeros((1,d))
    
    
    for j in range(0,d):
        ind = np.zeros(pm.shape, dtype = 'bool')
        for i in range(0,d):
            test = np.where(pm[:,j] ==i)[0]
            ind[test, i] = True
        z[ind] = y[ind]
        fz2 = func(z)
        fmarg = ((fx-fz1/2-fz2/2)*(fz1-fz2)).T
        phi1 = phi1 + fmarg@ind/n
        phi2 = phi2 + fmarg**2@ind/n;
        fz1 = fz2;

    
    
    return phi1
    
    
     
