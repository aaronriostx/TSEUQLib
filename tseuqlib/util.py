import numpy as np


def create_symbolic_variables(nVar):
    """
    Create a list of symbolic variables.
    
    Parameters
    ----------
    nVar : int
        Number of variables.
    
    Returns
    -------
    list
        List of symbolic variables.
    """
    import sympy as sy
    
    xSym = [] 
    for i in range(nVar):
        xSym.append(sy.Symbol('x' + str(i+1)))
    
    return xSym


def extract_derivatives(oti_number, nVar, tse_order_max):
    
    '''
    Extract derivatives from an OTI number, up to fifth-order
    
    Parameters
    ----------
    oti_number : list
        List of OTI numbers.
    nVar : int
        Number of variables.
    tse_order_max : int
        Order of Taylor series expansion.
    
    Returns:
        f0: function evaluated at the expansion point
        S1: First-order derivatives
        S2: Second-order derivatives
        S3: Third-order derivatives
        S4: Fourth-order derivatives
        S5: Fifth-order derivatives
    '''
    
    deriv_ind = np.arange(0,nVar)
    nPoints = np.size(oti_number,0)
    
    f0 = np.zeros(nPoints)
    S1 = np.zeros([nPoints,nVar])
    S2 = np.zeros([nPoints,nVar,nVar])
    S3 = np.zeros([nPoints,nVar,nVar,nVar])
    S4 = np.zeros([nPoints,nVar,nVar,nVar,nVar])
    S5 = np.zeros([nPoints,nVar,nVar,nVar,nVar,nVar])
    
    for i in range(nPoints):
        f0[i] = oti_number[i].get_deriv([0]) 
        c_n1=0
        for i_n1 in deriv_ind:
            S1[i,c_n1] = oti_number[i].get_deriv([int(i_n1)+1])
            if tse_order_max>=2:
                c_n2=0
                for i_n2 in deriv_ind:
                    S2[i,c_n1,c_n2] = oti_number[i].get_deriv([int(i_n1)+1, int(i_n2)+1])
                    
                    if tse_order_max>=3:
                        c_n3=0
                        for i_n3 in deriv_ind:
                            S3[i,c_n1,c_n2,c_n3] = oti_number[i].get_deriv([int(i_n1)+1, int(i_n2)+1, int(i_n3)+1])
                            
                            if tse_order_max>=4:
                                c_n4=0
                                for i_n4 in deriv_ind:
                                    S4[i,c_n1,c_n2,c_n3,c_n4] = oti_number[i].get_deriv([int(i_n1)+1, int(i_n2)+1, int(i_n3)+1, int(i_n4)+1])
                                    
                                    if tse_order_max>=5:
                                        c_n5=0
                                        for i_n5 in deriv_ind:
                                            S5[i,c_n1,c_n2,c_n3,c_n4,c_n5] = oti_number[i].get_deriv([int(i_n1)+1, int(i_n2)+1, int(i_n3)+1, int(i_n4)+1, int(i_n5)+1])
                                            
                                            c_n5+=1
                                    c_n4+=1
                            c_n3+=1
                    c_n2+=1
            c_n1+=1
    
    
    return (f0, S1, S2, S3, S4, S5)