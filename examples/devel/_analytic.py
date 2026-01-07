# ===========================================================================
import pyoti.sparse as oti
import numpy as np
# ===========================================================================

# ===========================================================================
def analytic_solution(sample,ti,
                      **kwargs):
    '''
    DESCRIPTION: Analytical solution to a thermal fin subjected to a wall 
                 temperature and convective surfaces. This problem appeared
                 on the paper:

    [1] "HYPAD-UQ: A Derivative-Based Uncertainty Quantification Method 
         Using a Hypercomplex Finite Element Method."
         Balcer, M., et al. ASME. J. Verif. Valid. Uncert. June 2023; 
         8(2): 021002. https://doi.org/10.1115/1.4062459

    INPUTS: 
        - x (scalar or array): Location on the fin where to compute the
            temperature. x=0 corresponds to the fin's tip.
            NOTE: If you want temperature at multiple locations, you can 
            define x as an array with the locations. If you also want 
            multiple times, then x must have the same shape as t (see below).
        - t (scalar or array): Times at which temperature solution is 
            required. Time t=0 is start time.
            NOTE: If you want temperature at multiple times, you can 
            define t as an array with the locations. If you also want 
            multiple locations, then x and t must be arrays of equal shape.
        - b (scalar): Fin length.
        - d (scalar): Fin heigth.
        - ρ (scalar): Density of the material.
        - Cp (scalar): Specific heat capacity of the material.
        - dt (scalar <not used>): Time increment.
        - k (scalar): Thermal conductivity of the material.
        - hu (scalar): Convection coefficient
        - T0 (scalar): Initial temperature
        - Tw (scalar): Wall temperature
        - Tinf (scalar): temperature of the fluid (for convective boundary)
        - base (int, optional, default numpy): Base library to access math 
            functions cos(), cosh(), sqrt() and exp()
        - nterms (int, optional, default 100): Number of terms in the 
            summation
    
    OUTPUTS:
        The output of this program is a tuple with 4 parameters:
        ( T, theta, theta_tau, theta_ss )
        - T: Temperature in standard units. Same units as input units. 
            If either x or t are arrays, this will also be an array.
        - thetaθ: Nondimensional temperature solution. This will be an array
            if x or t are arrays.
        - theta_tau: Nondimensional transient component of the solution.
        - theta_ss: Steady state component of the nondimensional solution.
    '''
    
    d =  4.75/1000
    dt = 1
    Tinf  =  283.0
    T0 = Tinf
    x = 0
    t = ti
    base = kwargs['base']
    nterms = 100
    # Adimensionalization of input parameters
    
    if base==np:
        k    = np.array(sample[:,0])
        Cp   = 580
        ρ  = 4430
        hu   = np.array(sample[:,1])
        Tinf = np.array(sample[:,2])
        Tw   = np.array(sample[:,3])
        b    = 51e-3
    else:
        k    = oti.array(sample[0])
        Cp   = oti.array(sample[1])
        ρ  =   oti.array(sample[2])
        hu   = oti.array(sample[3])
        Tinf = oti.array(sample[4])
        Tw   = oti.array(sample[5])
        b    = oti.array(sample[6])
    theta_0 = (T0 - Tinf)/(Tw - Tinf)
    
    X = x/b
    NN = (2*hu*b**2)/(k*d)
    N = base.sqrt(NN)

    tau = t*k/(b**2*ρ*Cp) # Nondimensional time.
    # Start transient solution.
    theta_tau  = 0
    
    theta_ss = base.cosh(N*X)/base.cosh(N)
    
    # Summation term.
    for n in range(1,nterms+1):
        
        lamda_n = ( ( 2 * n - 1) / 2 ) * np.pi
        sign = ( -1 )**( n + 1 )
        
        # It was like this in Juan's paper.
        # t1 = theta_0 / lamda_n - ( lamda_n**2 ) / ( NN + lamda_n**2 ) 
        t1 = theta_0 / lamda_n - ( lamda_n ) / ( NN + lamda_n**2 )
        t2 = base.cos( lamda_n * X )
        t3 = base.exp( -( NN + lamda_n**2 ) * tau )
        
        theta_tau += sign * t1 * t2 * t3
        
    theta_tau *= 2
    
    theta = theta_ss + theta_tau
    T = theta * ( Tw - Tinf ) + Tinf
    
    return T