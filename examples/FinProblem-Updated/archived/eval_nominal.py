# ===========================================================================
import pyoti.sparse as oti
import numpy as np
from fin_problem import * # This imports all in fin_problem.
                          # This includes
                          # - analytic_solution
                          # - "mean" values of variables.
from timeit import default_timer as time
import matplotlib.pyplot as plt
# ===========================================================================


if __name__ == '__main__':

    order = 1   # Order of derivative to be computed.
    t_max = 450 # Total seconds to evaluate.
    dt    = 1.0 # Time increment (Evaluate solution every dt seconds)
    neval = t_max // dt
    
    # These are the perturbations used for all runs in the finite element 
    # analises and other calculations.

    k    = k.real       + oti.e(1,order=order)
    Cp   = Cp.real      + oti.e(2,order=order)
    rho  = rho.real     + oti.e(3,order=order)
    hu   = hu.real      #+ oti.e(4,order=order)
    Tinf = Tinf.real    #+ oti.e(5,order=order)
    Tw   = Tw.real      #+ oti.e(6,order=order)
    b    = b.real       #+ oti.e(7,order=order)
    L    = L.real       #+ oti.e(8,order=order)
    
    T0 = Tinf # Define initial temp.

    t = np.arange(dt,(neval+1)*dt,dt)

    np.save('export_solutions/t_an.npy',t)
    
    x = 0; # Fin's tip
    
    Tanaly = oti.zeros((t.size,1))
    
    i=0    
    start_time = time()
    
    for ti in t:
        
        print("Evaluating time {0}/{1}".format(ti,t_max))

        ti_im = ti + oti.e(4,order=order)
        
        Ti,Th_i,Th_ti,Th_ssi = analytic_solution(x, ti_im, b, d, L, rho, Cp, dt, k,
                                                hu, T0, Tw, Tinf, 
                                                base = oti, nterms = 1000 )
        Tanaly[i,:] = Ti
        
        i+=1
    
    # end for

    end_time = time()

    # Export results. You can export the matso array (matrix of sparse otis)
    oti.save(Tanaly,"export_solutions/T_tip_nominal_n{0:d}.matso".format(order))

    plt.plot(t,Tanaly.real,'C1',alpha=0.5,label='sample')

    # plt.plot(t,Tnominal.real,'C0',label='mean')
    plt.xlabel('time [s]')
    plt.ylabel('Temperature [K]')
    plt.legend()
    plt.title('Nominal values (as in Paper)')
    plt.show()

    t = np.arange(dt,(neval+1)*dt,dt)

    np.save('export_solutions/t_an.npy',t)
    
    x_vec = oti.array(np.linspace(0,b.real,21)) - oti.e(5,order=order) # Fin tip
    
    Tanaly = oti.zeros((x_vec.shape[0],1))
    
    i=0    
    ti_im = t_max + oti.e(4,order=order)
    
    start_time = time()
    
    
    for i in range(x_vec.shape[0]):
        
        print("Evaluating time {0}/{1}".format(ti,t_max))

        xi = x_vec[i, 0]
        
        Ti,Th_i,Th_ti,Th_ssi = analytic_solution(xi, ti_im, b, d, L, rho, Cp, dt, k,
                                                hu, T0, Tw, Tinf, 
                                                base = oti, nterms = 1000 )
        Tanaly[i,:] = Ti
        
        i+=1
    
    # end for

    end_time = time()
    plt.figure()
    plt.plot(x_vec.real,Tanaly.real,'C1',alpha=0.5,label='sample')

    # plt.plot(t,Tnominal.real,'C0',label='mean')
    plt.xlabel('x-coordinate [mm]')
    plt.ylabel('Temperature [K]')
    plt.legend()
    plt.title('Nominal values (as in Paper)')
    plt.show()


# end if 