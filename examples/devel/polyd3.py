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
print(" verification example in Appendix A.   ")
print(" ---------------------------------------------------")

# Extracted from Table 2.1. (Pg. 35)


extra_bases = [5]

# Random variable parameters
rv_pdf_name = np.array(['K']*4) # PDF of each variable
rv_a = np.array([1+1*oti.e(5,order=3),1+0*oti.e(5,order=1),2,2]) # shape parameter of each variable
rv_b = np.array([3+0*oti.e(5,order=1),8,6,4]) # scale parameter of each variable

# +oti.e(5, order=1)
n_dim = len(rv_a) # number of random variables (dimensions)

# store random variable parameters in rv_params as rv_params = [rv_pdf_name, rv_a, rv_b, rv_mean, rv_stdev]
rv_params = rv.rv_mean_from_ab([rv_pdf_name, rv_a, rv_b])


print(rv_params)

# order of Taylor series expansion
tse_order_max = 3


# random variable(s) with nominal values at the mean and perturbed with OTI imaginary directions
x = [0]*n_dim
for d in range(n_dim):
    x[d] = rv_params[3][d] + oti.e(int(d+1), order=tse_order_max)
#x[0]+=oti.e(5,order=1)
print('x =', x)


# OTI evaluation:
y =  funct(x)
print(' Function evaluated at OTI-perturbed inputs:')
print('tse =', y)
print(" ---------------------------------------------------")
print()


# generate moments of the random variables
rv_moments_order = int(tse_order_max*4) #TODO: generalize for moments past 12th order

# compute central moments of the random variables
# rv moments object
rv_moments = rv.rv_central_moments(rv_params, rv_moments_order)
# compute moments
rv_moments.compute_central_moments()
print(rv_moments.rv_mu)



print(" ---------------------------------------------------")
print(" A total of 4 variables are considered with a       ")
print(" Kumaraswamy distribution. ")
print()
print(" mean values: ", rv_moments.rv_mean)
print(" central moments: ",)
# Print moment list.
for i in range(len(rv_moments.rv_mu)):
    '''
    try:
        rv_moments.rv_mu[i].append(rv_moments.rv_mu[i][0].get_deriv([5]))
    except:
        rv_moments.rv_mu[i].append(0)
    rv_moments.rv_mu[i][0] = rv_moments.rv_mu[i][0].real
    #rv_moments.rv_mu[i].append(1)
    '''
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



# Build the joint distribution from uncorrelated variables.
rv_mu_joint = uoti.build_rv_joint_moments(rv_moments.rv_mu)


# create oti_moments object
oti_mu = moti.tse_uq(n_dim, rv_mu_joint, extra_bases=extra_bases)


# non-central moment
# mu2c     = oti_mu.central_moment(y, 2, Ey=2)
# print('mu2c =', mu2c.real)
# print(float(563263737029933/3832653825))
# input('i')

# mum1     = oti_mu.central_moment(y, -1)
# print('mum1 =', mum1.real)

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

print( " Stats. results from OTI. Compare to Table 2.3 Pg 39 ")
print(f" Expected Value:   {mu_y.real:10.3f}")
print(f" Total Variance:   {mu2.real:10.3f}")
print(f" central_moment 3: {mu3.real:10.3f}")
print(f" central_moment 4: {mu4.real:10.3f}")
print()

print(" ---------------------------------------------------")
print()

# Compute Sobol
sobol, Vy_hdmr = oti_mu.sobol_indices(y)

Shap = [0+oti.e(5,order=1)]*n_dim # oti_mu.shapley_values_v2(y)

print('sobol =')
for i in range(len(sobol)):
    print(f" -> Order --  {i}:")
    print('real sob:   ', [sobol[i][j].real for j in range(len(sobol[i]))] )
    print('real Vi :   ', [Vy_hdmr[i][j].real for j in range(len(Vy_hdmr[i]))] )
    try:
        print('derivs sob: ', [sobol[i][j].get_deriv([5]) for j in range(len(sobol[i]))] )
        print('derivs Vi : ', [Vy_hdmr[i][j].get_deriv([5]) for j in range(len(Vy_hdmr[i]))] )
        Vi_db1 = [Vy_hdmr[1][j].get_deriv([5]) for j in range(len(Vy_hdmr[1]))]
        Si_db1 = [sobol[1][j].get_deriv([5]) for j in range(len(Vy_hdmr[1]))]
        Vij_db1 = [Vy_hdmr[2][j].get_deriv([5]) for j in range(len(Vy_hdmr[2]))]
        Sij_db1 = [sobol[2][j].get_deriv([5]) for j in range(len(Vy_hdmr[2]))]
        Vijk_db1 = [Vy_hdmr[3][j].get_deriv([5]) for j in range(len(Vy_hdmr[3]))]
        Sijk_db1 = [sobol[3][j].get_deriv([5]) for j in range(len(Vy_hdmr[3]))]
    except:
        pass

# store derivatives of sobol indices and partial variances
try:
    c=1e-16
    Vi_db1 = [Vy_hdmr[1][j].get_deriv([5])+c for j in range(len(Vy_hdmr[1]))]
    Si_db1 = [sobol[1][j].get_deriv([5])+c for j in range(len(Vy_hdmr[1]))]
    Vij_db1 = [Vy_hdmr[2][j].get_deriv([5])+c for j in range(len(Vy_hdmr[2]))]
    Sij_db1 = [sobol[2][j].get_deriv([5])+c for j in range(len(Vy_hdmr[2]))]
    Vijk_db1 = [Vy_hdmr[3][j].get_deriv([5])+c for j in range(len(Vy_hdmr[3]))]
    Sijk_db1 = [sobol[3][j].get_deriv([5])+c for j in range(len(Vy_hdmr[3]))]
    Shap_db1 = [Shap[j].get_deriv([5])+c for j in range(n_dim)]
except:
    pass

# finite difference sensitivities
deriv_type = 'a1'
if deriv_type=='a1':
    mu_y_fd = 72.03697293789446
    mu2_fd = 17291.546864726115
    mu3_fd = 7260391.322585444
    mu4_fd = 9109645385.742188
    Si_fd = [ 0.022708204552857,  -0.0007268544793826, -0.0047243756129411, -0.013707572854571 ]
    Vi_fd = [ 2111.951737333584,    534.7335900296457,  3873.116083923378,  10021.638336183969 ]
    Sij_fd = [ 0.0001392993996519,  0.0010223964578594, -0.0003270162255136,  0.0026387944520451, -0.000850389077732,  -0.0061815827109846]
    Vij_fd = [ 14.557769532075326, 106.71700991338184,   15.486760247540587, 275.6951016635867,    39.79127939146565,  293.73806569310545 ]
    Sijk_fd = [ 1.4320872605995665e-06,  3.6832292078759713e-06,  2.7535855472399055e-05, -2.3555166335249150e-05]
    Vijk_fd = [0.1807532895581332, 0.4648849361021945, 3.475484078308,     0.   +c             ]
    Shap_fd = [ 2311.8103854358196,   569.8667882825248,  4082.30578614166,   10327.564014005475 ]
elif deriv_type=='b1':
    mu_y_fd= -17.32428953005183
    mu2_fd = -4936.039415042615
    mu3_fd = -2.3466392679450414e+06
    mu4_fd = -2.9273285350799561e+09
    Si_fd =   [-0.0182985894145449,  0.0007231069562574,  0.0050817282742921,  0.0135852659122193]
    Vi_fd =   [-1213.225361880177,    -126.01158118741296,  -912.7920720857219,  -2361.169765208615  ]
    Sij_fd =  [-0.0001294794846969, -0.0009488869126204,  0.0001097473957415, -0.0024519328685468,  0.0002848829748492,  0.0020756037617808]
    Vij_fd =  [   -8.789713735524174,    -64.4011946064893,       -3.5738677510721573,  -166.44094165485512,      -9.182602781265814,    -67.7857071877952   ]
    Sijk_fd = [-1.6741740677431952e-06, -4.3058596950831547e-06, -3.2190646937597326e-05,  6.7240502242224033e-06]
    Vijk_fd = [ -0.1169580111926027,  -0.3008079005883957,  -2.2488426485445245,   0.+c                ]
elif deriv_type=='a2':
    mu_y_fd = 82.5031594331449
    mu2_fd = 21639.735787175596
    mu3_fd = 9948877.866069477
    mu4_fd = 12517096643.447876
    Si_fd = [-0.0033283006212192,  0.0574006754280876, -0.0146339737550072, -0.0397469246404825]
    Vi_fd = [ 1003.2125601355801,  3681.103220287696,   4396.565946080955,  11374.746445653727 ]
    Sij_fd = [ 0.0006765701573205, -0.0007404693481636,  0.0029757993497744, -0.0019240135092774,  0.007689627454753,  -0.008452238110801 ]
    Vij_fd = [ 44.16083653424607,  29.21532328059584, 194.2290894874077,   75.06458246098191, 501.96058211327,    330.60775990634284]
    Sijk_fd = [ 8.8863041299792260e-06,  2.2854958468707066e-05, -4.9744843051073399e-05,  1.0125267032418898e-04]
    Vijk_fd = [0.5926378060516768, 1.524223371962563,  0.+c,                 6.752656618314035 ]
elif deriv_type=='Na1':
    mu_y_fd = 19175.99976877682
    mu2_fd = 27236805114.746094
    mu3_fd = 1.0840780373333334e+17
    mu4_fd = 6.351473389810483e+23
    Si_fd = [1.4986544991102413e-05, 8.2426357406184536e-04, 1.5336307224167456e-03, 1.2820201844565560e-03]
    Vi_fd = [2.5132032394409180e+08, 4.6929714965820312e+09, 7.9626239013671875e+09, 6.7398911285400391e+09]
    Sij_fd = [ 6.2707098628239644e-05,  6.0201075895160994e-05, -4.4216921790685149e-05,  6.4224240075433370e-05,  1.5000084507832412e-04, -5.8836130412132093e-04]
    Vij_fd = [8.9579518437385559e+07, 1.3996800065040588e+08, 2.2394879722595215e+09, 1.2192767858505249e+08, 1.9508428573608398e+09, 3.0481920242309570e+09]
    Sijk_fd = [-0.0001343245382545, -0.0001170115978937, -0.0001828306213023, -0.0029252899408361]
    Vijk_fd = [0.+c, 0.+c, 0.+c, 0.+c]
    Shap_fd = [4.2705780029296875e+08, 6.8329264831542969e+09, 1.0676447906494141e+10, 9.3003721618652344e+09]
elif deriv_type=='Nb1':
    mu_y_fd = 2160.0001491606236
    mu2_fd = 9439044799.804688
    mu3_fd = 3.752348394666667e+16
    mu4_fd = 2.8660417161068544e+23
    Si_fd = [ 0.0059589162076934, -0.001966098117645,  -0.0035782685148256, -0.0029513194665931]
    Vi_fd = [2.2845715117454529e+09, 7.9526702880859375e+08, 1.2426046752929688e+09, 1.0824468231201172e+09]
    Sij_fd = [ 0.0015879954165868,  0.0028094500901035, -0.0021180618370309,  0.0023449821468305, -0.0017797351092863, -0.0030659599165039]
    Vij_fd = [6.091407346725464e+08, 1.077753598690033e+09, 0.000000000000000e+00 +c, 8.995553267002106e+08, 0.000000000000000e+00 +c, 0.000000000000000e+00 +c]
    Sijk_fd = [ 0.0011669591183129,  0.0010165510555722,  0.0015883610204014, -0.0010137731973336]
    Vijk_fd = [4.4789759933948517e+08, 3.9016857564449310e+08, 6.0963839888572693e+08, 0.0000000000000000e+00+c]
    Shap_fd = [4.0603649902343750e+09, 1.3791934204101562e+09, 2.1339942932128906e+09, 1.8654939270019531e+09]

# relative errors
# try:
mu_err = 1 - (dev_db1/mu_y_fd)
mu2_err = 1 - (dmu2_db1/mu2_fd)
mu3_err = 1 - (dmu3_db1/mu3_fd)
mu4_err = 1 - (dmu4_db1/mu4_fd)
Vi_err = [1 - (Vi_db1[i]/Vi_fd[i]) for i in range(len(Vi_fd))]
Si_err = [1 - (Si_db1[i]/Si_fd[i]) for i in range(len(Si_fd))]
Vij_err = [1 - (Vij_db1[i]/Vij_fd[i]) for i in range(len(Vij_fd))]
Sij_err = [1 - (Sij_db1[i]/Sij_fd[i]) for i in range(len(Sij_fd))]
Vijk_err = [1 - (Vijk_db1[i]/Vijk_fd[i]) for i in range(len(Vijk_fd))]
Sijk_err = [1 - (Sijk_db1[i]/Sijk_fd[i]) for i in range(len(Sijk_fd))]
#Shap_err = [1 - (Shap_db1[i]/Shap_fd[i]) for i in range(len(Shap_fd))]

print()
print('mu_err =', mu_err)
print('mu2_err =', mu2_err)
print('mu3_err =', mu3_err)
print('mu4_err =', mu4_err)
print('Vi_err =', Vi_err)
print('Si_err =', Si_err)
print('Vij_err =', Vij_err)
print('Sij_err =', Sij_err)
print('Vijk_err =', Vijk_err)
print('Sijk_err =', Sijk_err)
#print('Shap_err =', Shap_err)
#except: pass



input('i')

# print(y)

## remainder term
l_list = [1,2,3,4]
y_ev_rem = np.zeros((tse_order_max, len(l_list)))
y_mu2_rem = np.zeros((tse_order_max, len(l_list)))
y_mu3_rem = np.zeros((tse_order_max, len(l_list)))
y_mu4_rem = np.zeros((tse_order_max, len(l_list)))
y_Vi_rem = np.zeros((tse_order_max, len(l_list), n_dim))
y_Vij_rem = np.zeros((tse_order_max, len(l_list), len(sobol[2])))
y_Vijk_rem = np.zeros((tse_order_max, len(l_list), len(sobol[3])))

for n in range(tse_order_max):
    for il, l in enumerate(l_list):
        y_ev_rem[n,il] = oti_mu.expectation_remainder(y, n, l)
        y_mu2_rem[n,il] = oti_mu.central_moment_remainder(y, n, l, 2)
        y_mu3_rem[n,il] = oti_mu.central_moment_remainder(y, n, l, 3)
        y_mu4_rem[n,il] = oti_mu.central_moment_remainder(y, n, l, 4)
        Si_rem, Vi_rem = oti_mu.sobol_indices_remainder(y, n, l)
        y_Vi_rem[n,il,:] = Vi_rem[1]
        y_Vij_rem[n,il,:] = Vi_rem[2]
        y_Vijk_rem[n,il,:] = Vi_rem[3]



print('y_ev_rem =', y_ev_rem)
print('y_mu2_rem =', y_mu2_rem)
print('y_mu3_rem =', y_mu3_rem)
print('y_mu4_rem =', y_mu4_rem)
print('y_Vi_rem =', y_Vi_rem)
print('y_Vij_rem =', y_Vij_rem)
print('y_Vijk_rem =', y_Vijk_rem)


Ef =   209949406/675675 # 310.725431605432
# Ef = 14417135.6000000 #  226308.250000000
mu2f = 400885535046969013/7761123995625
mu3f = 16706254.3029196
mu4f = 15315376784.8466
Vif = [2804.97817054685, 1709.49893144332, 12298.6662155643, 32051.5121562420]
Vijf = [21.9932412121491, 161.03052423422196, 416.39363489248126, 96.71944211738446, 250.0764012058571, 1831.249642229268]
Vijkf = [0.31897640073080424, 0.820385167155699, 6.133207244650748, 3.634493050980268]


'''
print()

n = 5+1
k = 4

Yn = y.truncate_order(n)
print('Yn =', Yn)
Ynk = y.truncate_order(n+k)
print('Ynk =', Ynk)

E_rem = oti_mu.expectation(Ynk) - oti_mu.expectation(Yn)
print('E_rem =', E_rem)


E_err_true = 1- (Ef / oti_mu.expectation(Yn))
print('E_err_true =', E_err_true)

print(E_rem/Ef)


input('i')
'''

Ef_rel_err = np.zeros((tse_order_max, len(l_list)))
mu2_rel_err = np.zeros((tse_order_max, len(l_list)))
mu3_rel_err = np.zeros((tse_order_max, len(l_list)))
mu4_rel_err = np.zeros((tse_order_max, len(l_list)))
# Vi_rel_err = np.zeros_like(y_Si_rem)
Vi_rel_err = np.zeros((tse_order_max, len(l_list), n_dim))
Vij_rel_err = np.zeros((tse_order_max, len(l_list), len(sobol[2])))
Vijk_rel_err = np.zeros((tse_order_max, len(l_list), len(sobol[3])))

for n in range(tse_order_max):    
    for ck, k in enumerate(l_list):
        Ef_rel_err[n,ck] = 1-(y_ev_rem[n,ck]/(Ef - oti_mu.expectation(y.truncate_order(n+1))))
        mu2_rel_err[n,ck] = 1-(y_mu2_rem[n,ck]/(mu2f - oti_mu.central_moment(y.truncate_order(n+1),2)))
        mu3_rel_err[n,ck] = 1-(y_mu3_rem[n,ck]/(mu3f - oti_mu.central_moment(y.truncate_order(n+1),3)))
        mu4_rel_err[n,ck] = 1-(y_mu4_rem[n,ck]/(mu4f - oti_mu.central_moment(y.truncate_order(n+1),4)))
        Si_rem, Vi_rem = oti_mu.sobol_indices(y.truncate_order(n+1))
        Vi_rel_err[n,ck,:] = 1-(np.array(y_Vi_rem[n,ck,:])/(np.array(Vif) - np.array(Vi_rem[1])))
        Vij_rel_err[n,ck,:] = 1-(np.array(y_Vij_rem[n,ck,:])/(np.array(Vijf) - np.array(Vi_rem[2])))
        Vijk_rel_err[n,ck,:] = 1-(np.array(y_Vijk_rem[n,ck,:])/(np.array(Vijkf) - np.array(Vi_rem[3])))

print()
print('Ef_rel_err =', Ef_rel_err)
print('mu2_rel_err =', mu2_rel_err)
print('mu3_rel_err =', mu3_rel_err)
print('mu4_rel_err =', mu4_rel_err)
print('Vi_rel_err =', Vi_rel_err)
print('Vij_rel_err =', Vij_rel_err)
print('Vijk_rel_err =', Vijk_rel_err)



input('i')

