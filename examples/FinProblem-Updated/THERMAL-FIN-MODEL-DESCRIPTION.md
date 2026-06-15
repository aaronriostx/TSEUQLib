# Transient thermal fin problem
A 1D thermal fin problem is presented in this subdirectory. The non-dimensionalized temperature response, $\theta(X, \tau)$ is defined as

$$
\theta(X, \tau) = \frac{T(X, \tau) - T_{\infty}}{T_{W} - T_{\infty}} = \theta(X, \tau)_{SS} + \theta(X, \tau)_{\tau}
$$

where $X$ is the normalized distance along the thermal fin and $\tau$ is the time constant. $\theta(X, \tau)_{SS}$ is the steady-state thermal response and $\theta(X, \tau)_{\tau}$ is the transient thermal response. $T(X, \tau)$ is the full temperature response and $T_{W}$ and $T_{\infty}$ are the temperature at the wall and environment temperature, respectively. 

$X$ is defined as

$$
X = x/b
$$

where $x$ is the distance from the insulated fin tip $(x=0)$ and $b$ is the total fin length. The time constant, $\tau$ is defined as 

$$
\tau = \frac{tk}{b^{2} \rho C_{p}}
$$

where $t$ is time, $k$ is thermal conductivity, $\rho$ is density and $C_{p}$ is specific heat.

The steady-state thermal response is calculated as 

$$
\theta(X, \tau)_{SS} = \frac{\cosh \left(\omega X\right)}{\cosh \left(\omega\right)}
$$

where $\omega^{2}$ depends on convection coefficient, $h_{U}$, fin thickness, $\delta$ and fin width, $L$, shown below.

$$
\omega^{2} = \frac{2h_{U}b^{2}}{k \delta L}
$$

The transient thermal response is calculated as a truncated series shown below

$$
\theta(X, \tau)_{\tau} = 2 \sum_{j = 1}^{\infty} \left(- 1\right)^{j + 1} \left(\frac{\theta_{0}}{\lambda_{j}} - \frac{\lambda_{j}}{\omega^{2} + \lambda_{j}^{2}}\right) \text{cos} \left(\lambda_{j} X\right) e^{- \left(\omega^{2} + \lambda_{j}^{2}\right) \tau}
$$

$\lambda_{j}$ terms are calculated simply as: 

$$
\lambda_{j} = \pi(2j-1)/2
$$

The series is truncated to $j=100$. $j$ is simply an index defined as $j=1,2,3,...,100$. The initial non-dimensionalized temperature, $\theta_{0}$, is 

$$
\theta_{0} = \frac{T_{0} - T_{\infty}}{T_{W} - T_{\infty}}
$$

where $T_{0}$ is the initial temperature of the fin at $t=0$. For this model, it is assumed that $T_{0} = T_{\infty}$.

## Model parameter definitions

In the `fin_params.py` file, the parameters of this model are defined. In this model, we are interested in the thermal response at the fin tip through time, i.e., $\theta(X=0, \tau)$ and $T(X=0, \tau)$. Fin width and thickness are fixed constants ($\delta=4.75 \times 10^{-3} \thinspace \text{m}, \thinspace L = 100 \times 10^{-3} \thinspace \text{m}$). All other model parameters are either defined as random or epistemic variables.