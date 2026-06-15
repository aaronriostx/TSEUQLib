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

or

$$
\omega = \sqrt{\frac{2h_{U}b^{2}}{k \delta L}}
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

In the `fin_params.py` file, the parameters of this model are defined. In this model, we are interested in the thermal response at the insulated fin tip through time, i.e., $T(X=0, \tau)$. Fin width and thickness are fixed constants ($\delta=4.75 \times 10^{-3} \thinspace \text{m}, \thinspace L = 100 \times 10^{-3} \thinspace \text{m}$). All other model parameters are either defined as random or epistemic variables.

## Quantities-of-interest (QOIs)
 In particular, we are interested in computing the expectation of the insulated fin tip thermal response, $\mathbb{E}[T(0, \tau)]$ as well as central moments (variance, skewness, kurtosis).

$$
\mu_{n}[T(0, \tau)]=\mathbb{E}\bigg[ \big( T(0, \tau) - \mathbb{E}[(T(0, \tau)] \big)^{n} \bigg]
$$

 Ideally, we want to plot expectation and central moments through time using different UQ methods.

## 1st-order TSE for mixed stochastic-epistemic uncertainty (`TSE_mixed.py`)

In the mixed uncertainty case, model parameters are partitioned into two sets:

- **Stochastic parameters** $\mathbf{X} = [k,\, C_p,\, \rho,\, h_U]$, treated as random variables with known normal distributions and means $\boldsymbol{\mu}_X$.
- **Epistemic parameters** $\mathbf{y} = [T_\infty,\, T_W,\, b,\, T_0]$, known only to lie within intervals $[\underline{y}_i,\, \overline{y}_i]$. Since $T_0 = T_\infty$ by assumption, $T_0$ is coupled to $T_\infty$ and is not an independent variable.

### Nominal point

The expansion is performed about the nominal point $(\boldsymbol{\mu}_X, \mathbf{m}_y)$, where $\mathbf{m}_y$ is the vector of interval midpoints:

$$
m_{y_i} = \frac{\underline{y}_i + \overline{y}_i}{2}
$$

### 1st-order Taylor series expansion of $\mathbb{E}[T]$

Expanding $T(\mathbf{X}, \mathbf{y})$ to first order about the nominal point:

$$
T(\mathbf{X}, \mathbf{y}) \approx T(\boldsymbol{\mu}_X, \mathbf{m}_y) + \nabla_{\mathbf{X}} T\big|_{\text{nom}} \cdot (\mathbf{X} - \boldsymbol{\mu}_X) + \nabla_{\mathbf{y}} T\big|_{\text{nom}} \cdot (\mathbf{y} - \mathbf{m}_y)
$$

Taking the expectation over $\mathbf{X}$ (with $\mathbf{y}$ fixed), the stochastic correction terms vanish since $\mathbb{E}[\mathbf{X} - \boldsymbol{\mu}_X] = \mathbf{0}$:

$$
\mathbb{E}_{\mathbf{X}}\left[T(\mathbf{X}, \mathbf{y})\right] \approx T(\boldsymbol{\mu}_X, \mathbf{m}_y) + \sum_{i} \frac{\partial T}{\partial y_i}\bigg|_{\text{nom}} (y_i - m_{y_i})
$$

### Interval arithmetic for epistemic bounds

Each epistemic deviation $y_i - m_{y_i}$ is bounded by a centered interval of half-width (radius) $\Delta y_i$:

$$
\delta y_i = y_i - m_{y_i} \;\in\; [-\Delta y_i,\; +\Delta y_i], \qquad \Delta y_i = \frac{\overline{y}_i - \underline{y}_i}{2}
$$

Multiplying each centered interval by its partial derivative and summing using interval arithmetic:

$$
\sum_{i} \frac{\partial T}{\partial y_i}\bigg|_{\text{nom}} \cdot [-\Delta y_i,\, +\Delta y_i] = \left[-\sum_{i} \left|\frac{\partial T}{\partial y_i}\bigg|_{\text{nom}}\right| \Delta y_i,\;\; +\sum_{i} \left|\frac{\partial T}{\partial y_i}\bigg|_{\text{nom}}\right| \Delta y_i \right]
$$

This yields lower and upper bounds on $\mathbb{E}[T(0, \tau)]$ at each time step:

$$
\mathbb{E}[T(0, \tau)] \in T(\boldsymbol{\mu}_X, \mathbf{m}_y)\big|_\tau \;\pm\; \sum_{i} \left|\frac{\partial T}{\partial y_i}\bigg|_{\text{nom}}\right| \Delta y_i
$$

The partial derivatives $\partial T / \partial y_i$ are functions of time and are evaluated numerically via numerical differentiation at the nominal point.