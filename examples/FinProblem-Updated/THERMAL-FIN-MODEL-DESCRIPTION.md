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

## Quantifying epistemic contributions via affine and Taylor models (`TSE_mixed_models.py`)

Standard interval arithmetic collapses the result to a single bound $[\underline{T}, \overline{T}]$, discarding information about which uncertainty source drove that width. Affine arithmetic and Taylor models retain symbolic noise variables that carry each source through the calculation.

### Affine model

Each centered epistemic deviation $\delta y_i \in [-\Delta y_i, +\Delta y_i]$ is associated with an independent symbolic noise variable $\varepsilon_i \in [-1, +1]$:

$$
\delta y_i = \Delta y_i \cdot \varepsilon_i
$$

Substituting into the 1st-order TSE gives the **affine form** of $\mathbb{E}[T]$:

$$
\hat{\mathbb{E}}[T(\tau)] = T_0(\tau) + \underbrace{\frac{\partial T}{\partial T_\infty}\bigg|_{\text{nom}} \Delta T_\infty}_{c_1(\tau)} \varepsilon_1 + \underbrace{\frac{\partial T}{\partial T_W}\bigg|_{\text{nom}} \Delta T_W}_{c_2(\tau)} \varepsilon_2 + \underbrace{\frac{\partial T}{\partial b}\bigg|_{\text{nom}} \Delta b}_{c_3(\tau)} \varepsilon_3
$$

where $T_0(\tau) = T(\boldsymbol{\mu}_X, \mathbf{m}_y)\big|_\tau$ is the nominal temperature history. The coefficients $c_i(\tau)$ are time-varying and encode the sensitivity of $\mathbb{E}[T]$ to each source scaled by its half-width.

Converting the affine form to a standard interval recovers the same bound as before, since $\varepsilon_i \in [-1,+1]$:

$$
\mathbb{E}[T(\tau)] \in \left[T_0(\tau) - \sum_i |c_i(\tau)|,\;\; T_0(\tau) + \sum_i |c_i(\tau)|\right]
$$

The key advantage is that each $|c_i(\tau)|$ is the individual half-width contribution of source $i$, making it possible to rank sources at any time step.

### Fractional contribution

The fractional contribution of each source at time $\tau$ is:

$$
S_i(\tau) = \frac{|c_i(\tau)|}{\displaystyle\sum_j |c_j(\tau)|}
$$

with $\sum_i S_i(\tau) = 1$. Plotting $S_i(\tau)$ over time reveals how dominance shifts between sources as the fin evolves from transient to steady state.

### Taylor model

A **Taylor model** of order $p$ is a pair $(p(\boldsymbol{\varepsilon}),\, R)$ where $p$ is a multivariate polynomial in the noise variables and $R$ is an interval that rigorously bounds the remainder from truncating the Taylor series:

$$
\mathbb{E}[T(\tau)] \in p(\boldsymbol{\varepsilon})\big|_{\boldsymbol{\varepsilon}\in[-1,1]^n} + R(\tau)
$$

For the 1st-order expansion, the polynomial part is the affine form above and the remainder captures 2nd-order and higher contributions:

$$
R(\tau) = \left[\min_{\boldsymbol{s} \in \{-1,+1\}^n} r(\boldsymbol{s},\tau),\;\; \max_{\boldsymbol{s} \in \{-1,+1\}^n} r(\boldsymbol{s},\tau)\right]
$$

where

$$
r(\boldsymbol{s},\tau) = T\!\left(\boldsymbol{\mu}_X,\, \mathbf{m}_y + \boldsymbol{s} \odot \boldsymbol{\Delta y}\right)\bigg|_\tau - \left(T_0(\tau) + \sum_i s_i\, c_i(\tau)\right)
$$

is the pointwise approximation error evaluated at each of the $2^n$ corners of the epistemic box. The Taylor model enclosure is then:

$$
\mathbb{E}[T(\tau)] \in \left[T_0(\tau) - \sum_i |c_i(\tau)| + \underline{R}(\tau),\;\; T_0(\tau) + \sum_i |c_i(\tau)| + \overline{R}(\tau)\right]
$$

The remainder $R$ is zero for a model that is exactly linear in the epistemic parameters; a non-zero $R$ indicates nonlinearity not captured by the 1st-order expansion.

## 2nd-order TSE for aleatory uncertainty (`TSE_2nd_order.py`)

This script builds a 2nd-order Taylor expansion of $T(\mathbf{X}, \mathbf{y})$ about the nominal point $(\boldsymbol{\mu}_X, \mathbf{m}_y)$, with the epistemic parameters $\mathbf{y} = [T_\infty, T_W, b]$ **fixed** at their midpoints $\mathbf{m}_y$ and only the stochastic parameters $\mathbf{X} = [k, C_p, \rho, h_U]$ treated as random:

$$
T(\mathbf{X}, \mathbf{m}_y) \approx T_0 + \sum_i f_i\, \delta X_i + \frac{1}{2}\sum_i \sum_j H_{ij}\, \delta X_i\, \delta X_j
$$

where $T_0 = T(\boldsymbol{\mu}_X, \mathbf{m}_y)$, $\delta X_i = X_i - \mu_{X_i}$, and $f_i = \partial T/\partial X_i$, $H_{ij} = \partial^2 T/\partial X_i \partial X_j$ are the gradient and Hessian evaluated at the nominal point via central finite differences (relative step $h_i = 10^{-3}\mu_{X_i}$; 33 model evaluations total for 4 variables).

### Expectation

Assuming $X_i$ are independent with $\mathbb{E}[\delta X_i] = 0$, $\mathrm{Var}[\delta X_i] = \sigma_i^2$, the linear term vanishes in expectation and only the diagonal curvature survives (off-diagonal terms vanish since $\mathbb{E}[\delta X_i \delta X_j] = 0$ for $i \neq j$):

$$
\mathbb{E}[T] \approx T_0 + \frac{1}{2}\sum_i H_{ii}\, \sigma_i^2
$$

### Variance

Writing $T - \mathbb{E}[T] = \underbrace{\sum_i f_i\, \delta X_i}_{L} + \underbrace{\left(\frac{1}{2}\sum_{i,j} H_{ij}\, \delta X_i \delta X_j - \frac{1}{2}\sum_i H_{ii}\sigma_i^2\right)}_{Q - \mathbb{E}[Q]}$, and using that odd central moments of a symmetric distribution (e.g. normal) vanish under independence — which makes $\mathrm{Cov}(L, Q) = 0$ — gives:

$$
\mathrm{Var}[T] \approx \underbrace{\sum_i f_i^2\, \sigma_i^2}_{\text{1st-order (delta method)}} + \underbrace{\frac{1}{2}\sum_i \sum_j H_{ij}^2\, \sigma_i^2 \sigma_j^2}_{\text{2nd-order correction}}
$$

The 2nd-order correction is the additional variance contributed by curvature that a 1st-order (linear) expansion misses entirely. The script cross-checks both formulas against a fresh 20,000-sample Monte Carlo draw at the same nominal point; $\mathbb{E}[T]$ matches to within noise, while $\mathrm{Var}[T]$ runs a few percent low at steady state, since 3rd-order-and-higher curvature is still truncated.

## 2nd-order TSE with interval-propagated epistemic bounds (`TSE_2nd_order_bounds.py`)

The script above fixes $\mathbf{y}$ at its midpoint, discarding the epistemic interval information entirely. Re-deriving the aleatory expansion at many sampled points across the epistemic box and numerically taking the min/max (an earlier, superseded version of this script) is cheaper than brute-force Monte Carlo, but it is still a *sampling* procedure — it does not give a closed-form bound and can miss the true extremum between grid points. The corrected approach builds **one** combined 2nd-order expansion in $(\mathbf{X}, \mathbf{y})$ together, then propagates the epistemic uncertainty through that single expansion algebraically.

### Combined expansion

About $(\boldsymbol{\mu}_X, \mathbf{m}_y)$, with $\delta X_i = X_i - \mu_{X_i}$ (aleatory) and $\delta y_j = y_j - m_{y_j}$ (epistemic):

$$
T \approx T_0 + \sum_i f_i \delta X_i + \sum_j g_j \delta y_j + \frac{1}{2}\sum_{i,i'} H_{ii'}\, \delta X_i \delta X_{i'} + \sum_{i,j} C_{ij}\, \delta X_i \delta y_j + \frac{1}{2}\sum_{j,j'} G_{jj'}\, \delta y_j \delta y_{j'}
$$

where $g_j = \partial T/\partial y_j$, $G_{jj'} = \partial^2 T/\partial y_j \partial y_{j'}$ (epistemic gradient/Hessian) and $C_{ij} = \partial^2 T/\partial X_i \partial y_j$ (mixed Hessian) are new terms not needed above. All derivatives are evaluated **once**, via central finite differences on the combined 7-variable vector $(\mathbf{X}, \mathbf{y})$ — 99 evaluations total ($1 + 2(7) + 4\binom{7}{2}$) — with no re-evaluation of the physics model afterward.

### Conditional expectation and variance

Taking $\mathbb{E}_{\mathbf{X}}[\cdot]$ over the aleatory randomness only, and leaving $\delta y_j$ symbolic (the cross term $\sum_{i,j} C_{ij}\delta X_i \delta y_j$ vanishes in expectation since it is linear in $\delta X_i$):

$$
\mathbb{E}_{\mathbf{X}}[T](\boldsymbol{\delta y}) = \underbrace{\left[T_0 + \frac{1}{2}\sum_i H_{ii}\sigma_i^2\right]}_{\text{constant}} + \sum_j g_j \delta y_j + \frac{1}{2}\sum_{j,j'} G_{jj'}\, \delta y_j \delta y_{j'}
$$

Following the same variance derivation as above, but now the effective linear coefficient on $\delta X_i$ picks up a $\boldsymbol{\delta y}$-dependent correction from the mixed Hessian:

$$
\mathrm{Var}_{\mathbf{X}}[T](\boldsymbol{\delta y}) = \sum_i \left[f_i + \sum_j C_{ij}\, \delta y_j\right]^2 \sigma_i^2 \;+\; \frac{1}{2}\sum_{i,i'} H_{ii'}^2\, \sigma_i^2 \sigma_{i'}^2
$$

Both expressions are now closed-form polynomials **purely in the epistemic deviations** $\boldsymbol{\delta y}$ — the expensive physics model no longer appears past the one-time derivative computation above.

### The $h_U$ standard deviation as an epistemic interval

$h_U$'s standard deviation, $\sigma_{h_U}$, is itself only known to lie in an interval $[\underline{\sigma}_{h_U}, \overline{\sigma}_{h_U}]$ (`STD_hU` in `fin_params_mixed.py`). It never appears as an argument to $T$, so it needs no derivative — it enters the formulas above algebraically, wherever $\sigma_i^2$ is indexed at $i = h_U$, as the interval $\sigma_{h_U}^2 \in [\underline{\sigma}_{h_U}^2, \overline{\sigma}_{h_U}^2]$.

### Interval-arithmetic propagation

Each epistemic deviation is bounded by a centered interval of half-width $\Delta y_j = (\overline{y}_j - \underline{y}_j)/2$: $\delta y_j \in [-\Delta y_j, +\Delta y_j]$. The `Interval` class in the script implements:

- **Addition**: $[\underline{a},\overline{a}] + [\underline{b},\overline{b}] = [\underline{a}+\underline{b},\, \overline{a}+\overline{b}]$ — exact.
- **Multiplication**: $[\underline{a},\overline{a}] \cdot [\underline{b},\overline{b}] = [\min(\underline{a}\underline{b},\underline{a}\overline{b},\overline{a}\underline{b},\overline{a}\overline{b}),\, \max(\cdots)]$ — exact for **independent** intervals, but naively squaring an interval against itself this way overestimates the true range (e.g. $[-a,a]\cdot[-a,a]$ naively gives $[-a^2,a^2]$, when the true range of $x^2$ for $x\in[-a,a]$ is $[0,a^2]$). This is the classic interval-arithmetic **dependency problem**: a variable that appears more than once in an expression is treated as if each occurrence were independent.
- **Square**: a dedicated operator computing the tight range of $x^2$ over $[\underline{a},\overline{a}]$ (accounting for whether $0$ lies inside the interval), used for every $\delta y_j^2$ and for the squared bracket $\left[f_i + \sum_j C_{ij}\delta y_j\right]^2$ in the variance formula. Because the bracket itself is an *affine* (linear) function of independent intervals, its range from ordinary interval addition/scaling is already exact, so squaring it with the dedicated operator gives the exact range of the squared bracket too.

Substituting $\delta y_j$, and $\sigma_{h_U}^2$ where relevant, as intervals into the two boxed expressions above and evaluating with this arithmetic gives lower/upper bounds on $\mathbb{E}[T(0,\tau)]$ and $\mathrm{Var}[T(0,\tau)]$ at every time step, with no sweep over the epistemic space.

### Caveat: bounds are conservative, not necessarily tight

Interval arithmetic is exact for a *single* affine combination of independent intervals (including a bracket squared, as noted above), but **summing several such individually-exact terms that share a common $\delta y_j$** (e.g. two different bracket-squared terms both involving $\delta T_\infty$) can still overestimate the true joint range, because each term's own extremum may not occur at the same point in the epistemic box. This is the interval-arithmetic **wrapping effect**, and it means the resulting $[\underline{T},\overline{T}]$ is a valid but possibly loose outer enclosure of the 2nd-order polynomial's true range — not a certified tight bound. Affine arithmetic (propagating a shared symbolic noise variable per $\delta y_j$, as in `TSE_mixed_models.py`, so correlated terms cancel correctly on summation) or Taylor models would tighten this further.

Separately, and independently of the propagation method: any gap between these bounds and a brute-force Monte Carlo sweep of the epistemic grid (`double_loop_MCS_data.npz`) reflects **2nd-order truncation error** in the underlying polynomial (most visibly in $\mathrm{Var}[T]$, which depends on curvature the expansion doesn't fully capture) — not a flaw in the interval arithmetic itself.