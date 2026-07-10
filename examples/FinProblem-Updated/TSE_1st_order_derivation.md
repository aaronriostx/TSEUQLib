# 1st-order Taylor series expansion of the thermal fin problem

This note states the 1st-order Taylor series expansion (TSE) of the fin
temperature history $T(0, \tau)$ (written as $T$ below), combining both
uncertainty types together — including $h_U$'s epistemic standard deviation,
handled via a location-scale reparameterization.

## Variable partition

- **Stochastic (aleatory) parameters**: $\mathbf{X} = [k,\, C_p,\, \rho,\, Z]$, where $k, C_p, \rho$ are random variables with fixed, precisely-known means and standard deviations, and $Z \sim \mathcal{N}(0,1)$ is a *standardized* random variable (see below).
- **Epistemic parameters**: $\mathbf{y} = [T_\infty,\, T_W,\, b]$, known only to lie within intervals, with midpoints $\mathbf{m}_y$. Since $T_0 = T_\infty$ by assumption, $T_0$ is coupled to $T_\infty$ and is not an independent variable.
- **Epistemic hyperparameter**: $\sigma_{h_U}$, the standard deviation of $h_U$, known only to lie in an interval (`STD_hU`). It is treated separately from $\mathbf{y}$ below because, unlike $T_\infty, T_W, b$, it is a *scale* parameter rather than a *location* parameter, and propagates differently.

### Location-scale reparameterization of $h_U$

A plain derivative $\partial T/\partial h_U$ cannot carry $\sigma_{h_U}$'s uncertainty, since $\sigma_{h_U}$ never appears as an argument to the model — only $h_U$ itself does. Writing $h_U$ in location-scale form exposes it:

$$
h_U = \mu_{h_U} + \sigma_{h_U} \cdot Z, \qquad Z \sim \mathcal{N}(0,1)
$$

$Z$ is the aleatory quantity that actually appears in $\mathbf{X}$; $\sigma_{h_U}$ is a coefficient that scales it.

**A subtlety worth flagging.** It's tempting to treat $\sigma_{h_U}$ as "just another epistemic variable," Taylor-expand $T$ in it the same way as $T_\infty, T_W, b$, and read off a gradient $\partial T/\partial \sigma_{h_U}$. Doing so gives, by the chain rule, $\partial T/\partial \sigma_{h_U} = (\partial T/\partial h_U)\cdot Z$, which is exactly $0$ at the nominal point ($Z=0$). That derivative is correct, but it answers a different question than the one that matters for variance propagation: it says perturbing $\sigma_{h_U}$ while $h_U$ sits exactly at its mean does nothing — true, but irrelevant, since $\mathrm{Var}[h_U] = \sigma_{h_U}^2$ is already an **exact**, closed-form function of $\sigma_{h_U}$, not something that benefits from (or should be) linearized. The correct treatment, used below, expands $T$ in $Z$ only and keeps $\sigma_{h_U}$ as an exact parameter throughout.

## Nominal point

$$
(\boldsymbol{\mu}_X, \mathbf{m}_y), \qquad Z = 0, \qquad m_{y_j} = \frac{\underline{y}_j + \overline{y}_j}{2}
$$

so the nominal value of $h_U$ is $\mu_{h_U} + \sigma_{h_U} \cdot 0 = \mu_{h_U}$, for any $\sigma_{h_U}$.

## Transforming $\partial T/\partial h_U$ into $\partial T/\partial Z$

$T$ depends on the model through $h_U$ directly (it's the quantity actually passed to `analytic_solution`); it has no direct dependence on $Z$. The chain rule relates the two gradients through $h_U = \mu_{h_U} + \sigma_{h_U} Z$:

$$
\frac{\partial T}{\partial Z} = \frac{\partial T}{\partial h_U}\cdot\frac{\partial h_U}{\partial Z} = \frac{\partial T}{\partial h_U}\cdot \sigma_{h_U}, \qquad \text{since } \frac{\partial h_U}{\partial Z} = \sigma_{h_U}
$$

Evaluated at the nominal point ($Z=0 \Leftrightarrow h_U = \mu_{h_U}$):

$$
\frac{\partial T}{\partial Z}\bigg|_{\text{nom}} = \frac{\partial T}{\partial h_U}\bigg|_{\text{nom}} \cdot \sigma_{h_U}
$$

So the $h_U$-direction's contribution to the 1st-order expansion can be written **either** in the original variable,

$$
\frac{\partial T}{\partial h_U}\bigg|_{\text{nom}} \big(h_U - \mu_{h_U}\big),
$$

**or**, substituting the exact identity $h_U - \mu_{h_U} = \sigma_{h_U} Z$ and the chain-rule result above, in the standardized variable:

$$
\frac{\partial T}{\partial h_U}\bigg|_{\text{nom}} \sigma_{h_U}\, Z \;=\; \frac{\partial T}{\partial Z}\bigg|_{\text{nom}}\, Z
$$

These are the *same* term — reparameterizing doesn't change the 1st-order contribution itself, only which variable it's expressed in terms of. What it changes is how that term behaves once $\sigma_{h_U}$ is treated as uncertain: differentiating $\partial T/\partial Z\big|_{\text{nom}} = (\partial T/\partial h_U\big|_{\text{nom}})\,\sigma_{h_U}$ once more, this time with respect to $\sigma_{h_U}$, recovers exactly $\partial T/\partial h_U\big|_{\text{nom}}$ — a constant, confirming $\sigma_{h_U}$ enters this coefficient as an exact linear factor rather than something requiring its own linearization (consistent with the subtlety noted above).

## 1st-order expansion

$$
T(\mathbf{X}, \mathbf{y}, \tau) \;\approx\; T(\boldsymbol{\mu}_X, \mathbf{m}_y, \tau) \;+\; \nabla_{\mathbf{X}} T\big|_{\text{nom}} \cdot (\mathbf{X} - \boldsymbol{\mu}_X) \;+\; \nabla_{\mathbf{y}} T\big|_{\text{nom}} \cdot (\mathbf{y} - \mathbf{m}_y)
$$

Written out term by term, using $\partial T/\partial Z\big|_{\text{nom}} = (\partial T/\partial h_U\big|_{\text{nom}})\,\sigma_{h_U}$ from above (so $\sigma_{h_U}$ appears as an exact coefficient on $Z$, not as its own expansion variable):

$$
T \approx \underbrace{T(\boldsymbol{\mu}_X, \mathbf{m}_y, \tau)}_{T_0(\tau)}
+ \frac{\partial T}{\partial k}\bigg|_{\text{nom}}\!(k-\mu_k)
+ \frac{\partial T}{\partial C_p}\bigg|_{\text{nom}}\!(C_p-\mu_{C_p})
+ \frac{\partial T}{\partial \rho}\bigg|_{\text{nom}}\!(\rho-\mu_\rho)
+ \frac{\partial T}{\partial h_U}\bigg|_{\text{nom}}\, \sigma_{h_U}\, Z
$$

$$
+\; \frac{\partial T}{\partial T_\infty}\bigg|_{\text{nom}}\!(T_\infty - m_{T_\infty})
+ \frac{\partial T}{\partial T_W}\bigg|_{\text{nom}}\!(T_W - m_{T_W})
+ \frac{\partial T}{\partial b}\bigg|_{\text{nom}}\!(b - m_b)
$$

Every coefficient — $T_0(\tau)$ and each partial derivative — is a function of $\tau$ (equivalently $t$), since this is a Taylor expansion of the entire time history, done pointwise at each timestep. $\sigma_{h_U}$ does not get its own additive term; it multiplies the $Z$ term exactly, carrying its full (possibly interval-valued) range through unapproximated.

## Expectation and variance of the thermal history

Write $f_i = \partial T/\partial X_i\big|_{\text{nom}}$ for $i \in \{k, C_p, \rho, h_U\}$ and $g_j = \partial T/\partial y_j\big|_{\text{nom}}$ for $j \in \{T_\infty, T_W, b\}$, and $\delta k = k - \mu_k$, etc.:

$$
T \approx T_0(\tau) + f_k\,\delta k + f_{C_p}\,\delta C_p + f_\rho\,\delta\rho + f_{h_U}\,\sigma_{h_U}\, Z + g_{T_\infty}\,\delta T_\infty + g_{T_W}\,\delta T_W + g_b\,\delta b
$$

### Expectation

Taking $\mathbb{E}_{\mathbf{X}}[\cdot]$ over the aleatory variables ($\mathbb{E}[\delta k] = \mathbb{E}[\delta C_p] = \mathbb{E}[\delta \rho] = \mathbb{E}[Z] = 0$ — the last one exactly, for *any* $\sigma_{h_U}$, since $\mathbb{E}[Z]=0$ doesn't depend on the scale multiplying it), and leaving $\mathbf{y}$ symbolic:

$$
\mathbb{E}_{\mathbf{X}}[T](\mathbf{y}) \approx T_0(\tau) + g_{T_\infty}\,\delta T_\infty + g_{T_W}\,\delta T_W + g_b\,\delta b
$$

Propagating $\delta T_\infty, \delta T_W, \delta b$ as centered intervals of half-width $\Delta y_j$ via interval arithmetic (as in `TSE_mixed.py`):

$$
\mathbb{E}[T(\tau)] \in T_0(\tau) \;\pm\; \sum_{j} |g_j|\, \Delta y_j
$$

$\sigma_{h_U}$ contributes nothing here — not because of a vanishing gradient, but simply because scale never affects the mean of a zero-mean perturbation, at any order.

### Variance

$$
\mathrm{Var}_{\mathbf{X}}[T] \approx f_k^2 \sigma_k^2 + f_{C_p}^2 \sigma_{C_p}^2 + f_\rho^2 \sigma_\rho^2 + f_{h_U}^2\, \sigma_{h_U}^2
$$

Because $\sigma_{h_U}$ was kept exact (never linearized), $\sigma_{h_U}^2$ appears here as the genuine epistemic interval $[\underline{\sigma}_{h_U}^2, \overline{\sigma}_{h_U}^2]$ — making $\mathrm{Var}_{\mathbf{X}}[T]$ **interval-valued already at 1st order**:

$$
\mathrm{Var}[T(\tau)] \in \Big[f_k^2 \sigma_k^2 + f_{C_p}^2 \sigma_{C_p}^2 + f_\rho^2 \sigma_\rho^2 + f_{h_U}^2\, \underline{\sigma}_{h_U}^2,\;\; f_k^2 \sigma_k^2 + f_{C_p}^2 \sigma_{C_p}^2 + f_\rho^2 \sigma_\rho^2 + f_{h_U}^2\, \overline{\sigma}_{h_U}^2\Big]
$$

This is a genuinely different situation from $T_\infty, T_W, b$: those are *location* parameters, and a location shift only affects variance through curvature (a 2nd-order/cross-Hessian effect — see `TSE_2nd_order_bounds.py`). $\sigma_{h_U}$ is a *scale* parameter multiplying a random variable directly, so its effect on variance is already exact at 1st order, no curvature needed. So at 1st order:

- $\mathbb{E}[T]$ gets an interval from $T_\infty, T_W, b$, and none from $\sigma_{h_U}$.
- $\mathrm{Var}[T]$ gets an interval from $\sigma_{h_U}$ alone; $T_\infty, T_W, b$ don't widen it until 2nd order.

This confirms (rather than revises) the treatment already used in `TSE_2nd_order_bounds.py`, which multiplies the squared aleatory-gradient bracket for $h_U$ by the exact interval $\sigma_{h_U}^2$ (`sigma_A_sq[3] = sigma_hU_sq`) — no correction to that script is needed.

## Main-effect Sobol indices

$\mathrm{Var}_{\mathbf{X}}[T]$ above is **exactly additive** across the four aleatory directions — a linear (1st-order) expansion has zero interaction terms by construction, so this sum already *is* a complete Sobol variance decomposition, not an approximation of one. Define each variable's partial variance:

$$
V_k = f_k^2\sigma_k^2, \qquad V_{C_p} = f_{C_p}^2\sigma_{C_p}^2, \qquad V_\rho = f_\rho^2\sigma_\rho^2, \qquad V_{h_U}(\sigma_{h_U}^2) = f_{h_U}^2\,\sigma_{h_U}^2
$$

so that $\mathrm{Var}_{\mathbf{X}}[T] = V_k + V_{C_p} + V_\rho + V_{h_U}$ exactly, and the main-effect Sobol index of variable $i$ is $S_i = V_i / \mathrm{Var}_{\mathbf{X}}[T]$. Since $V_{h_U}$ is interval-valued and sits inside the denominator, *every* $S_i$ inherits an interval — even $k, C_p, \rho$, whose own $V_i$ is a fixed constant.

### $S_k, S_{C_p}, S_\rho$: exact via interval arithmetic

$V_k, V_{C_p}, V_\rho$ don't depend on $\sigma_{h_U}^2$ at all, so plain interval division is exact here — the numerator has zero width, so there's no numerator/denominator correlation to lose:

$$
S_i \in \left[\frac{V_i}{\overline{\mathrm{Var}}[T]},\; \frac{V_i}{\underline{\mathrm{Var}}[T]}\right], \qquad i \in \{k, C_p, \rho\}
$$

using the $\underline{\mathrm{Var}}[T], \overline{\mathrm{Var}}[T]$ bounds derived above.

### $S_{h_U}$: naive interval division over-widens — use monotonicity instead

$S_{h_U}$ is different: $\sigma_{h_U}^2$ appears in *both* the numerator and denominator, so they're correlated — the same dependency problem flagged for `TSE_2nd_order_bounds.py`. Writing $V_{\text{fixed}} = V_k + V_{C_p} + V_\rho$ (a constant) and $x = \sigma_{h_U}^2$:

$$
S_{h_U}(x) = \frac{f_{h_U}^2 x}{V_{\text{fixed}} + f_{h_U}^2 x}
$$

Dividing the separately-computed intervals $V_{h_U} \in [f_{h_U}^2\underline{x}, f_{h_U}^2\overline{x}]$ and $\mathrm{Var}[T] \in [\underline{\mathrm{Var}}[T], \overline{\mathrm{Var}}[T]]$ would pair the smallest numerator with the largest denominator (and vice versa) — combinations that can't actually co-occur, since both move together with $x$. Instead, note $S_{h_U}(x)$ is **monotonically increasing** in $x$ (for $V_{\text{fixed}} > 0$):

$$
\frac{dS_{h_U}}{dx} = \frac{f_{h_U}^2\, V_{\text{fixed}}}{\big(V_{\text{fixed}} + f_{h_U}^2 x\big)^2} > 0
$$

so its exact range comes from evaluating **pointwise** at the two interval endpoints, not from interval division:

$$
S_{h_U} \in \left[\frac{f_{h_U}^2\,\underline{\sigma}_{h_U}^2}{V_{\text{fixed}} + f_{h_U}^2\,\underline{\sigma}_{h_U}^2},\;\; \frac{f_{h_U}^2\,\overline{\sigma}_{h_U}^2}{V_{\text{fixed}} + f_{h_U}^2\,\overline{\sigma}_{h_U}^2}\right]
$$

which is strictly narrower than what naive interval division would give.

### Sanity check and caveat

At any *fixed* $\sigma_{h_U}^2$, $\sum_i S_i = 1$ exactly, since $\sum_i V_i = \mathrm{Var}[T]$ exactly with no missing interaction terms — unlike the Monte Carlo estimates in `sobol_crude_vs_double_loop.py`, which summed to less than 1 because the true (nonlinear) model has real interaction effects a 1st-order expansion can't see. But the *bounds* don't need to sum to anything in particular: $S_k$'s lower bound is achieved at $\sigma_{h_U}^2 = \overline{\sigma}_{h_U}^2$, while $S_{h_U}$'s lower bound is achieved at $\sigma_{h_U}^2 = \underline{\sigma}_{h_U}^2$ — different points in the epistemic interval — so $\sum_i \underline{S}_i \neq 1$ in general.

$T_\infty, T_W, b$ don't appear anywhere in this decomposition: since they don't affect $\mathrm{Var}[T]$ at 1st order at all, their implied main-effect variance is exactly zero at this order — any nonzero sensitivity for them only appears once curvature (2nd order) is included.

## Notes

- **$T_\infty$ is a total derivative.** $T_\infty$ appears twice in `analytic_solution` (the explicit `Tinf` argument and `T0 = Tinf`), so $\partial T/\partial T_\infty$ must be computed by perturbing both occurrences together — exactly as `TSE_mixed.py` already does.
- **Relation to the 2nd-order expansion.** This is the linear part of the 2nd-order expansion documented in `TSE_2nd_order_bounds.py` (see `THERMAL-FIN-MODEL-DESCRIPTION.md`), just without the curvature terms $H$ (aleatory-aleatory), $C$ (mixed), $G$ (epistemic-epistemic) that let $T_\infty, T_W, b$ start to affect variance.
