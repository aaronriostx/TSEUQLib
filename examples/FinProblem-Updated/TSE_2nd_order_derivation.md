# 2nd-order Taylor series expansion of the thermal fin problem

This note derives the 2nd-order Taylor series expansion (TSE) of the fin
temperature history $T(0,\tau)$ (written as $T$ below) implemented in
`TSE_2nd_order_bounds.py`, extending `TSE_1st_order_derivation.md` with the
curvature (Hessian) terms that let the epistemic location parameters
$T_\infty, T_W, b$ start to affect $\mathrm{Var}[T]$, and that couple
$\sigma_{h_U}$'s epistemic scale into the expansion's curvature as well as
its linear part.

## Variable partition

Same partition as the 1st-order note:

- **Aleatory** $\mathbf{X} = [k,\, C_p,\, \rho,\, h_U]$: independent Gaussian random variables with fixed, precisely-known means and standard deviations — *except* $h_U$, whose standard deviation $\sigma_{h_U}$ is itself only known to lie in an interval.
- **Epistemic** $\mathbf{y} = [T_\infty,\, T_W,\, b]$: interval-valued location parameters, midpoints $\mathbf{y}_0$.

## Nominal point

$$
(\boldsymbol{\mu}_X, \mathbf{y}_0), \qquad y_{0,j} = \frac{\underline{y}_j + \overline{y}_j}{2}
$$

exactly as in the 1st-order note. Write $\Delta X_i = X_i - \mu_{X_i}$ (aleatory deviations, random) and $\Delta\theta_j = y_j - y_{0,j}$ (epistemic deviations, symbolic/interval-valued — kept as free variables, not integrated over).

## $\sigma_{h_U}$ still needs the location-scale treatment — but it now provably doesn't change how the code computes it

The 1st-order note flagged a subtlety: $\sigma_{h_U}$ is a *scale* parameter, and naively Taylor-expanding $T$ in $\sigma_{h_U}$ directly (rather than through $h_U = \mu_{h_U} + \sigma_{h_U} Z$) gives the wrong object for variance propagation. The same subtlety applies at 2nd order, and it's worth checking explicitly whether `TSE_2nd_order_bounds.py` needs to compute derivatives with respect to $Z$ (via the reparameterization) rather than the $\partial T/\partial h_U$, $\partial^2 T/\partial h_U^2$ it actually finite-differences.

**Claim: it doesn't need to.** Let $\dfrac{\partial T}{\partial h_U}\bigg|_{\text{nom}}, \dfrac{\partial^2 T}{\partial h_U^2}\bigg|_{\text{nom}}, \dfrac{\partial^2 T}{\partial h_U\,\partial X_i}\bigg|_{\text{nom}}\;(i\in\{k,C_p,\rho\}), \dfrac{\partial^2 T}{\partial h_U\,\partial y_j}\bigg|_{\text{nom}}\;(j\in\{T_\infty,T_W,b\})$ denote the gradient/Hessian entries computed by finite-differencing $T$ directly in $h_U$ (exactly what the code does — $h_U$ is just the 4th entry of $z$ in `TSE_2nd_order_bounds.py`). Since $h_U = \mu_{h_U} + \sigma_{h_U} Z$ is an *affine* function of $Z$ with constant slope $\sigma_{h_U}$, the chain rule gives the corresponding $Z$-derivatives exactly:

$$
\dfrac{\partial T}{\partial Z}\bigg|_{\text{nom}} = \sigma_{h_U} \dfrac{\partial T}{\partial h_U}\bigg|_{\text{nom}}, \qquad \dfrac{\partial^2 T}{\partial Z^2}\bigg|_{\text{nom}} = \sigma_{h_U}^2\, \dfrac{\partial^2 T}{\partial h_U^2}\bigg|_{\text{nom}}, \qquad \dfrac{\partial^2 T}{\partial Z\,\partial X_i}\bigg|_{\text{nom}} = \sigma_{h_U}\, \dfrac{\partial^2 T}{\partial h_U\,\partial X_i}\bigg|_{\text{nom}}, \qquad \dfrac{\partial^2 T}{\partial Z\,\partial y_j}\bigg|_{\text{nom}} = \sigma_{h_U}\, \dfrac{\partial^2 T}{\partial h_U\,\partial y_j}\bigg|_{\text{nom}}
$$

(no extra terms, since $\partial^2 h_U/\partial Z^2 = 0$). Substituting these into every place a "$h_U$" gradient/Hessian entry would appear in the $\mathrm{Var}$ formula derived below, together with $\mathrm{Var}[Z] = 1$ exactly (fixed, not epistemic), reproduces **exactly** the same expression as substituting the raw $\partial T/\partial h_U$ and $\partial^2 T/\partial h_U\,\partial(\cdot)$ entries (gradient and every Hessian row/column touching $h_U$), each evaluated at nom, together with $\mathrm{Var}[h_U] = \sigma_{h_U}^2$ (epistemic) — every $\sigma_{h_U}$ power introduced by the chain rule exactly cancels/matches the power that would otherwise come from using $\mathrm{Var}[h_U]=\sigma_{h_U}^2$ directly. (Checked term-by-term in the variance derivation below; e.g. the diagonal curvature term $\tfrac12 \left(\dfrac{\partial^2 T}{\partial Z^2}\bigg|_{\text{nom}}\right)^{\!2}\mathrm{Var}[Z]^2 = \tfrac12\left(\sigma_{h_U}^2 \dfrac{\partial^2 T}{\partial h_U^2}\bigg|_{\text{nom}}\right)^2 = \tfrac12 \left(\dfrac{\partial^2 T}{\partial h_U^2}\bigg|_{\text{nom}}\right)^{\!2}\sigma_{h_U}^4$, identical to treating $h_U$ as an ordinary aleatory variable with $\dfrac{\partial^2 T}{\partial h_U^2}\bigg|_{\text{nom}}$ and $\mathrm{Var}[h_U]^2=\sigma_{h_U}^4$.)

So: **treat $h_U$ as the 4th aleatory variable with $\mathrm{Var}[h_U] := \sigma_{h_U}^2$ (epistemic interval), using raw finite differences in $h_U$** — exactly what the code already does — is not a shortcut that happens to work, it is the reparameterized derivation, algebraically simplified. No code change is implied by this note; it's a correctness check on the existing implementation.

## 2nd-order expansion

$$
T \approx T_0(\tau) + \sum_i \dfrac{\partial T}{\partial X_i}\bigg|_{\text{nom}}\,\Delta X_i + \sum_j \dfrac{\partial T}{\partial y_j}\bigg|_{\text{nom}}\,\Delta\theta_j + \frac12\sum_{i,i'} \dfrac{\partial^2 T}{\partial X_i\,\partial X_{i'}}\bigg|_{\text{nom}}\,\Delta X_i \Delta X_{i'} + \sum_{i,j} \dfrac{\partial^2 T}{\partial X_i\,\partial y_j}\bigg|_{\text{nom}}\,\Delta X_i\,\Delta\theta_j + \frac12\sum_{j,j'} \dfrac{\partial^2 T}{\partial y_j\,\partial y_{j'}}\bigg|_{\text{nom}}\,\Delta\theta_j\Delta\theta_{j'}
$$

with $i,i'$ ranging over the aleatory indices $\{k, C_p, \rho, h_U\}$ and $j,j'$ over the epistemic indices $\{T_\infty, T_W, b\}$ — so the $\Delta X_i\Delta X_{i'}$ term above is the aleatory–aleatory curvature, $\Delta X_i\Delta\theta_j$ the mixed curvature, and $\Delta\theta_j\Delta\theta_{j'}$ the epistemic–epistemic curvature. All coefficients are functions of $\tau$, as in the 1st-order note.

## Expectation

Take $\mathbb{E}_{\mathbf{X}}[\cdot]$, leaving $\Delta\theta$ symbolic. Since $\mathbf{X}$ is independent with $\mathbb{E}[\Delta X_i]=0$: the linear-in-$\Delta X$ term vanishes; the mixed term $\dfrac{\partial^2 T}{\partial X_i\,\partial y_j}\bigg|_{\text{nom}}\Delta X_i\Delta\theta_j$ vanishes ($\mathbb{E}[\Delta X_i]=0$); the aleatory–aleatory quadratic term reduces to its diagonal, since $\mathbb{E}[\Delta X_i\Delta X_{i'}] = 0$ for $i\neq i'$ (independence) and $=\sigma_i^2$ for $i=i'$:

$$
\mathbb{E}_{\mathbf{X}}[T](\Delta\theta) = \underbrace{T_0(\tau) + \frac12\sum_i \dfrac{\partial^2 T}{\partial X_i^2}\bigg|_{\text{nom}}\,\sigma_i^2}_{\text{constant in }\Delta\theta} + \sum_j \dfrac{\partial T}{\partial y_j}\bigg|_{\text{nom}}\,\Delta\theta_j + \frac12\sum_{j,j'} \dfrac{\partial^2 T}{\partial y_j\,\partial y_{j'}}\bigg|_{\text{nom}}\,\Delta\theta_j\Delta\theta_{j'}
$$

Written out term by term (no summation operators), with $\Delta T_\infty = T_\infty - T_{\infty,0}$, etc.:

$$
\mathbb{E}_{\mathbf{X}}[T](\Delta\theta) = T_0(\tau)
+ \frac12\dfrac{\partial^2 T}{\partial k^2}\bigg|_{\text{nom}}\sigma_k^2
+ \frac12\dfrac{\partial^2 T}{\partial C_p^2}\bigg|_{\text{nom}}\sigma_{C_p}^2
+ \frac12\dfrac{\partial^2 T}{\partial \rho^2}\bigg|_{\text{nom}}\sigma_\rho^2
+ \frac12\dfrac{\partial^2 T}{\partial h_U^2}\bigg|_{\text{nom}}\sigma_{h_U}^2
$$

$$
+\; \dfrac{\partial T}{\partial T_\infty}\bigg|_{\text{nom}}\Delta T_\infty
+ \dfrac{\partial T}{\partial T_W}\bigg|_{\text{nom}}\Delta T_W
+ \dfrac{\partial T}{\partial b}\bigg|_{\text{nom}}\Delta b
$$

$$
+\; \frac12\dfrac{\partial^2 T}{\partial T_\infty^2}\bigg|_{\text{nom}}\Delta T_\infty^2
+ \frac12\dfrac{\partial^2 T}{\partial T_W^2}\bigg|_{\text{nom}}\Delta T_W^2
+ \frac12\dfrac{\partial^2 T}{\partial b^2}\bigg|_{\text{nom}}\Delta b^2
$$

$$
+\; \dfrac{\partial^2 T}{\partial T_\infty\,\partial T_W}\bigg|_{\text{nom}}\Delta T_\infty\Delta T_W
+ \dfrac{\partial^2 T}{\partial T_\infty\,\partial b}\bigg|_{\text{nom}}\Delta T_\infty\Delta b
+ \dfrac{\partial^2 T}{\partial T_W\,\partial b}\bigg|_{\text{nom}}\Delta T_W\Delta b
$$

(the last two lines are the epistemic–epistemic curvature: three diagonal terms and three off-diagonal terms, the latter already merged using $\partial^2T/\partial T_\infty\partial T_W = \partial^2T/\partial T_W\partial T_\infty$, etc., so each unordered pair appears once with coefficient $1$ rather than twice with coefficient $\tfrac12$.)

This is the formula in `TSE_2nd_order_bounds.py`. The curvature correction $\tfrac12\sum_i \dfrac{\partial^2 T}{\partial X_i^2}\bigg|_{\text{nom}}\sigma_i^2$ is new relative to 1st order — even the *constant* term now depends on how curved $T$ is along each aleatory direction, including $h_U$ via $\dfrac{\partial^2 T}{\partial h_U^2}\bigg|_{\text{nom}}\sigma_{h_U}^2$ (epistemic-interval-valued, since $\sigma_{h_U}^2$ is).

## Variance

Write $T - \mathbb{E}_{\mathbf{X}}[T] = L + Q$, where

$$
L = \sum_i \left(\dfrac{\partial T}{\partial X_i}\bigg|_{\text{nom}} + \sum_j \dfrac{\partial^2 T}{\partial X_i\,\partial y_j}\bigg|_{\text{nom}}\,\Delta\theta_j\right)\Delta X_i, \qquad Q = \frac12\sum_{i,i'} \dfrac{\partial^2 T}{\partial X_i\,\partial X_{i'}}\bigg|_{\text{nom}}\,\Delta X_i\Delta X_{i'} - \frac12\sum_i \dfrac{\partial^2 T}{\partial X_i^2}\bigg|_{\text{nom}}\sigma_i^2
$$

$L$ collects everything linear in $\Delta X$: each $\Delta X_i$'s coefficient is its plain gradient $\partial T/\partial X_i|_{\text{nom}}$ plus the epistemic correction from the mixed Hessian, $\sum_j \partial^2T/\partial X_i\partial y_j|_{\text{nom}}\Delta\theta_j$. $Q$ is the aleatory quadratic form, re-centered so $\mathbb{E}[Q]=0$. Then $\mathrm{Var}_{\mathbf{X}}[T] = \mathbb{E}[L^2] + 2\,\mathbb{E}[LQ] + \mathbb{E}[Q^2]$.

**$\mathbb{E}[L^2]$.** By independence and zero mean, cross terms $\mathbb{E}[\Delta X_i\Delta X_{i'}]=0$ for $i\neq i'$:

$$
\mathbb{E}[L^2] = \sum_i \left(\dfrac{\partial T}{\partial X_i}\bigg|_{\text{nom}} + \sum_j \dfrac{\partial^2 T}{\partial X_i\,\partial y_j}\bigg|_{\text{nom}}\,\Delta\theta_j\right)^{\!2}\sigma_i^2
$$

**$\mathbb{E}[LQ] = 0$ exactly.** This is exactly the place a 3rd central moment would enter — $LQ$ is cubic in $\Delta X$. But every cubic term is a product of an odd number of independent, zero-mean $\Delta X_i$ factors — either one factor alone ($\mathbb{E}[\Delta X_i]=0$), or one factor times a square of a *different* independent variable ($\mathbb{E}[\Delta X_i]\cdot\mathbb{E}[\Delta X_{i'}^2]=0$), or three distinct independent zero-mean factors ($\mathbb{E}[\Delta X_i]\mathbb{E}[\Delta X_{i'}]\mathbb{E}[\Delta X_{i''}]=0$), or (when all three factors coincide) $\mathbb{E}[\Delta X_i^3]$, the 3rd central moment itself, which is exactly $0$ for any distribution symmetric about its mean. Every case vanishes *exactly*, not approximately — so no 3rd-moment symbol survives into the formula, and this needs only independence and symmetry, not specifically Gaussianity.

**$\mathbb{E}[Q^2]$: this step does need Gaussianity.** $Q$ is the centered version of the quadratic form $q=\sum_{i,i'} a_{ii'}\Delta X_i\Delta X_{i'}$ with $a_{ii'} = \tfrac12 \dfrac{\partial^2 T}{\partial X_i\,\partial X_{i'}}\bigg|_{\text{nom}}$ (symmetric), so $\mathbb{E}[Q^2] = \mathrm{Var}[q]$. Splitting $q$ into diagonal and off-diagonal parts (independent of each other, and off-diagonal pairs mutually uncorrelated, by independence of the $\Delta X_i$):

$$
\mathrm{Var}[q] = \sum_i a_{ii}^2\,\mathrm{Var}[\Delta X_i^2] + 4\sum_{i<i'} a_{ii'}^2\,\sigma_i^2\sigma_{i'}^2
$$

The off-diagonal term needs only independence: $\mathrm{Var}[\Delta X_i\Delta X_{i'}] = \mathbb{E}[\Delta X_i^2]\mathbb{E}[\Delta X_{i'}^2] = \sigma_i^2\sigma_{i'}^2$. The diagonal term needs the 4th central moment: $\mathrm{Var}[\Delta X_i^2] = \mu_{4,i} - \sigma_i^4$. For **Gaussian** $\Delta X_i$ (true here — $k,C_p,\rho$ are modeled Gaussian, and $h_U$'s aleatory part is the Gaussian $Z$), $\mu_{4,i}=3\sigma_i^4$, so $\mathrm{Var}[\Delta X_i^2] = 2\sigma_i^4$. Substituting $a_{ii'}=\tfrac12 \dfrac{\partial^2 T}{\partial X_i\,\partial X_{i'}}\bigg|_{\text{nom}}$:

$$
\mathbb{E}[Q^2] = \frac12\sum_i \left(\dfrac{\partial^2 T}{\partial X_i^2}\bigg|_{\text{nom}}\right)^{\!2}\sigma_i^4 + \sum_{i<i'} \left(\dfrac{\partial^2 T}{\partial X_i\,\partial X_{i'}}\bigg|_{\text{nom}}\right)^{\!2}\sigma_i^2\sigma_{i'}^2 = \frac12\sum_{i,i'} \left(\dfrac{\partial^2 T}{\partial X_i\,\partial X_{i'}}\bigg|_{\text{nom}}\right)^{\!2}\,\sigma_i^2\sigma_{i'}^2
$$

(the last equality just re-merges the diagonal and doubled off-diagonal pairs into one symmetric double sum, matching the code's loop structure).

**Where did the 4th moment go?** It's still there, just absorbed into a coefficient rather than shown as a symbol. The diagonal term's prefactor is $a_{ii}^2 = \tfrac14\big(\partial^2T/\partial X_i^2\big)^2$, and for Gaussian $\Delta X_i$, $\mathrm{Var}[\Delta X_i^2] = 2\sigma_i^4$ (from $\mu_{4,i}=3\sigma_i^4$ above) — so the diagonal term is $\tfrac14\cdot 2\,\big(\partial^2T/\partial X_i^2\big)^2\sigma_i^4 = \tfrac12\big(\partial^2T/\partial X_i^2\big)^2\sigma_i^4$. That leading $\tfrac12$ happens to be *exactly* the coefficient the off-diagonal terms already carry from the (moment-free, independence-only) $4\sum_{i<i'}a_{ii'}^2\sigma_i^2\sigma_{i'}^2 = \sum_{i<i'}\big(\partial^2T/\partial X_i\partial X_{i'}\big)^2\sigma_i^2\sigma_{i'}^2$ derivation. That coincidence is what lets the diagonal and off-diagonal pieces merge into the single uniform double sum below — but only the diagonal piece secretly used Gaussianity to land on that coefficient; the off-diagonal one would have it regardless of distribution. If $\mu_{4,i}$ ever needed to be something other than $3\sigma_i^4$ (a non-Gaussian $\Delta X_i$), that $\tfrac12$ would stop matching and the merge into one double sum would no longer be valid — see the Notes below for the substitution.

Putting it together:

$$
\boxed{\ \mathrm{Var}_{\mathbf{X}}[T](\Delta\theta) = \sum_i \left(\dfrac{\partial T}{\partial X_i}\bigg|_{\text{nom}} + \sum_j \dfrac{\partial^2 T}{\partial X_i\,\partial y_j}\bigg|_{\text{nom}}\,\Delta\theta_j\right)^{\!2}\sigma_i^2 \;+\; \frac12\sum_{i,i'} \left(\dfrac{\partial^2 T}{\partial X_i\,\partial X_{i'}}\bigg|_{\text{nom}}\right)^{\!2}\,\sigma_i^2\sigma_{i'}^2\ }
$$

Written out term by term (no summation operators) — the square in each of the first four terms is left unexpanded, since expanding it would trade the $\sum_j$ nesting for 6 more cross-terms per variable (24 total) rather than removing terms:

$$
\mathrm{Var}_{\mathbf{X}}[T](\Delta\theta) = \left(\dfrac{\partial T}{\partial k}\bigg|_{\text{nom}} + \dfrac{\partial^2 T}{\partial k\,\partial T_\infty}\bigg|_{\text{nom}}\Delta T_\infty + \dfrac{\partial^2 T}{\partial k\,\partial T_W}\bigg|_{\text{nom}}\Delta T_W + \dfrac{\partial^2 T}{\partial k\,\partial b}\bigg|_{\text{nom}}\Delta b\right)^{\!2}\sigma_k^2
$$

$$
+\; \left(\dfrac{\partial T}{\partial C_p}\bigg|_{\text{nom}} + \dfrac{\partial^2 T}{\partial C_p\,\partial T_\infty}\bigg|_{\text{nom}}\Delta T_\infty + \dfrac{\partial^2 T}{\partial C_p\,\partial T_W}\bigg|_{\text{nom}}\Delta T_W + \dfrac{\partial^2 T}{\partial C_p\,\partial b}\bigg|_{\text{nom}}\Delta b\right)^{\!2}\sigma_{C_p}^2
$$

$$
+\; \left(\dfrac{\partial T}{\partial \rho}\bigg|_{\text{nom}} + \dfrac{\partial^2 T}{\partial \rho\,\partial T_\infty}\bigg|_{\text{nom}}\Delta T_\infty + \dfrac{\partial^2 T}{\partial \rho\,\partial T_W}\bigg|_{\text{nom}}\Delta T_W + \dfrac{\partial^2 T}{\partial \rho\,\partial b}\bigg|_{\text{nom}}\Delta b\right)^{\!2}\sigma_\rho^2
$$

$$
+\; \left(\dfrac{\partial T}{\partial h_U}\bigg|_{\text{nom}} + \dfrac{\partial^2 T}{\partial h_U\,\partial T_\infty}\bigg|_{\text{nom}}\Delta T_\infty + \dfrac{\partial^2 T}{\partial h_U\,\partial T_W}\bigg|_{\text{nom}}\Delta T_W + \dfrac{\partial^2 T}{\partial h_U\,\partial b}\bigg|_{\text{nom}}\Delta b\right)^{\!2}\sigma_{h_U}^2
$$

$$
+\; \tfrac12\left(\dfrac{\partial^2 T}{\partial k^2}\bigg|_{\text{nom}}\right)^{\!2}\sigma_k^4 + \tfrac12\left(\dfrac{\partial^2 T}{\partial C_p^2}\bigg|_{\text{nom}}\right)^{\!2}\sigma_{C_p}^4 + \tfrac12\left(\dfrac{\partial^2 T}{\partial \rho^2}\bigg|_{\text{nom}}\right)^{\!2}\sigma_\rho^4 + \tfrac12\left(\dfrac{\partial^2 T}{\partial h_U^2}\bigg|_{\text{nom}}\right)^{\!2}\sigma_{h_U}^4
$$

$$
+\; \left(\dfrac{\partial^2 T}{\partial k\,\partial C_p}\bigg|_{\text{nom}}\right)^{\!2}\sigma_k^2\sigma_{C_p}^2 + \left(\dfrac{\partial^2 T}{\partial k\,\partial \rho}\bigg|_{\text{nom}}\right)^{\!2}\sigma_k^2\sigma_\rho^2 + \left(\dfrac{\partial^2 T}{\partial k\,\partial h_U}\bigg|_{\text{nom}}\right)^{\!2}\sigma_k^2\sigma_{h_U}^2 + \left(\dfrac{\partial^2 T}{\partial C_p\,\partial \rho}\bigg|_{\text{nom}}\right)^{\!2}\sigma_{C_p}^2\sigma_\rho^2 + \left(\dfrac{\partial^2 T}{\partial C_p\,\partial h_U}\bigg|_{\text{nom}}\right)^{\!2}\sigma_{C_p}^2\sigma_{h_U}^2 + \left(\dfrac{\partial^2 T}{\partial \rho\,\partial h_U}\bigg|_{\text{nom}}\right)^{\!2}\sigma_\rho^2\sigma_{h_U}^2
$$

exactly the formula in `TSE_2nd_order_bounds.py`, now derived rather than asserted. It is **exact for the 2nd-order-truncated $T$** (no moment-matching approximation beyond Gaussianity of $\Delta X$) — any remaining error relative to the true $\mathrm{Var}[T]$ comes entirely from truncating $T$'s Taylor series at 2nd order, not from this derivation.

Both sums are functions of $\Delta\theta$ alone (all aleatory randomness has been integrated out), with $\sigma_{h_U}^2$ appearing wherever $\sigma_i^2$ or $\sigma_{i'}^2$ takes $i=h_U$ — an epistemic interval sitting inside an otherwise-closed-form polynomial in $\Delta\theta$.

## Propagating the epistemic quantities

$\mathbb{E}_{\mathbf{X}}[T](\Delta\theta)$ and $\mathrm{Var}_{\mathbf{X}}[T](\Delta\theta)$ above are closed-form polynomials in the four epistemic quantities $(\Delta T_\infty, \Delta T_W, \Delta b, \sigma_{h_U}^2)$ — interval arithmetic is applied to *those*, not to further sweeps of the physics model (`TSE_2nd_order_bounds.py`).

**Caveat (already noted in that script, confirmed here).** Unlike the 1st-order case — where each epistemic quantity appears exactly once, so interval arithmetic is exact — here $\Delta\theta_j$ appears repeatedly (linearly inside each squared $\mathbb{E}[L^2]$ term, and again inside the $\dfrac{\partial^2 T}{\partial y_j\,\partial y_{j'}}\bigg|_{\text{nom}}$ cross term) and $\sigma_{h_U}^2$ appears squared (the $i=i'=h_U$ term in $\mathbb{E}[Q^2]$). Plain interval arithmetic evaluated straightforwardly through repeated occurrences of the same variable is a valid but generally conservative enclosure (the "dependency problem"). This is the case where affine arithmetic or a rigorous Taylor-model remainder bound (the two methods validated as exact-reproductions of IA at 1st order in `TSE_1st_order_bounds.py`) would do real, non-cosmetic tightening — that extension to the 2nd-order script is future work, not yet built.

## Notes

- **Relation to the 1st-order expansion.** Setting $\dfrac{\partial^2 T}{\partial X_i\,\partial X_{i'}}\bigg|_{\text{nom}}=0$, $\dfrac{\partial^2 T}{\partial X_i\,\partial y_j}\bigg|_{\text{nom}}=0$, $\dfrac{\partial^2 T}{\partial y_j\,\partial y_{j'}}\bigg|_{\text{nom}}=0$ collapses both boxed formulas above exactly to the 1st-order note's $\mathbb{E}[T] = T_0 + \sum_j \dfrac{\partial T}{\partial y_j}\bigg|_{\text{nom}}\Delta\theta_j$ and $\mathrm{Var}[T] = \sum_i \left(\dfrac{\partial T}{\partial X_i}\bigg|_{\text{nom}}\right)^{\!2}\sigma_i^2$.
- **Gaussianity is load-bearing exactly once.** Independence alone (no distributional assumption) gives $\mathbb{E}[L^2]$ and $\mathbb{E}[LQ]=0$. Only the *diagonal* piece of $\mathbb{E}[Q^2]$ needs a distributional assumption, and specifically needs $\mu_{4,i}=3\sigma_i^4$ (Gaussian excess kurtosis 0). If $k, C_p, \rho$, or $Z$ were ever changed to a non-Gaussian family, only that one coefficient (the $\tfrac12 \left(\dfrac{\partial^2 T}{\partial X_i^2}\bigg|_{\text{nom}}\right)^{\!2}\sigma_i^4$ terms, not the off-diagonal Hessian terms or the $\mathbb{E}[L^2]$ terms) would need the substitution $2\sigma_i^4 \to (\mu_{4,i}-\sigma_i^4)$.
- **Cost.** The combined Hessian over all 7 variables ($k,C_p,\rho,h_U,T_\infty,T_W,b$) costs $1 + 2n + 4\binom{n}{2} = 99$ model evaluations for $n=7$, done once — no epistemic-grid sweep, as in `TSE_2nd_order_bounds.py`.
