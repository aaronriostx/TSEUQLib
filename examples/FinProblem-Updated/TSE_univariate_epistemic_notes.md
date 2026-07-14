# Univariate 2nd-order TSE — which terms are interval-valued

Worked toy cases for a single variable, as a warm-up for the multivariate
treatment in `TSE_1st_order_derivation.md` and `TSE_2nd_order_derivation.md`.

## Case 1: $x$ itself is epistemic

$x$ is known only to lie in an interval $x \in [\underline{x}, \overline{x}]$. Expand about the midpoint:

$$
x_0 = \frac{\underline{x}+\overline{x}}{2}, \qquad r = \frac{\overline{x}-\underline{x}}{2}, \qquad \Delta x = x - x_0 \in [-r, r]
$$

$$
f(x) = f(x_0) + \left(\dfrac{\partial f}{\partial x}\right)_{x_0}\Delta x + \tfrac{1}{2}\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{x_0}\Delta x^2
$$

| Term | Value | Interval-valued? |
|---|---|---|
| $f(x_0)$, $\left(\dfrac{\partial f}{\partial x}\right)_{x_0}$, $\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{x_0}$ | fixed real numbers (evaluated at the midpoint) | **No** — crisp |
| $\Delta x$ | $[-r,\ r]$ | **Yes** |
| $\left(\dfrac{\partial f}{\partial x}\right)_{x_0}\Delta x$ | $\left(\dfrac{\partial f}{\partial x}\right)_{x_0}\cdot[-r, r]$ | **Yes** — symmetric about 0 |
| $\Delta x^2$ | $[0,\ r^2]$ | **Yes** — one-sided: since $\Delta x^2\ge 0$ always, not $[-r^2, r^2]$ |
| $\tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{x_0}\Delta x^2$ | $\tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{x_0}\cdot[0, r^2]$ | **Yes** — one-sided (biases the result, doesn't just widen it) |

## Case 2: $x$ is random, its mean is epistemic

$x = \mu_x + \sigma_x Z$, with $Z\sim N(0,1)$ (aleatory) and $\sigma_x$ fixed and known. The mean $\mu_x$ is epistemic:

$$
\mu_x = \mu_{x,0} + \Delta\mu_x, \qquad \mu_{x,0} = \frac{\underline{\mu_x}+\overline{\mu_x}}{2}, \qquad r_\mu = \frac{\overline{\mu_x}-\underline{\mu_x}}{2}, \qquad \Delta\mu_x \in [-r_\mu, r_\mu]
$$

### 2nd-order expansion with precise $\mu_x$ and $\sigma_x$

If $\mu_x$ and $\sigma_x$ were both precisely known (no epistemic uncertainty yet), the standard 2nd-order Taylor expansion of $f(x)$ about the mean $\mu_x$ is

$$
f(x) = f(\mu_x) + \left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\Delta x + \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\Delta x^2, \qquad \Delta x := x - \mu_x = \sigma_x Z
$$

This step is exact — no truncation yet. Unlike Case 1's $\Delta x$ (an epistemic interval), this $\Delta x$ is purely aleatory: a random variable with mean $0$ and variance $\sigma_x^2$.

Now suppose $\mu_x$ is only known to lie in $[\underline{\mu_x},\overline{\mu_x}]$, so it is no longer a single fixed number to evaluate $f$ and its derivatives at. Write $\mu_x = \mu_{x,0}+\Delta\mu_x$ and Taylor-expand each coefficient about the nominal point $\mu_{x,0}$, keeping only the order needed to keep the whole expression 2nd-order overall in the combined perturbation $(\Delta\mu_x, \Delta x)$:

- $f(\mu_x)$ multiplies $\Delta x^0$, so keep the full 2nd order in $\Delta\mu_x$: $f(\mu_x) \approx f(\mu_{x,0}) + \left(\dfrac{\partial f}{\partial x}\right)_{\mu_{x,0}}\Delta\mu_x + \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}\Delta\mu_x^2$
- $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}$ multiplies $\Delta x$ (already $O(\Delta x)$), so only 1st order in $\Delta\mu_x$ survives at combined 2nd order: $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_x} \approx \left(\dfrac{\partial f}{\partial x}\right)_{\mu_{x,0}} + \left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}\Delta\mu_x$
- $\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}$ multiplies $\Delta x^2$ (already $O(\Delta x^2)$), so 0th order suffices: $\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x} \approx \left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}$

Substituting these three approximations back in and collecting terms gives exactly the expression in Option 1 below.

### Option 1: Taylor-expand about the midpoint $\mu_{x,0}$

Expand about the mean $\mu_x$ (standard delta method, using $\Delta x = x - \mu_x = \sigma_x Z$ exactly), then evaluate the coefficients at the nominal point $\mu_x = \mu_{x,0}$, keeping each to the order needed for a combined 2nd-order result. Grouping the two terms sharing $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_{x,0}}$ and the three terms sharing $\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}$:

$$
f(x) = f(\mu_{x,0}) \;+\; \left(\dfrac{\partial f}{\partial x}\right)_{\mu_{x,0}}\big(\Delta\mu_x + \Delta x\big) \;+\; \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}\Big(\Delta\mu_x^2 + 2\,\Delta x\,\Delta\mu_x + \Delta x^2\Big)
$$

Note $\Delta\mu_x + \Delta x = x - \mu_{x,0}$ exactly — so this combined form is identical to Case 1 with $x_0 \to \mu_{x,0}$, just with the deviation split into its epistemic piece ($\Delta\mu_x$) and aleatory piece ($\Delta x$). The reason to keep the six-term expanded form below (rather than stopping here) is that combining $\Delta\mu_x$ and $\Delta x$ back together would hide which pieces are epistemic-only, aleatory-only, or mixed — expanding keeps that visible:

$$
f(x) = f(\mu_{x,0}) \;+\; \left(\dfrac{\partial f}{\partial x}\right)_{\mu_{x,0}}\Delta\mu_x \;+\; \left(\dfrac{\partial f}{\partial x}\right)_{\mu_{x,0}}\Delta x \;+\; \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}\Delta\mu_x^2 \;+\; \left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}\Delta x\,\Delta\mu_x \;+\; \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}\Delta x^2
$$

| Term | Interval? | Random? |
|---|---|---|
| $f(\mu_{x,0})$ | No | No — crisp |
| $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_{x,0}}\Delta\mu_x$ | **Yes** | No — pure epistemic, linear |
| $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_{x,0}}\Delta x$ | No | **Yes** — pure aleatory, linear |
| $\tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}\Delta\mu_x^2$ | **Yes** (one-sided, per Case 1) | No — pure epistemic, quadratic |
| $\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}\Delta x\,\Delta\mu_x$ | **Yes** | **Yes** — mixed: an interval whose realized value also depends on $\Delta x$ (random, $=\sigma_x Z$) |
| $\tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_{x,0}}\Delta x^2$ | No | **Yes** — pure aleatory, quadratic |

### Option 2: evaluate $f$ and its derivatives directly over the true interval of $\mu_x$

Instead of Taylor-expanding in $\Delta\mu_x$, evaluate $f(\mu_x)$, $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}$, $\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}$ each as the exact range of that function over the full interval $\mu_x \in [\underline{\mu_x}, \overline{\mu_x}]$ directly — no linearization, no truncation error in $\mu_x$:

$$
f(x) = f(\mu_x) \;+\; \left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\sigma_x Z \;+\; \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\sigma_x^2 Z^2
$$

| Term | Interval? | Random? |
|---|---|---|
| $f(\mu_x)$ | **Yes** | No — pure epistemic |
| $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\sigma_x Z$ | **Yes** | **Yes** — mixed |
| $\tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\sigma_x^2 Z^2$ | **Yes** | **Yes** — mixed |

Unlike Option 1, every term is interval-valued here, since $f$, $\dfrac{\partial f}{\partial x}$, and $\dfrac{\partial^2 f}{\partial x^2}$ are themselves interval-valued functions of $\mu_x$ rather than numbers evaluated at one nominal point. This is exact in $\mu_x$ — no truncation error from Taylor-expanding in $\Delta\mu_x$ — but it requires the derivatives of $f$ as genuine interval-valued functions, not single evaluations at a nominal point, which is expensive if $f$ is a costly black-box model.

## Case 3: $x$ is random, its standard deviation is epistemic

$x = \mu_x + \sigma_x Z$, with $Z\sim N(0,1)$ (aleatory) and $\mu_x$ fixed and known **exactly**. Now $\sigma_x$ is epistemic:

$$
\sigma_x = \sigma_{x,0} + \Delta\sigma_x, \qquad \sigma_{x,0} = \frac{\underline{\sigma_x}+\overline{\sigma_x}}{2}, \qquad r_\sigma = \frac{\overline{\sigma_x}-\underline{\sigma_x}}{2}, \qquad \Delta\sigma_x \in [-r_\sigma, r_\sigma]
$$

$\sigma_x$ is a **scale** parameter, not a location parameter like $\mu_x$ in Case 2 — see [[scale_vs_location_taylor_propagation]]. Since $\mu_x$ is exact here, the delta-method step needs no midpoint approximation at all: $f(\mu_x)$, $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}$, $\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}$ are already crisp numbers.

$$
f(x) = f(\mu_x) + \left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\Delta x + \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\Delta x^2, \qquad \Delta x = x - \mu_x = \sigma_x Z
$$

Writing $\sigma_x$ directly as its interval, $\sigma_x \in [\underline{\sigma_x}, \overline{\sigma_x}]$, this same exact expansion reads

$$
f(x) = f(\mu_x) + \left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\sigma_x Z + \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\sigma_x^2 Z^2, \qquad \sigma_x \in [\underline{\sigma_x}, \overline{\sigma_x}]
$$

with $\sigma_x$ (and hence $\sigma_x^2$) interval-valued directly — this is exact and needs no midpoint/deviation split at all, mirroring Option 2 in Case 2. Decomposing $\sigma_x = \sigma_{x,0}+\Delta\sigma_x$ below is only so the term table can separate the epistemic, aleatory, and mixed pieces explicitly.

Substituting $\sigma_x = \sigma_{x,0}+\Delta\sigma_x$ into $\Delta x = \sigma_x Z$ and $\Delta x^2 = \sigma_x^2 Z^2$ is **exact, not a truncation** — $\sigma_x$ and $\sigma_x^2$ are already degree-$\le\!2$ polynomials in $\Delta\sigma_x$, so nothing is being approximated the way $f(\mu_x)$ was in Case 2:

$$
\Delta x = \sigma_{x,0} Z + \Delta\sigma_x\, Z, \qquad \Delta x^2 = \sigma_{x,0}^2 Z^2 + 2\sigma_{x,0}\,\Delta\sigma_x\, Z^2 + \Delta\sigma_x^2\, Z^2
$$

Substituting back in and grouping the two terms sharing $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}$ and the three sharing $\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}$:

$$
f(x) = f(\mu_x) \;+\; \left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\big(\sigma_{x,0} + \Delta\sigma_x\big)Z \;+\; \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\Big(\sigma_{x,0}^2 + 2\sigma_{x,0}\Delta\sigma_x + \Delta\sigma_x^2\Big)Z^2
$$

Since $\sigma_{x,0}+\Delta\sigma_x = \sigma_x$ exactly, this combined form is nothing more than the pre-substitution expression above — $f(\mu_x) + \left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\Delta x + \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\Delta x^2$ with $\Delta x = \sigma_x Z$ written back out. As in Case 2, this collapsed form hides which pieces are epistemic vs. aleatory vs. mixed, so expanding it out is what the term table below needs:

$$
f(x) = f(\mu_x) \;+\; \left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\sigma_{x,0} Z \;+\; \left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\Delta\sigma_x\, Z \;+\; \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\sigma_{x,0}^2 Z^2 \;+\; \left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\sigma_{x,0}\,\Delta\sigma_x\, Z^2 \;+\; \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\Delta\sigma_x^2\, Z^2
$$

| Term | Interval? | Random? |
|---|---|---|
| $f(\mu_x)$ | No | No — crisp |
| $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\sigma_{x,0} Z$ | No | **Yes** — pure aleatory, linear |
| $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\Delta\sigma_x\, Z$ | **Yes** | **Yes** — mixed |
| $\tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\sigma_{x,0}^2 Z^2$ | No | **Yes** — pure aleatory, quadratic |
| $\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\sigma_{x,0}\,\Delta\sigma_x\, Z^2$ | **Yes** | **Yes** — mixed |
| $\tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\Delta\sigma_x^2\, Z^2$ | **Yes** (one-sided, per Case 1) | **Yes** — mixed |

## Case 4: $x$ is random with precise $\mu_x, \sigma_x$; the derivatives are computed numerically

$x = \mu_x + \sigma_x Z$, with both $\mu_x$ and $\sigma_x$ exact — the same starting point as the precise expansion above:

$$
f(x) = f(\mu_x) + \left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}\Delta x + \tfrac12\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}\Delta x^2, \qquad \Delta x = \sigma_x Z \text{ (purely aleatory)}
$$

No distribution parameter is epistemic here. Instead, suppose $f$ has no closed form (e.g. it's an expensive black-box model), so $\left(\dfrac{\partial f}{\partial x}\right)_{\mu_x}$ and $\left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x}$ are estimated by finite differences with step size $h$. Truncation error (from the finite-difference formula itself) and roundoff error (from floating-point arithmetic) mean the *true* derivative is only known to lie in an interval around the *computed* value:

$$
\left(\dfrac{\partial f}{\partial x}\right)_{\mu_x} = \widehat{\left(\dfrac{\partial f}{\partial x}\right)}_{\mu_x} + \delta_1, \quad \delta_1\in[-\epsilon_1,\epsilon_1], \qquad \left(\dfrac{\partial^2 f}{\partial x^2}\right)_{\mu_x} = \widehat{\left(\dfrac{\partial^2 f}{\partial x^2}\right)}_{\mu_x} + \delta_2, \quad \delta_2\in[-\epsilon_2,\epsilon_2]
$$

where $\widehat{(\cdot)}$ is the fixed, known finite-difference estimate actually computed, $\epsilon_1,\epsilon_2$ are crisp a priori error bounds from the scheme's truncation+roundoff analysis (e.g. the standard central-difference bound $\epsilon_1 \sim \tfrac{h^2}{6}|f'''| + \tfrac{2\varepsilon_{\text{mach}}|f|}{h}$), and $\delta_1,\delta_2$ are the actual, unknown (epistemic) errors.

Substituting:

$$
f(x) = f(\mu_x) \;+\; \widehat{\left(\dfrac{\partial f}{\partial x}\right)}_{\mu_x}\Delta x \;+\; \delta_1\,\Delta x \;+\; \tfrac12\widehat{\left(\dfrac{\partial^2 f}{\partial x^2}\right)}_{\mu_x}\Delta x^2 \;+\; \tfrac12\delta_2\,\Delta x^2
$$

| Term | Interval? | Random? |
|---|---|---|
| $f(\mu_x)$ | No | No — crisp |
| $\widehat{\left(\dfrac{\partial f}{\partial x}\right)}_{\mu_x}\Delta x$ | No | **Yes** — pure aleatory, linear |
| $\delta_1\,\Delta x$ | **Yes** | **Yes** — mixed |
| $\tfrac12\widehat{\left(\dfrac{\partial^2 f}{\partial x^2}\right)}_{\mu_x}\Delta x^2$ | No | **Yes** — pure aleatory, quadratic |
| $\tfrac12\delta_2\,\Delta x^2$ | **Yes** | **Yes** — mixed |

Same structural signature as Case 3: the epistemic pieces ($\delta_1,\delta_2$) never appear except multiplying the random $\Delta x$ or $\Delta x^2$ — no pure-epistemic, no-randomness term exists, because numerical-differentiation error is a correction to a *coefficient*, not an additive shift to $x$ itself. Unlike Case 3, though, $\delta_1$ and $\delta_2$ are never squared here (each enters the expression only to the first power), so there's no one-sided-interval subtlety to track — this case really is the simplest of the four.

The qualitative signature of a **scale** parameter, contrasted with Case 2's **location** parameter: every term touched by $\Delta\sigma_x$ is also random — there is no "pure epistemic, no randomness" term at all, because $\sigma_x$ never appears except multiplying $Z$ or $Z^2$. In Case 2, $\Delta\mu_x$ could appear by itself (the pure-epistemic linear and quadratic terms) because $\mu_x$ enters $x$ additively. Here $\sigma_x$ enters multiplicatively, so its epistemic uncertainty can only ever show up *combined with* the randomness it scales — it never produces an interval-only term.
