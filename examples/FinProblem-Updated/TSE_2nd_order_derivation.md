# 2nd-order Taylor series expansion of the thermal fin problem

This note rebuilds the 2nd-order Taylor series expansion (TSE) of the fin
temperature history $T(0,\tau)$ (written $T$ below) from the univariate cases
worked out in `TSE_univariate_epistemic_notes.md`, rather than asserting the
multivariate formula and checking it after the fact (as the previous version
of this note did).

## Variable partition, mapped to the univariate cases

| Variable | Role | Case (`TSE_univariate_epistemic_notes.md`) |
|---|---|---|
| $k, C_p, \rho$ | aleatory Gaussian, mean **and** std precisely known | none — the plain precise-$\mu_x,\sigma_x$ baseline at the top of Case 2 |
| $h_U$ | aleatory Gaussian, mean precise, **std $\sigma_{h_U}$ epistemic** | **Case 3** |
| $T_\infty, T_W, b$ | not random — deterministic, known only within an interval | **Case 1** |

Case 2 (epistemic mean) and Case 4 (numerical-differentiation error) don't
apply to anything here: nothing has an epistemic mean, and the
finite-difference gradient/Hessian below are used as exact, not carrying an
FD-error interval of their own.

## Nominal point and deviations

$$
\boldsymbol{\mu}_X = (\mu_k, \mu_{C_p}, \mu_\rho, \mu_{h_U}), \qquad
\mathbf{y}_0 = (T_{\infty,0}, T_{W,0}, b_0), \qquad y_{0,j} = \frac{\underline{y}_j+\overline{y}_j}{2}
$$

- $k, C_p, \rho$: $\Delta X_i = X_i - \mu_i = \sigma_i Z_i$, $Z_i\sim N(0,1)$ iid, $\sigma_i$ fixed and precise ($i \in \{k, C_p, \rho\}$).
- $h_U$ (**Case 3**): $\Delta X_{h_U} = h_U - \mu_{h_U} = \sigma_{h_U} Z_{h_U}$, $Z_{h_U}\sim N(0,1)$, independent of $Z_k, Z_{C_p}, Z_\rho$; $\sigma_{h_U}\in[\underline{\sigma_{h_U}},\overline{\sigma_{h_U}}]$ is epistemic. As in Case 3, $\sigma_{h_U}$ is substituted **directly as its interval** — no midpoint/deviation split ($\sigma_{h_U} = \sigma_{h_U,0}+\Delta\sigma_{h_U}$) is needed, since that split was only ever for separating epistemic/aleatory/mixed pieces in a term table, not a computational requirement (see Case 3's discussion of Option "exact interval substitution" vs. the decomposed form).
- $T_\infty, T_W, b$ (**Case 1**): $\Delta\theta_j = y_j - y_{0,j} \in [-r_j, r_j]$, $j\in\{T_\infty,T_W,b\}$ — deterministic intervals, no randomness.

## General 2nd-order multivariate TSE (before substituting case-specific forms)

$$
T \approx T_0(\tau) + \sum_i \dfrac{\partial T}{\partial X_i}\bigg|_{\text{nom}}\Delta X_i + \sum_j \dfrac{\partial T}{\partial y_j}\bigg|_{\text{nom}}\Delta\theta_j + \frac12\sum_{i,i'} \dfrac{\partial^2 T}{\partial X_i\partial X_{i'}}\bigg|_{\text{nom}}\Delta X_i\Delta X_{i'} + \sum_{i,j} \dfrac{\partial^2 T}{\partial X_i\partial y_j}\bigg|_{\text{nom}}\Delta X_i\Delta\theta_j + \frac12\sum_{j,j'} \dfrac{\partial^2 T}{\partial y_j\partial y_{j'}}\bigg|_{\text{nom}}\Delta\theta_j\Delta\theta_{j'}
$$

with $i,i'$ over $\{k,C_p,\rho,h_U\}$ and $j,j'$ over $\{T_\infty,T_W,b\}$. This
polynomial's coefficients (the partials, evaluated at the nominal point) don't
care yet how $\Delta X_i$ is generated — that's where the case substitutions
come in next.

## Substituting the case-specific forms

Substitute $\Delta X_i = \sigma_i Z_i$ ($i\in\{k,C_p,\rho\}$, $\sigma_i$ crisp)
and $\Delta X_{h_U} = \sigma_{h_U} Z_{h_U}$ ($\sigma_{h_U}$ epistemic) into
**every** place $\Delta X_i$ appears above — this is an exact substitution,
not a further truncation, since $\Delta X_{h_U}$ literally equals
$\sigma_{h_U}Z_{h_U}$ (Case 3). Tracking only where $h_U$ appears:

- **Linear term**: $\dfrac{\partial T}{\partial h_U}\Big|_{\text{nom}}\,\sigma_{h_U} Z_{h_U}$ — Case 3's "mixed, linear" signature ($\partial f/\partial x \cdot \sigma_x Z$).
- **Pure quadratic** ($i=i'=h_U$): $\tfrac12\dfrac{\partial^2 T}{\partial h_U^2}\Big|_{\text{nom}}\,\sigma_{h_U}^2 Z_{h_U}^2$ — Case 3's "mixed, quadratic" signature.
- **Aleatory–aleatory cross** ($i=h_U$, $i'\in\{k,C_p,\rho\}$): $\dfrac{\partial^2 T}{\partial h_U\,\partial X_{i'}}\Big|_{\text{nom}}\,\sigma_{h_U}Z_{h_U}\,\sigma_{i'}Z_{i'}$ — mixed in $\sigma_{h_U}$, but both factors are random.
- **Mixed aleatory–epistemic cross** ($i=h_U$, epistemic $y_j$): $\dfrac{\partial^2 T}{\partial h_U\,\partial y_j}\Big|_{\text{nom}}\,\sigma_{h_U}Z_{h_U}\,\Delta\theta_j$ — a **doubly mixed** term: epistemic-scale ($\sigma_{h_U}$) $\times$ epistemic-location ($\Delta\theta_j$) $\times$ aleatory ($Z_{h_U}$).

This is the same substitution $\Delta X_{h_U}\to\sigma_{h_U}Z_{h_U}$ applied
uniformly, term by term, into a polynomial whose coefficients (the partials)
don't change under it. That confirms — rather than merely asserting via a
separate chain-rule argument, as the previous version of this note did — that
finite-differencing directly in $h_U$ and later substituting
$\mathrm{Var}[h_U] := \sigma_{h_U}^2$ wherever a second moment of $h_U$ is
needed reproduces exactly the Case-3 substitution, *including* the mixed
$h_U\times y_j$ Hessian term the earlier equivalence check didn't explicitly
re-verify.

## Expectation

Take $\mathbb{E}_Z[\cdot]$ over $Z_k, Z_{C_p}, Z_\rho, Z_{h_U}$ (independent
standard normal), leaving $\Delta\theta$ (Case 1, deterministic) symbolic.
Every term linear in a single $Z_i$ vanishes ($\mathbb{E}[Z_i]=0$), including
the doubly-mixed $h_U\times y_j$ term above (it's linear in $Z_{h_U}$ alone).
Aleatory–aleatory cross terms vanish for $i\neq i'$ (independence,
$\mathbb{E}[Z_i]=0$); the diagonal survives with $\mathbb{E}[Z_i^2]=1$:

$$
\mathbb{E}_{\mathbf{X}}[T](\Delta\theta) = \underbrace{T_0(\tau) + \frac12\sum_i \dfrac{\partial^2 T}{\partial X_i^2}\bigg|_{\text{nom}}\sigma_i^2}_{\text{constant in }\Delta\theta,\ \sigma_{h_U}^2\text{ epistemic}} + \sum_j \dfrac{\partial T}{\partial y_j}\bigg|_{\text{nom}}\Delta\theta_j + \frac12\sum_{j,j'} \dfrac{\partial^2 T}{\partial y_j\,\partial y_{j'}}\bigg|_{\text{nom}}\Delta\theta_j\Delta\theta_{j'}
$$

where $\sigma_i^2$ at $i=h_U$ is the epistemic interval
$[\underline{\sigma_{h_U}}^2,\overline{\sigma_{h_U}}^2]$ (Case 3, substituted
directly, exact), and $\Delta\theta_j,\ \Delta\theta_j\Delta\theta_{j'}$ are
Case 1 objects. **Case 1's one-sided-square caveat applies to the diagonal
$j=j'$ terms**: $\Delta\theta_j^2 \in [0, r_j^2]$, not $[-r_j^2,r_j^2]$, since
$\Delta\theta_j^2\ge 0$ always — this biases $\mathbb{E}[T]$ rather than
merely widening it. Off-diagonal $j\neq j'$ products are ordinary interval
products (can be negative), not subject to that one-sided restriction, since
$\Delta\theta_j$ and $\Delta\theta_{j'}$ are different independent epistemic
quantities.

## Variance

Same $L+Q$ split as before, $T-\mathbb{E}_{\mathbf X}[T] = L+Q$:

$$
L = \sum_i \left(\dfrac{\partial T}{\partial X_i}\bigg|_{\text{nom}} + \sum_j \dfrac{\partial^2 T}{\partial X_i\,\partial y_j}\bigg|_{\text{nom}}\Delta\theta_j\right)\Delta X_i, \qquad Q = \frac12\sum_{i,i'} \dfrac{\partial^2 T}{\partial X_i\partial X_{i'}}\bigg|_{\text{nom}}\Delta X_i\Delta X_{i'} - \frac12\sum_i \dfrac{\partial^2 T}{\partial X_i^2}\bigg|_{\text{nom}}\sigma_i^2
$$

with $\Delta X_i = \sigma_i Z_i$ substituted throughout (case-specific: crisp
$\sigma_i$ for $k,C_p,\rho$; epistemic-interval $\sigma_{h_U}$ for $h_U$, Case
3). $\mathrm{Var}_{\mathbf X}[T] = \mathbb{E}[L^2] + 2\mathbb{E}[LQ] + \mathbb{E}[Q^2]$.

**$\mathbb{E}[L^2]$** (independence, $\mathbb{E}[Z_i]=0$, cross terms vanish):

$$
\mathbb{E}[L^2] = \sum_i \left(\dfrac{\partial T}{\partial X_i}\bigg|_{\text{nom}} + \sum_j \dfrac{\partial^2 T}{\partial X_i\,\partial y_j}\bigg|_{\text{nom}}\Delta\theta_j\right)^{\!2}\sigma_i^2
$$

The $i=h_U$ term of this sum is exactly the doubly-mixed term's contribution:
writing $B_{h_U} := \partial T/\partial h_U|_{\text{nom}} + \sum_j \partial^2T/\partial h_U\partial y_j|_{\text{nom}}\Delta\theta_j$,
the linear-in-$Z_{h_U}$ piece of $L$ is $B_{h_U}\sigma_{h_U}Z_{h_U}$, so
$\mathbb{E}[(B_{h_U}\sigma_{h_U}Z_{h_U})^2] = B_{h_U}^2\sigma_{h_U}^2\,\mathbb{E}[Z_{h_U}^2] = B_{h_U}^2\sigma_{h_U}^2$
— Case 3's delta-method variance contribution $(\partial f/\partial x)^2\sigma_x^2$,
generalized with the Case-1 correction $B_{h_U}$ absorbing the epistemic
$T_\infty,T_W,b$ dependence.

**$\mathbb{E}[LQ]=0$ exactly**, by the same odd-moment/symmetry argument as
before (independence and zero-mean symmetric $Z_i$; no Gaussianity needed).

**$\mathbb{E}[Q^2]$** needs Gaussianity only in its diagonal piece, exactly as
before ($\mu_{4,i}=3\sigma_i^4$ for $i\in\{k,C_p,\rho,h_U\}$ — $Z_{h_U}$ is
still standard normal, Case 3 only makes its *scale* epistemic, not its
shape):

$$
\mathbb{E}[Q^2] = \frac12\sum_{i,i'} \left(\dfrac{\partial^2 T}{\partial X_i\,\partial X_{i'}}\bigg|_{\text{nom}}\right)^{\!2}\sigma_i^2\sigma_{i'}^2
$$

Putting it together:

$$
\boxed{\ \mathrm{Var}_{\mathbf{X}}[T](\Delta\theta) = \sum_i \left(\dfrac{\partial T}{\partial X_i}\bigg|_{\text{nom}} + \sum_j \dfrac{\partial^2 T}{\partial X_i\,\partial y_j}\bigg|_{\text{nom}}\Delta\theta_j\right)^{\!2}\sigma_i^2 \;+\; \frac12\sum_{i,i'} \left(\dfrac{\partial^2 T}{\partial X_i\,\partial X_{i'}}\bigg|_{\text{nom}}\right)^{\!2}\sigma_i^2\sigma_{i'}^2\ }
$$

identical in form to the previous derivation's boxed result — but every
$\sigma_{h_U}$ occurrence here is traceable to an explicit Case-3 substitution
($\Delta X_{h_U}=\sigma_{h_U}Z_{h_U}$) carried through term by term, rather
than asserted equivalent after the fact. $i=i'=h_U$ in the second sum gives
$\tfrac12(\partial^2T/\partial h_U^2)^2\sigma_{h_U}^4$: since
$\sigma_{h_U}\ge 0$ always, $\sigma_{h_U}^4=[\underline{\sigma_{h_U}}^4,\overline{\sigma_{h_U}}^4]$
directly (monotonic on $\sigma_{h_U}\ge0$, no one-sided subtlety *at this
term alone* — but seen together with every other term $\sigma_{h_U}$ appears
in, the repeated-variable dependency-problem caveat below still applies).

## Propagating the epistemic quantities: one-sided squares and the dependency problem

$\mathbb{E}_{\mathbf X}[T](\Delta\theta)$ and $\mathrm{Var}_{\mathbf X}[T](\Delta\theta)$
are closed-form polynomials in the four epistemic quantities
$(\Delta T_\infty,\Delta T_W,\Delta b,\sigma_{h_U}^2)$ — interval arithmetic
is applied to *those*, not to further sweeps of the physics model.

- **Case 1's one-sided square** applies wherever a single $\Delta\theta_j$ (or
  $\sigma_{h_U}$) is squared against itself: $\Delta\theta_j^2\in[0,r_j^2]$
  and $\sigma_{h_U}^4\in[\underline{\sigma_{h_U}}^4,\overline{\sigma_{h_U}}^4]$
  — an interval implementation must use a dedicated tight-square operation
  (not generic interval multiplication, which would wrongly allow negative
  values from $[-r_j,r_j]\times[-r_j,r_j]$'s lower-left/upper-right products).
- **Dependency problem**: $\Delta\theta_j$ appears repeatedly (inside each
  squared $\mathbb{E}[L^2]$ bracket, and again in the $y_j$–$y_{j'}$ Hessian
  term), and $\sigma_{h_U}^2$ appears squared. Plain interval arithmetic
  through repeated occurrences of the same variable is a valid but generally
  conservative enclosure. Affine arithmetic or a Taylor-model remainder bound
  would tighten this (as validated exact at 1st order in
  `TSE_1st_order_bounds.py`); extending that to 2nd order remains future work.

## Term count: how many terms, as a function of $d$, $p$, $n$

Let $d = d_A + d_B$ be the total number of Taylor-expansion variables
($d_A$ aleatory, $d_B$ deterministic-epistemic/Case 1 — here $d=7$,
$d_A=4$, $d_B=3$), $n$ the TSE order ($n=2$ here), and $p\le d_A$ the number
of aleatory variables whose distribution carries an epistemic-interval
hyperparameter (Case 2, epistemic mean, or Case 3, epistemic scale — here
$p=1$: only $\sigma_{h_U}$). A hyperparameter-flagged variable is still just
one of the $d$ dimensions — $p$ does **not** add new Taylor-expansion
dimensions of its own; it only labels which of the existing $d_A$ aleatory
dimensions has an interval-valued (rather than crisp) second moment.

### Raw Taylor polynomial (before taking $\mathbb{E}_Z[\cdot]$)

The number of monomials of total degree $\le n$ in $d$ variables is the
standard stars-and-bars count

$$
N_{\text{raw}}(d,n) = \binom{d+n}{n}
$$

independent of $p$: the "General 2nd-order multivariate TSE" section above is
built purely from $\Delta X_i,\Delta\theta_j$, and which variable's $\sigma$
later turns out to be epistemic doesn't change how many monomials exist —
only what gets substituted into $\Delta X_i$ afterward (Case 3:
$\Delta X_i=\sigma_iZ_i$; precise baseline: same substitution, $\sigma_i$
crisp). For $n=2$:

$$
N_{\text{raw}}(d,2) = \binom{d+2}{2} = \underbrace{1}_{\text{constant}} + \underbrace{d}_{\text{linear}} + \underbrace{d}_{\text{diagonal quadratic}} + \underbrace{\binom{d}{2}}_{\text{cross quadratic}}
$$

For $d=7$: $\binom{9}{2}=36 = 1+7+7+21$.

### After taking $\mathbb{E}_Z[\cdot]$: term counts for $\mathbb{E}[T]$ and $\mathrm{Var}[T]$

Integrating the aleatory randomness out (independence + zero third moment
kills every term linear or cubic in a lone $\Delta X_i$; Gaussianity fixes
the 4th moment) collapses the $N_{\text{raw}}(d,2)=36$ monomials down to the
much smaller set that actually survives in the boxed formulas — counting the
"written out term by term" blocks above directly:

$$
N_{\mathbb{E}[T]}(d_A,d_B) = \underbrace{1}_{T_0} + \underbrace{d_A}_{\frac12 H_{ii}\sigma_i^2} + \underbrace{d_B}_{\text{linear }\Delta\theta_j} + \underbrace{d_B}_{\text{diagonal }\Delta\theta_j^2} + \underbrace{\binom{d_B}{2}}_{\text{cross }\Delta\theta_j\Delta\theta_{j'}} = 1 + d_A + 2d_B + \binom{d_B}{2}
$$

$$
N_{\mathrm{Var}[T]}(d_A) = \underbrace{d_A}_{\text{bracket}^2\text{ terms}} + \underbrace{d_A}_{\text{diagonal }H_{ii}^2\sigma_i^4} + \underbrace{\binom{d_A}{2}}_{\text{cross }H_{ii'}^2\sigma_i^2\sigma_{i'}^2} = 2d_A + \binom{d_A}{2}
$$

For $d_A=4,\ d_B=3$: $N_{\mathbb{E}[T]} = 1+4+6+3=14$ and
$N_{\mathrm{Var}[T]} = 8+6=14$ — matching the two "written out" formulas
above exactly (each has 14 terms).

### Where $p$ actually enters: not the count, but which terms are interval-valued

$p$ leaves $N_{\mathbb{E}[T]}$ and $N_{\mathrm{Var}[T]}$ unchanged — the same
terms exist whether or not any $\sigma_i$ happens to be epistemic. What $p$
determines is how many of those terms carry an epistemic-interval
coefficient rather than a crisp scalar one:

- In $\mathbb{E}[T]$: exactly $p$ of the $d_A$ variance-correction terms
  $\tfrac12 H_{ii}\sigma_i^2$ are interval-valued (here: 1 of 4, $\sigma_{h_U}^2$).
- In $\mathrm{Var}[T]$: $p$ of the $d_A$ bracket$^2$ terms, $p$ of the $d_A$
  diagonal terms $H_{ii}^2\sigma_i^4$, and $\binom{d_A}{2}-\binom{d_A-p}{2}$
  of the $\binom{d_A}{2}$ cross terms $H_{ii'}^2\sigma_i^2\sigma_{i'}^2$
  (those touching at least one flagged variable) are interval-valued. For
  $d_A=4,p=1$: $1$ bracket term, $1$ diagonal term, and
  $\binom42-\binom32=6-3=3$ cross terms (matching $h_U$'s 3 cross-Hessian
  terms with $k,C_p,\rho$ in the written-out formula) — 5 of the 14
  $\mathrm{Var}[T]$ terms are interval-valued, the other 9 crisp.

Extending $N_{\mathbb{E}[T]}$/$N_{\mathrm{Var}[T]}$ to general order $n>2$
would require the corresponding higher-order central moments (skewness,
6th moment, ...) the way $\mathrm{Var}[Q^2]$ needed the 4th here — not
derived in this note; see the Gaussianity note below.

## Notes

- **Relation to the 1st-order expansion.** Zeroing every 2nd derivative
  collapses both boxed formulas to the 1st-order note's
  $\mathbb{E}[T]=T_0+\sum_j (\partial T/\partial y_j)\Delta\theta_j$ and
  $\mathrm{Var}[T]=\sum_i(\partial T/\partial X_i)^2\sigma_i^2$ — consistent
  with Case 1 alone (no quadratic epistemic term) and Case 3 alone (no mixed
  correction $B_i$) both degenerating to their linear pieces.
- **Gaussianity is load-bearing exactly once**: only the diagonal piece of
  $\mathbb{E}[Q^2]$, same as before. This is unaffected by which variable's
  $\sigma$ happens to be epistemic (Case 3) vs. crisp — Case 3 only makes the
  *scale* an interval, not the *shape* of $Z_{h_U}$'s distribution.
- **Cost.** Unchanged: the combined Hessian over all 7 variables
  ($k,C_p,\rho,h_U,T_\infty,T_W,b$) costs $1+2n+4\binom n2=99$ model
  evaluations for $n=7$, done once, no epistemic-grid sweep.
