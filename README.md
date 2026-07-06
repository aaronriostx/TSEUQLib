# TSEUQLib: Taylor Series Expansion Uncertainty Quantification Library
Construct Taylor series expansions for surrogate modelling. Compute UQ and sensitivity analysis metrics of the Taylor series expansion.

# Features

| Function              | Description                                                     | Symbol |
|----------------------|-----------------------------------------------------------------|--------|
| `tse`                | *n*-th-order Taylor Series                                      | $T_{n}(\boldsymbol{x})$ |
| `expectation`        | Expected Value                                                  | $E\left[T_{n}(\boldsymbol{x})\right]$ |
| `central_moment`     | Central Moment                                                  | $\mu_{k}\left[T_{n}(\boldsymbol{x})\right]$ |
| `sobol_indices`      | Partial Variances                                               | $V_{ij\ldots}\left[T_{n}(\boldsymbol{x})\right]$ |
| `sobol_indices`      | Sobol’ Indices                                                  | $S_{ij\ldots}\left[T_{n}(\boldsymbol{x})\right]$ |
| `total_sobol_indices`| Total Sobol’ Indices                                            | $S_{i}^{T}\left[T_{n}(\boldsymbol{x})\right]$ |
| `shapley_values`     | Shapley effects                                                 | $\phi_{i}\left[T_{n}(\boldsymbol{x})\right]$ |
| `remainder`          | Error of $\xi\left[T_{n}(\boldsymbol{x})\right]$              | $\varepsilon\left[\xi\left[T_{n}(\boldsymbol{x})\right]\right]$ |
| —                    | Sensitivities, $\frac{\partial \xi[T_n(\boldsymbol{x})]}{\partial \theta_i}$ | $S^{\xi}_{\theta_i}$ |

# Module Overview

Main functions and classes in `tseuqlib/`:

| File | Function / Class | Description |
|---|---|---|
| `oti_moments.py` | `tse_uq` (class) | Core UQ engine that operates on OTI numbers to propagate Taylor-series-expansion (TSE) statistics. |
| | `.expectation(y)` | Computes the expectation of the TSE represented by OTI number `y`. |
| | `.central_moment(y, k, Ey=None)` | Computes the `k`-th central moment of the TSE expansion. |
| | `.conditional_expectation(y, basis)` | Computes the conditional expectation of the TSE w.r.t. a given basis (subset of variables). |
| | `.sobol_indices(y, max_order=None)` | Computes Sobol sensitivity indices up to a given interaction order. |
| | `.build_hdmr(y, max_order=None, Ey=None)` | Builds the High-Dimensional Model Representation (HDMR) decomposition of `y`. |
| | `.shapley_values_v1/v2(y, max_order=None)` | Two implementations for computing Shapley effects from variance decomposition. |
| | `.tse_remainder` / `.expectation_remainder` / `.central_moment_remainder` / `.sobol_indices_remainder` | Estimate truncation-error remainder terms for the TSE and its statistics. |
| `oti_moments_sens.py` | `tse_uq` (class) | Extended/sensitivity-focused version of `tse_uq` (adds HDMR-aware variants of expectation, central moment, conditional expectation). |
| | `get_extra_bases(abases, n_dim)` | Helper to build extra OTI bases used when constructing the class. |
| | `.central_moment_hdmr`, `.conditional_expectation_hdmr` | HDMR-consistent versions of the central moment / conditional expectation calculations. |
| `oti_util.py` | `build_rv_joint_moments(mu_ind)` | Builds the joint-distribution moment structure from independent input random variables. |
| | `gen_OTI_basis_vector(nvars, order)` | Generates the list of OTI bases for a given number of variables/order. |
| | `gen_OTI_indices(nvars, order)` | Generates index lists identifying each basis term in an OTI number. |
| | `convert_index_to_exponent_form(lst)` | Converts basis-index lists into exponent form. |
| | `tse_remainder(y, n, k)` | Estimates the remainder term of an `n`-th order Taylor series expansion. |
| `rv_moments.py` | `rv_central_moments` (class) | Computes central moments of input random variables from their PDFs/parameters. |
| | `get_pdf_params(rv_pdf_name, rv_mean, rv_stdev)` | Derives distribution parameters (e.g., shape/scale) from mean & stdev for a named PDF. |
| | `gamma_oti(z)` / `beta_oti(a, b)` | OTI-compatible (Lanczos-approximation-based) gamma/beta functions. |
| | `convert_raw_moments_to_central_moments(mu_r, mean)` | Converts raw moments to central moments. |
| | `joint_central_moments(rv_mu, rv_moments_order)` | Computes joint central moments across independent random variables. |
| `rv_moments_v2.py` | `get_pdf_params(...)`, `build_joint(mu_ind)`, `generate_moments(...)` | Newer/alternate versions of the PDF-parameter and joint-moment-building utilities above. |
| `tse_moments.py` | `tse_central_moment` (class) | Large (auto-generated) class with explicit closed-form expressions for TSE statistics. |
| | `.tseEvFirst … tseEvFifth` | Expected value of the TSE at 1st–5th order. |
| | `.tseVarFirst … tseVarFourth` | Variance of the TSE at 1st–4th order. |
| | `.tseTcmFirst … tseTcmThird` | Third central moment (skewness component) at 1st–3rd order. |
| | `.tseFcmFirst … tseFcmThird` | Fourth central moment (kurtosis component) at 1st–3rd order. |
| `util.py` | `create_symbolic_variables(nVar)` | Creates a list of symbolic variables (for symbolic derivative work). |
| | `extract_derivatives(oti_number, nVar, tse_order_max)` | Extracts derivatives (up to 5th order) from an OTI number. |

The two `tse_uq` classes (in `oti_moments.py` and `oti_moments_sens.py`) are the primary public API — they take an OTI-represented model output and compute expectation, variance/central moments, Sobol indices, HDMR decompositions, and Shapley values via Taylor-series propagation. The `rv_moments*.py` and `oti_util.py` files supply supporting machinery (input-distribution moments, OTI basis/index generation), and `tse_moments.py` contains large auto-generated closed-form moment expressions used internally.

# Prerequisites
Install OTILib from https://github.com/mauriaristi/otilib. This will create a new conda environment with OTILib installed.

Activate the otilib conda environment, for example:
```bash
conda activate pyoti
```

Add the TSEUQLib dependencies in one of two ways (TESTING IN PROGRESS). 
1) Update the pyoti environment with:
```bash
conda env update -f environment.yml
```
Or, 2) Update the pyoti environment with:
```bash
conda install line_profiler sympy scipy pandas
```

# Usage
Run the examples in examples/ with python. For example, run the ishigami example by:
```bash
cd TSEUQLib/examples/
python ishigami.py
```


# Copyright
LANL **O5031**

© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
