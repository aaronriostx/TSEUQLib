import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import gldpy
import fin_problem as fp

# Load data
data = np.load("MCS_data.npz")

# Load model parameters that were randomly selected
k, Cp, rho, hU, Tinf, Tw, b = data['k'], data['Cp'], data['rho'], data['hU'], data['Tinf'], data['Tw'], data['b']

# Load time
t = data['t']

# Load realizations
T, theta, theta_tau, theta_ss = data['T'], data['theta'], data['theta_tau'], data['theta_ss']
num_of_samples = T.shape[1]

# Figure 1: T vs t (all samples)
plt.figure(figsize=(3,3), dpi=300)
for i in range(num_of_samples):
    plt.plot(t, T[i, :], 'r', alpha=0.1)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.tight_layout()
plt.savefig('MCS_case_1_T_vs_t.png')
plt.close()

# Figure 2: Theta vs t (all samples)
plt.figure(figsize=(3,3), dpi=300)
for i in range(num_of_samples):
    plt.plot(t, theta[i, :], 'r', alpha=0.1)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel(r'$\theta$')
plt.tight_layout()
plt.savefig('MCS_case_1_theta_vs_t.png')
plt.close()

# Figure 3: Expectation of T vs t
T_expectation = np.mean(T, axis=0)

plt.figure(figsize=(3,3), dpi=300)
plt.plot(t, T_expectation, 'r')
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel(r'$\mathbb{E}[T]$ [K]')
plt.tight_layout()
plt.savefig('MCS_case_1_T_expectation_vs_t.png')
plt.close()

def plot_moments(data, filename, t=None, figsize=(12, 8)):
    """
    Plot the first 4 statistical moments across time.

    Parameters
    ----------
    data : np.ndarray, shape (n_simulations, n_timepoints)
    t    : np.ndarray, shape (n_timepoints,) — time vector, optional
    """
    if t is None:
        t = np.arange(data.shape[1])

    mean     = np.mean(data, axis=0)
    central = data - mean  
    variance = np.mean(central**2, axis=0) 
    skewness = stats.skew(data, axis=0)
    kurtosis = stats.kurtosis(data, axis=0) 

    mu2  = np.mean(central**2, axis=0)              # variance
    mu3  = np.mean(central**3, axis=0)              # 3rd central moment
    mu4  = np.mean(central**4, axis=0)              # 4th central moment

    skewness        = mu3 / mu2**(3/2)      # = mu3 / sigma^3
    kurtosis_pearson = mu4 / mu2**2         # = mu4 / sigma^4        (normal = 3)

    moments = [
        (mean,     "Mean",             "royalblue"),
        (variance, "Variance",         "tomato"),
        (skewness, "Skewness",         "seagreen"),
        (kurtosis_pearson, "Kurtosis",  "darkorchid"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")

    for ax, (values, label, color) in zip(axes.flat, moments):
        ax.plot(t, values, color=color, linewidth=1.5)
        ax.set_xlabel("Time")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename)

plot_moments(T, 'MCS_case_1_T_moments_vs_t.png', t=t)

def plot_pairplot(X, y, filename, var_names=None, output_name="Y"):
    """
    Pairplot of inputs and output for Monte Carlo data.

    Parameters
    ----------
    X         : np.ndarray, shape (n_samples, 7)  — input random variables
    y         : np.ndarray, shape (n_samples,)     — solver output
    var_names : list of str, optional              — names for the 7 inputs
    output_name : str                              — name for the output variable
    """
    if var_names is None:
        var_names = [f"X{i+1}" for i in range(X.shape[1])]

    # Combine inputs and output into a single DataFrame
    df = pd.DataFrame(X, columns=var_names)
    df[output_name] = y

    g = sns.pairplot(
        df,
        diag_kind="hist",      # Use histogram on the diagonal
        plot_kws={"alpha": 0.3, "s": 5, "rasterized": True},  # rasterize for large N
        diag_kws={"bins": 30, "edgecolor": "black", "linewidth": 0.5},
        corner=True,           # only lower triangle — cleaner for 8 variables
    )

    g.figure.suptitle("Pairplot — Inputs & Output", y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')

X = np.column_stack([k, Cp, rho, hU, Tinf, Tw, b])
print("X shape:", X.shape)
print("T shape:", T.shape)

# Check if we need to transpose T
Y = T[:, -1] if T.shape[0] == X.shape[0] else T[-1, :]
print("Y shape:", Y.shape)
print("Y unique values:", len(np.unique(Y)))
print("Y min/max:", Y.min(), Y.max())

plot_pairplot(X, Y, 'pairplot.png', var_names=["k", "Cp", "rho", "hU", "Tinf", "Tw", "b"], output_name="Temp at 450s")

# TEST: trying out gldpy to get lambda parameters
def compute_moments(data):
    mean     = np.mean(data, axis=0)
    central = data - mean  

    mu2  = np.mean(central**2, axis=0)              # variance
    mu3  = np.mean(central**3, axis=0)              # 3rd central moment
    mu4  = np.mean(central**4, axis=0)              # 4th central moment

    skewness        = mu3 / mu2**(3/2)      # = mu3 / sigma^3
    kurtosis_pearson = mu4 / mu2**2         # = mu4 / sigma^4        (normal = 3)
    return mean, mu2, skewness, kurtosis_pearson

gld = gldpy.GLD('VSL')
data = T[:,-1]
mean, mu2, skewness, kurtosis_pearson = compute_moments(data)
computed_moments = (mean, mu2, skewness, kurtosis_pearson)
param_MM = gld.fit_MM(data, [0.5,1], computed_moments=computed_moments, bins_hist = 20, maxiter=1000, maxfun=1000, disp_fit=True, test_gof=True)

from gldpy import GLD
def plot_gld_diagnostics(
    data        : np.ndarray,
    params      : np.ndarray,
    xlabel      : str,
    param_type  : str   = "VSL",
    n_grid      : int   = 500,
    label       : str   = "GLD fit",
    units       : str   = "",
    figsize     : tuple = (16, 5),
    savepath    : str   = None,
):
    """
    Plot PDF, CDF, and Q-Q diagnostics for a GLD fit against sampled data.
 
    Parameters
    ----------
    data        : np.ndarray, shape (N,)
        Raw Monte Carlo / sampled data.
    params      : array-like, shape (4,)
        GLD parameters [λ1, λ2, λ3, λ4].
    param_type  : str
        GLD parameterisation — "FMKL" (default), "RS", or "VSL".
    n_grid      : int
        Number of points for the PDF/CDF evaluation grid.
    label       : str
        Legend label for the GLD curve.
    units       : str
        Units string appended to x-axis labels (e.g. "m", "Pa").
    figsize     : tuple
        Figure size (width, height) in inches.
    savepath    : str, optional
        If provided, save the figure to this path.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes  — [ax_pdf, ax_cdf, ax_qq]
    """
 
    params = np.asarray(params, dtype=float)
    data   = np.asarray(data,   dtype=float).ravel()
    N      = len(data)
    gld    = GLD(param_type)
 
    # ── Evaluation grid (trim extreme tails to avoid boundary artefacts) ─────
    x_lo   = np.quantile(data, 0.001)
    x_hi   = np.quantile(data, 0.999)
    x_grid = np.linspace(x_lo, x_hi, n_grid)
 
    # ── GLD PDF and CDF on the grid ──────────────────────────────────────────
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf_vals = gld.PDF_num(x_grid, params)
        cdf_vals = gld.CDF_num(x_grid, params)
 
    # ── Empirical CDF ────────────────────────────────────────────────────────
    data_sorted = np.sort(data)
    ecdf_probs  = (np.arange(1, N + 1) - 0.5) / N   # Hazen plotting position
 
    # ── GLD theoretical quantiles for Q-Q plot ───────────────────────────────
    # Use the GLD quantile function Q(p) evaluated at the empirical probabilities
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gld_quantiles = gld.Q(ecdf_probs, params)
 
    # ── KDE for PDF reference ─────────────────────────────────────────────────
    kde = stats.gaussian_kde(data, bw_method="scott")
 
    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)
    ax_pdf = fig.add_subplot(gs[0])
    ax_cdf = fig.add_subplot(gs[1])
    ax_qq  = fig.add_subplot(gs[2])
 
    # ── Colour palette ────────────────────────────────────────────────────────
    C_DATA = "steelblue"
    C_GLD  = "tomato"
    C_REF  = "black"
 
    # ════════════════════════════════════════════════════════════
    # Panel 1 — PDF
    # ════════════════════════════════════════════════════════════
    ax_pdf.hist(data, bins=100, density=True, alpha=0.25,
                color=C_DATA, label="MC histogram")
    ax_pdf.plot(x_grid, kde(x_grid), color=C_DATA, lw=1.5,
                linestyle="--", label="KDE (empirical)")
    ax_pdf.plot(x_grid, pdf_vals, color=C_GLD, lw=2.5, label=label)
 
    ax_pdf.set_xlabel(xlabel)
    ax_pdf.set_ylabel("Density")
    ax_pdf.set_title("PDF", fontweight="bold")
    ax_pdf.legend(fontsize=8)
    ax_pdf.grid(True, linestyle="--", alpha=0.4)
 
    # ════════════════════════════════════════════════════════════
    # Panel 2 — CDF
    # ════════════════════════════════════════════════════════════
    ax_cdf.plot(data_sorted, ecdf_probs, color=C_DATA, lw=1.2,
                alpha=0.6, label="Empirical CDF")
    ax_cdf.plot(x_grid, cdf_vals, color=C_GLD, lw=2.5, label=label)
 
    ax_cdf.set_xlabel(xlabel)
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.set_title("CDF", fontweight="bold")
    ax_cdf.legend(fontsize=8)
    ax_cdf.grid(True, linestyle="--", alpha=0.4)
 
    # ════════════════════════════════════════════════════════════
    # Panel 3 — Q-Q plot
    # ════════════════════════════════════════════════════════════
    # Subsample to keep the scatter plot readable
    n_qq   = min(N, 2000)
    idx    = np.linspace(0, N - 1, n_qq, dtype=int)
 
    ax_qq.scatter(gld_quantiles[idx], data_sorted[idx],
                  s=4, alpha=0.3, color=C_DATA, label="Data quantiles",
                  rasterized=True)
 
    # Perfect-fit reference line through the 10th–90th percentile range
    lo, hi  = np.quantile(data_sorted, [0.05, 0.95])
    ax_qq.plot([lo, hi], [lo, hi], color=C_REF, lw=1.5,
               linestyle="--", label="Perfect fit (y = x)")
 
    ax_qq.set_xlabel(f"GLD theoretical quantiles ({units})" if units
                     else "GLD theoretical quantiles")
    ax_qq.set_ylabel(f"Empirical quantiles ({units})" if units
                     else "Empirical quantiles")
    ax_qq.set_title("Q-Q Plot", fontweight="bold")
    ax_qq.legend(fontsize=8)
    ax_qq.grid(True, linestyle="--", alpha=0.4)
 
    # ── Annotation: fitted parameters + moments ───────────────────────────────
    l1, l2, l3, l4 = params
    fit_mean = gld.mean(params)
    fit_std  = gld.std(params)
    fit_skew = gld.skewness(params)
    fit_kurt = gld.kurtosis(params)  
 
    param_txt = (
        f"λ1 = {l1:.5f}\n"
        f"λ2 = {l2:.4f}\n"
        f"λ3 = {l3:.5f}\n"
        f"λ4 = {l4:.5f}\n"
        f"─────────────\n"
        f"μ  = {fit_mean:.4f}\n"
        f"μ2  = {fit_std**2:.4f}\n"
        f"Skewness = {fit_skew:.4f}\n"
        f"Kurtosis = {fit_kurt:.4f}"
    )
    ax_cdf.text(0.03, 0.97, param_txt,
                transform=ax_cdf.transAxes, fontsize=7.5,
                va="top", family="monospace",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
 
    fig.suptitle(f"GLD Diagnostics  —  {param_type} parameterisation  "
                 f"(N = {N:,})",
                 fontsize=13, fontweight="bold", y=1.01)
 
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
 
    return fig, [ax_pdf, ax_cdf, ax_qq]

plot_gld_diagnostics(data, param_MM, "Fin temp at 450s", savepath='gld_comparison.png')