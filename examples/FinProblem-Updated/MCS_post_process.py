import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import fin_problem as fp

# Load data
data = np.load("MCS_data.npz")
t = data['t']
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

def plot_moments(data, t=None, figsize=(12, 8)):
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
    variance = np.var(data, axis=0)
    skewness = stats.skew(data, axis=0, bias=True)
    kurtosis = stats.kurtosis(data, axis=0, fisher=True, bias=True) 

    moments = [
        (mean,     "Mean",             "royalblue"),
        (variance, "Variance",         "tomato"),
        (skewness, "Skewness",         "seagreen"),
        (kurtosis, "Excess Kurtosis",  "darkorchid"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")

    for ax, (values, label, color) in zip(axes.flat, moments):
        ax.plot(t, values, color=color, linewidth=1.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Statistical Moments vs. Time", fontsize=14, fontweight="bold")
    plt.show()

plot_moments(theta, t=t)