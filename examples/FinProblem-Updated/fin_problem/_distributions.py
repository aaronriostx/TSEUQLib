import numpy as np
from scipy import stats
from typing import Union


SUPPORTED_DISTRIBUTIONS = [
    "normal", "lognormal", "uniform", "beta", "gamma",
    "exponential", "weibull", "gumbel", "triangular", "student_t"
]


def sample(
    dist_type: str,
    n_samples: int,
    params: dict,
    seed: int = None
) -> np.ndarray:
    """
    Generate random samples from a specified probability distribution.

    Parameters
    ----------
    dist_type : str
        Name of the distribution. Supported values:
            - "normal"      : params = {"mean": float, "std": float}
            - "lognormal"   : params = {"mean": float, "std": float}  <- moments of the underlying normal
            - "uniform"     : params = {"low": float, "high": float}
            - "beta"        : params = {"alpha": float, "beta": float}
            - "gamma"       : params = {"shape": float, "scale": float}
            - "exponential" : params = {"scale": float}               <- scale = 1/lambda
            - "weibull"     : params = {"shape": float, "scale": float}
            - "gumbel"      : params = {"loc": float, "scale": float}
            - "triangular"  : params = {"low": float, "mode": float, "high": float}
            - "student_t"   : params = {"df": float, "loc": float, "scale": float}

    n_samples : int
        Number of samples to generate.

    params : dict
        Dictionary of distribution hyperparameters. See dist_type for required keys.

    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    np.ndarray
        1D array of shape (n_samples,) containing the random samples.

    Raises
    ------
    ValueError
        If an unsupported distribution is requested or required params are missing.

    Examples
    --------
    >>> x = sample("normal", 1000, {"mean": 5.0, "std": 2.0}, seed=42)
    >>> x = sample("lognormal", 500, {"mean": 0.0, "std": 1.0}, seed=0)
    >>> x = sample("uniform", 200, {"low": -1.0, "high": 1.0})
    >>> x = sample("beta", 300, {"alpha": 2.0, "beta": 5.0})
    """
    rng = np.random.default_rng(seed)
    dist_type = dist_type.strip().lower()
    _check_params(dist_type, params)

    loc   = params.get("loc", 0.0)
    scale = params.get("scale", 1.0)

    dispatch = {
        "normal":      lambda: rng.normal(loc=params["mean"], scale=params["std"], size=n_samples),
        # params are the mean and std of the underlying normal (log-space)
        "lognormal":   lambda: rng.lognormal(mean=params["mean"], sigma=params["std"], size=n_samples),
        "uniform":     lambda: rng.uniform(low=params["low"], high=params["high"], size=n_samples),
        "beta":        lambda: rng.beta(a=params["alpha"], b=params["beta"], size=n_samples),
        "gamma":       lambda: rng.gamma(shape=params["shape"], scale=params["scale"], size=n_samples),
        "exponential": lambda: rng.exponential(scale=params["scale"], size=n_samples),
        # numpy's weibull only takes shape; multiply by scale manually
        "weibull":     lambda: params["scale"] * rng.weibull(a=params["shape"], size=n_samples),
        "gumbel":      lambda: rng.gumbel(loc=params["loc"], scale=params["scale"], size=n_samples),
        "triangular":  lambda: rng.triangular(left=params["low"], mode=params["mode"], right=params["high"], size=n_samples),
        "student_t":   lambda: stats.t.rvs(df=params["df"], loc=loc, scale=scale, size=n_samples, random_state=seed),
    }

    if dist_type not in dispatch:
        raise ValueError(
            f"Unsupported distribution: '{dist_type}'. "
            f"Choose from: {SUPPORTED_DISTRIBUTIONS}"
        )

    return dispatch[dist_type]()


# ---------------------------------------------------------------------------
# Required parameter keys for each distribution
# ---------------------------------------------------------------------------
_REQUIRED_PARAMS = {
    "normal":      ["mean", "std"],
    "lognormal":   ["mean", "std"],
    "uniform":     ["low", "high"],
    "beta":        ["alpha", "beta"],
    "gamma":       ["shape", "scale"],
    "exponential": ["scale"],
    "weibull":     ["shape", "scale"],
    "gumbel":      ["loc", "scale"],
    "triangular":  ["low", "mode", "high"],
    "student_t":   ["df"],
}


def _check_params(dist_type: str, params: dict) -> None:
    """Validate that all required parameters are present."""
    if dist_type not in _REQUIRED_PARAMS:
        raise ValueError(
            f"Unsupported distribution: '{dist_type}'. "
            f"Choose from: {SUPPORTED_DISTRIBUTIONS}"
        )
    required = _REQUIRED_PARAMS[dist_type]
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(
            f"Missing required parameter(s) for '{dist_type}': {missing}"
        )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configs = [
        ("normal",      1000, {"mean": 5.0,  "std": 2.0}),
        ("lognormal",   1000, {"mean": 0.0,  "std": 0.5}),
        ("uniform",     1000, {"low": -1.0,  "high": 1.0}),
        ("beta",        1000, {"alpha": 2.0, "beta": 5.0}),
        ("gamma",       1000, {"shape": 2.0, "scale": 1.0}),
        ("exponential", 1000, {"scale": 2.0}),
        ("weibull",     1000, {"shape": 1.5, "scale": 2.0}),
        ("gumbel",      1000, {"loc": 0.0,   "scale": 1.0}),
        ("triangular",  1000, {"low": 0.0,   "mode": 1.0, "high": 3.0}),
        ("student_t",   1000, {"df": 5.0,    "loc": 0.0,  "scale": 1.0}),
    ]

    print(f"{'Distribution':<14} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurt':>10}")
    print("-" * 56)
    for dist, n, params in configs:
        x = sample(dist, n, params, seed=42)
        print(f"{dist:<14} {x.mean():>10.4f} {x.std():>10.4f} "
              f"{stats.skew(x):>10.4f} {stats.kurtosis(x):>10.4f}")