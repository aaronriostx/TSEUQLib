"""
Compare four ways of bounding the first-order Sobol index

    S1(sigma1, sigma2) = sigma1^2 / (sigma1^2 + 4*sigma2^2)

over sigma1, sigma2 in [0.9, 1.1], for Y = X1 + 2*X2, X1 ~ N(0, sigma1),
X2 ~ N(0, sigma2), independent:

  1. Exact range (brute-force grid, ground truth)
  2. Interval arithmetic (IA)       -- naive, ignores repeated-variable dependency
  3. Affine arithmetic (AA)         -- tracks linear correlation via shared noise symbols
  4. First-order Taylor model (TM)  -- linear Taylor expansion + rigorous remainder bound

All the math is exposed through compute_S1_bounds(), which sobol_pie_uncertainty.py
imports directly -- so both scripts always agree on the same numbers.

Tweak s1_bounds / s2_bounds below to explore other intervals. Running this
file directly prints the comparison table and saves the enclosure plot.
"""

import numpy as np
import sympy as sp


# ============================================================
# 2. Interval arithmetic
# ============================================================
class Interval:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __add__(self, other):
        other = other if isinstance(other, Interval) else Interval(other, other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Interval) else Interval(other, other)
        vals = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi]
        return Interval(min(vals), max(vals))

    __rmul__ = __mul__

    def square(self):
        if self.lo >= 0:
            return Interval(self.lo**2, self.hi**2)
        if self.hi <= 0:
            return Interval(self.hi**2, self.lo**2)
        return Interval(0.0, max(self.lo**2, self.hi**2))

    def __truediv__(self, other):
        # assumes other is strictly positive
        return Interval(self.lo / other.hi, self.hi / other.lo)

    def __repr__(self):
        return f"[{self.lo:.5f}, {self.hi:.5f}]"


# ============================================================
# 3. Affine arithmetic
# ============================================================
class Affine:
    _counter = [0]

    def __init__(self, center, coeffs=None):
        self.center = center
        self.coeffs = dict(coeffs) if coeffs else {}

    @classmethod
    def variable(cls, center, radius):
        cls._counter[0] += 1
        return cls(center, {cls._counter[0]: radius})

    @classmethod
    def _new_symbol(cls):
        cls._counter[0] += 1
        return cls._counter[0]

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Affine(self.center + other, self.coeffs)
        c = dict(self.coeffs)
        for k, v in other.coeffs.items():
            c[k] = c.get(k, 0.0) + v
        return Affine(self.center + other.center, c)

    __radd__ = __add__

    def __rmul__(self, k):
        return Affine(self.center * k, {i: v * k for i, v in self.coeffs.items()})

    def square(self):
        # x = x0 + sum(x_i eps_i);  x^2 = x0^2 + 2 x0 sum(x_i eps_i) + (sum x_i eps_i)^2
        # the quadratic cross term lies in [0, (sum|x_i|)^2] -- bound it with one fresh symbol
        radsum = sum(abs(v) for v in self.coeffs.values())
        new_center = self.center**2 + 0.5 * radsum**2
        c = {k: 2 * self.center * v for k, v in self.coeffs.items()}
        c[Affine._new_symbol()] = 0.5 * radsum**2
        return Affine(new_center, c)

    def mul(self, other):
        # general affine * affine; cross term bounded by (sum|x_i|)(sum|y_i|)
        keys = set(self.coeffs) | set(other.coeffs)
        c = {}
        for k in keys:
            c[k] = self.coeffs.get(k, 0.0) * other.center + other.coeffs.get(k, 0.0) * self.center
        radx = sum(abs(v) for v in self.coeffs.values())
        rady = sum(abs(v) for v in other.coeffs.values())
        c[Affine._new_symbol()] = radx * rady
        return Affine(self.center * other.center, c)

    def reciprocal(self):
        # linearize 1/x on x's own interval range [a,b], a>0, using the
        # min-gap chord approximation: max gap of the chord above 1/x is
        # (1/sqrt(a) - 1/sqrt(b))^2, attained at x* = sqrt(a*b).
        a, b = self.to_interval()
        alpha = -1.0 / (a * b)
        max_gap = (1 / np.sqrt(a) - 1 / np.sqrt(b))**2
        const = (1 / a + 1 / b) - 0.5 * max_gap
        c = {k: alpha * v for k, v in self.coeffs.items()}
        c[Affine._new_symbol()] = 0.5 * max_gap
        return Affine(const + alpha * self.center, c)

    def to_interval(self):
        rad = sum(abs(v) for v in self.coeffs.values())
        return self.center - rad, self.center + rad

    def __repr__(self):
        lo, hi = self.to_interval()
        return f"[{lo:.5f}, {hi:.5f}]"


def S1_numeric(s1, s2, f1=1.0, f2=2.0):
    return (f1**2 * s1**2) / (f1**2 * s1**2 + f2**2 * s2**2)


def compute_S1_bounds(s1_bounds=(0.9, 1.1), s2_bounds=(0.9, 1.1), f1=1.0, f2=2.0, grid_n=2000):
    """
    Returns a dict {method_name: (lo, hi)} for S1(sigma1, sigma2), plus a
    'diagnostics' entry with the gradient/Hessian/remainder details used
    by the Taylor model, for anyone who wants to print/inspect them.
    """
    s1_lo, s1_hi = s1_bounds
    s2_lo, s2_hi = s2_bounds
    mid1, rad1 = (s1_lo + s1_hi) / 2, (s1_hi - s1_lo) / 2
    mid2, rad2 = (s2_lo + s2_hi) / 2, (s2_hi - s2_lo) / 2

    # ---- 1. Exact range (brute-force grid) ----
    g1 = np.linspace(s1_lo, s1_hi, grid_n)
    g2 = np.linspace(s2_lo, s2_hi, grid_n)
    G1, G2 = np.meshgrid(g1, g2)
    grid_vals = S1_numeric(G1, G2, f1, f2)
    exact = (grid_vals.min(), grid_vals.max())

    # ---- 2. Interval arithmetic ----
    S1_ia = Interval(s1_lo, s1_hi)
    S2_ia = Interval(s2_lo, s2_hi)
    num_ia = (f1**2) * S1_ia.square()
    den_ia = num_ia + (f2**2) * S2_ia.square()
    result_ia = num_ia / den_ia
    interval_arithmetic = (result_ia.lo, result_ia.hi)

    # ---- 3. Affine arithmetic ----
    s1_aa = Affine.variable(mid1, rad1)
    s2_aa = Affine.variable(mid2, rad2)
    num_aa = (f1**2) * s1_aa.square()
    den_aa = num_aa + (f2**2) * s2_aa.square()
    result_aa = num_aa.mul(den_aa.reciprocal())
    affine_arithmetic = result_aa.to_interval()

    # ---- 4. First-order Taylor model (linear part + rigorous remainder bound) ----
    sig1, sig2 = sp.symbols('sigma1 sigma2', positive=True)
    S1_expr = (f1**2 * sig1**2) / (f1**2 * sig1**2 + f2**2 * sig2**2)
    grad = [sp.diff(S1_expr, v) for v in (sig1, sig2)]
    hess = [[sp.diff(S1_expr, vi, vj) for vj in (sig1, sig2)] for vi in (sig1, sig2)]

    S1_mid = float(S1_expr.subs({sig1: mid1, sig2: mid2}))
    g1_mid = float(grad[0].subs({sig1: mid1, sig2: mid2}))
    g2_mid = float(grad[1].subs({sig1: mid1, sig2: mid2}))

    lin_lo = S1_mid - abs(g1_mid) * rad1 - abs(g2_mid) * rad2
    lin_hi = S1_mid + abs(g1_mid) * rad1 + abs(g2_mid) * rad2

    hess_funcs = [[sp.lambdify((sig1, sig2), hess[i][j], 'numpy') for j in range(2)] for i in range(2)]
    H11 = np.abs(hess_funcs[0][0](G1, G2)).max()
    H12 = np.abs(hess_funcs[0][1](G1, G2)).max()
    H22 = np.abs(hess_funcs[1][1](G1, G2)).max()

    remainder_bound = 0.5 * (H11 * rad1**2 + 2 * H12 * rad1 * rad2 + H22 * rad2**2)
    taylor_model = (lin_lo - remainder_bound, lin_hi + remainder_bound)

    return {
        "exact": exact,
        "interval_arithmetic": interval_arithmetic,
        "affine_arithmetic": affine_arithmetic,
        "taylor_model": taylor_model,
        "diagnostics": {
            "nominal": S1_numeric(mid1, mid2, f1, f2),
            "gradient": (g1_mid, g2_mid),
            "hessian_abs_max": (H11, H12, H22),
            "remainder_bound": remainder_bound,
        },
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    results = compute_S1_bounds()
    diag = results["diagnostics"]

    print(f"{'Method':<22}{'Lower':>10}{'Upper':>10}{'Width':>10}")
    print("-" * 52)
    display_names = {
        "exact": "Exact (grid)",
        "interval_arithmetic": "Interval arithmetic",
        "affine_arithmetic": "Affine arithmetic",
        "taylor_model": "Taylor model (1st ord.)",
    }
    for key, name in display_names.items():
        lo, hi = results[key]
        print(f"{name:<22}{lo:>10.4f}{hi:>10.4f}{(hi - lo):>10.4f}")

    print()
    print(f"Nominal S1 at (sigma1,sigma2)=(1,1): {diag['nominal']:.4f}")
    print(f"Gradient at midpoint: dS1/dsigma1={diag['gradient'][0]:.4f}, dS1/dsigma2={diag['gradient'][1]:.4f}")
    H11, H12, H22 = diag["hessian_abs_max"]
    print(f"Max |Hessian| over box: H11={H11:.4f}, H12={H12:.4f}, H22={H22:.4f}")
    print(f"Remainder bound: {diag['remainder_bound']:.4f}")

    # ---- Plot: side-by-side comparison of the four enclosures ----
    methods = ["Exact\n(grid)", "Interval\narithmetic", "Affine\narithmetic", "Taylor model\n(1st order)"]
    los = [results[k][0] for k in display_names]
    his = [results[k][1] for k in display_names]
    colors = ["#1D9E75", "#D85A30", "#378ADD", "#7F77DD"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(methods))
    for i, (lo, hi, c) in enumerate(zip(los, his, colors)):
        ax.plot([lo, hi], [i, i], color=c, linewidth=6, solid_capstyle='butt')
        ax.plot([lo, hi], [i, i], 'o', color=c, markersize=6)
        ax.text(hi + 0.005, i, f"[{lo:.3f}, {hi:.3f}]", va='center', fontsize=10)

    ax.axvline(diag["nominal"], color='gray', linestyle='--', linewidth=1, label='Nominal S1 (sigma=1)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('S1 (first-order Sobol index for X1)')
    ax.set_title('Enclosures of S1 over sigma1, sigma2 in [0.9, 1.1]')
    ax.set_xlim(0.05, 0.35)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('sobol_ia_aa_tm.png', dpi=150, bbox_inches='tight')
    plt.show()