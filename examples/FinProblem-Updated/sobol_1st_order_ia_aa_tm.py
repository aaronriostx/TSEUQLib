"""
Compare three ways of propagating sigma_hU's epistemic interval through the
main-effect Sobol indices S_k, S_Cp, S_rho, S_hU derived in
TSE_1st_order_derivation.md, evaluated at t=450s:

  1. Interval arithmetic (IA)       -- naive; treats the numerator and
                                        denominator's shared sigma_hU as if
                                        independent (the dependency problem)
  2. Affine arithmetic (AA)         -- tracks the shared noise symbol so
                                        numerator/denominator correlation
                                        isn't lost
  3. First-order Taylor model (TM)  -- linear expansion in sigma_hU + a
                                        rigorous 2nd-derivative remainder bound

k, Cp, rho have fixed, precisely-known variances -- only sigma_hU is
epistemic -- so every S_i here is a function of ONE scalar interval variable,
and an exact range (brute-force grid) is cheap to compute as ground truth.

sobol_1st_order_pie.py imports compute_Si_bounds() directly from here, so
both scripts always agree on the same numbers.
"""
import numpy as np
import sympy as sp
import fin_problem as fp
from fin_params_mixed import MEAN, STD, MEAN_hU, STD_hU, Tinf, Tw, b, d, L, x, t

# ── Nominal point + gradient at t=450s (same construction as TSE_1st_order_bounds.py) ─
Tinf_nom = (Tinf[0] + Tinf[1]) / 2
Tw_nom = (Tw[0] + Tw[1]) / 2
b_nom = (b[0] + b[1]) / 2
mu_X = np.array([MEAN[0], MEAN[1], MEAN[2], MEAN_hU])
m_y = np.array([Tinf_nom, Tw_nom, b_nom])


def eval_T_X(x_vec, y_vec):
    k_v, Cp_v, rho_v, hU_v = x_vec
    Tinf_v, Tw_v, b_v = y_vec
    T_out, _, _, _ = fp.analytic_solution(
        x, t, b_v, d, L, rho_v, Cp_v, k_v, hU_v, Tinf_v, Tw_v, Tinf_v,  # T0 = Tinf
    )
    return T_out


h_X = 1e-3 * mu_X
f = np.zeros(4)
for i in range(4):
    xp = mu_X.copy(); xp[i] += h_X[i]
    xm = mu_X.copy(); xm[i] -= h_X[i]
    f[i] = (eval_T_X(xp, m_y)[-1] - eval_T_X(xm, m_y)[-1]) / (2 * h_X[i])  # t=450s only

f_k, f_Cp, f_rho, f_hU = f
sigma_k, sigma_Cp, sigma_rho = STD[0], STD[1], STD[2]
sigma_hU_lo, sigma_hU_hi = STD_hU[0], STD_hU[1]

V_values = {
    "k": f_k**2 * sigma_k**2,
    "Cp": f_Cp**2 * sigma_Cp**2,
    "rho": f_rho**2 * sigma_rho**2,
}
V_fixed = sum(V_values.values())


def Si_numeric(which, sigma_hU):
    V_hU = f_hU**2 * sigma_hU**2
    num = V_hU if which == "hU" else V_values[which]
    return num / (V_fixed + V_hU)


# ============================================================
# Interval arithmetic
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
# Affine arithmetic
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


def compute_Si_bounds(which, grid_n=2000):
    """
    Returns a dict {method_name: (lo, hi)} for S_which(sigma_hU), plus an
    'exact' entry (brute-force grid) and a 'diagnostics' entry with the
    nominal value/derivative/remainder used by the Taylor model.
    """
    lo, hi = sigma_hU_lo, sigma_hU_hi
    mid, rad = (lo + hi) / 2, (hi - lo) / 2

    # ---- exact range (brute-force grid over the single variable sigma_hU) ----
    grid = np.linspace(lo, hi, grid_n)
    exact = (Si_numeric(which, grid).min(), Si_numeric(which, grid).max())

    # ---- interval arithmetic ----
    s_ia = Interval(lo, hi)
    Vhu_ia = (f_hU**2) * s_ia.square()
    den_ia = V_fixed + Vhu_ia
    num_ia = Vhu_ia if which == "hU" else Interval(V_values[which], V_values[which])
    result_ia = num_ia / den_ia
    interval_arithmetic = (result_ia.lo, result_ia.hi)

    # ---- affine arithmetic ----
    s_aa = Affine.variable(mid, rad)
    Vhu_aa = (f_hU**2) * s_aa.square()
    den_aa = V_fixed + Vhu_aa
    num_aa = Vhu_aa if which == "hU" else Affine(V_values[which])
    result_aa = num_aa.mul(den_aa.reciprocal())
    affine_arithmetic = result_aa.to_interval()

    # ---- first-order Taylor model (linear part + rigorous remainder bound) ----
    s = sp.symbols("sigma_hU", positive=True)
    Vhu_expr = f_hU**2 * s**2
    den_expr = V_fixed + Vhu_expr
    num_expr = Vhu_expr if which == "hU" else V_values[which]
    Si_expr = num_expr / den_expr

    Si_mid = float(Si_expr.subs(s, mid))
    dSi = sp.diff(Si_expr, s)
    dSi_mid = float(dSi.subs(s, mid))
    lin_lo = Si_mid - abs(dSi_mid) * rad
    lin_hi = Si_mid + abs(dSi_mid) * rad

    d2Si_func = sp.lambdify(s, sp.diff(Si_expr, s, 2), "numpy")
    H_max = np.abs(d2Si_func(grid)).max()
    remainder_bound = 0.5 * H_max * rad**2
    taylor_model = (lin_lo - remainder_bound, lin_hi + remainder_bound)

    return {
        "exact": exact,
        "interval_arithmetic": interval_arithmetic,
        "affine_arithmetic": affine_arithmetic,
        "taylor_model": taylor_model,
        "diagnostics": {
            "nominal": Si_mid,
            "derivative": dSi_mid,
            "hessian_abs_max": H_max,
            "remainder_bound": remainder_bound,
        },
    }


if __name__ == "__main__":
    display_names = {
        "exact": "Exact (grid)",
        "interval_arithmetic": "Interval arithmetic",
        "affine_arithmetic": "Affine arithmetic",
        "taylor_model": "Taylor model (1st ord.)",
    }
    for which in ["k", "Cp", "rho", "hU"]:
        results = compute_Si_bounds(which)
        print(f"\nS_{which} at t=450s")
        print(f"{'Method':<22}{'Lower':>10}{'Upper':>10}{'Width':>10}")
        print("-" * 52)
        for key, name in display_names.items():
            lo, hi = results[key]
            print(f"{name:<22}{lo:>10.4f}{hi:>10.4f}{(hi - lo):>10.4f}")
