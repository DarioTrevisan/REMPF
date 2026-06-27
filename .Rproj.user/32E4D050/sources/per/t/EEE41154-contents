from __future__ import annotations
import numpy as np

def r_dp(n: int, d: int, p: float) -> float:
    """
    Predicted scaling r(d,p)(n) for total optimal matching cost.

    This encodes the piecewise conjectural rates you described.
    Returned value is the leading-order scale (no unknown constant).
    """
    n = float(n)

    if d == 2:
        # Exceptional log regime you stated for p>=1
        if p >= 1.0:
            return n * (np.log(n) / n) ** (p / 2.0)
        # "regular" regime for p < d/2 = 1
        if p < 1.0:
            return n * n ** (-p / 2.0)

    if d == 1:
        # stated special case
        if abs(p - 0.5) < 1e-12:
            return n * (np.log(n) / n) ** 0.5
        if p > 0.5:
            return n**(1/2)
        # regular (heuristic) rate: n * n^{-p}
        return n * n ** (-p / 1.0)

    # d >= 3
    # regular rate (as stated): n * n^{-p/d}
    return n * n ** (-p / float(d))


def normalize_cost(cost: float, n: int, d: int, p: float) -> float:
    """Return cost / r(d,p)(n). Expect ~ constant if scaling is correct."""
    return float(cost) / float(r_dp(n, d, p))
