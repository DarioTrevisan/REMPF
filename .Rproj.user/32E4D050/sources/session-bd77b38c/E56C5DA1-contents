from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def bipartite_matching_cost_exact(x: np.ndarray, y: np.ndarray, p: float = 1.0) -> float:
    """
    Exact minimum bipartite matching cost between x and y using linear assignment.

    x, y: (n, d) arrays
    cost = sum_i ||x_i - y_{pi(i)}||^p
    """
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x{ x.shape } vs y{ y.shape }")
    d = cdist(x, y, metric="euclidean")
    if p != 1.0:
        d = d**p
    row_ind, col_ind = linear_sum_assignment(d)
    return float(d[row_ind, col_ind].sum())
