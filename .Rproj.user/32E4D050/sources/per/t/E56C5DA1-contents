from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def solve_perm_exact(x: np.ndarray, y: np.ndarray, p: float = 1.0) -> np.ndarray:
    """
    Return sigma (shape (n,)) minimizing sum_i ||x_i - y_{sigma(i)}||^p.
    """
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x{ x.shape } vs y{ y.shape }")
    dmat = cdist(x, y, metric="euclidean")
    if p != 1.0:
        dmat = dmat**p
    row_ind, col_ind = linear_sum_assignment(dmat)
    sigma = np.empty(x.shape[0], dtype=np.int64)
    sigma[row_ind] = col_ind
    return sigma


def eval_cost_given_perm(x: np.ndarray, y: np.ndarray, sigma: np.ndarray, q: float) -> float:
    """Compute sum_i ||x_i - y_{sigma(i)}||^q."""
    dif = x - y[sigma]
    dist = np.linalg.norm(dif, axis=1)
    return float(np.sum(dist**q))


def edge_lengths_given_perm(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    dif = x - y[sigma]
    return np.linalg.norm(dif, axis=1)

