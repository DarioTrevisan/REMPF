from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _pairwise_distances(x: np.ndarray, y: np.ndarray, *, domain: str = "cube") -> np.ndarray:
    if domain == "cube":
        return cdist(x, y, metric="euclidean")
    if domain == "torus":
        delta = np.abs(x[:, None, :] - y[None, :, :])
        delta = np.minimum(delta, 1.0 - delta)
        return np.linalg.norm(delta, axis=2)
    raise ValueError(f"Unsupported domain: {domain}")


def _matched_distances(x: np.ndarray, y: np.ndarray, sigma: np.ndarray, *, domain: str = "cube") -> np.ndarray:
    dif = x - y[sigma]
    if domain == "torus":
        dif = np.where(dif > 0.5, dif - 1.0, dif)
        dif = np.where(dif < -0.5, dif + 1.0, dif)
    elif domain != "cube":
        raise ValueError(f"Unsupported domain: {domain}")
    return np.linalg.norm(dif, axis=1)


def solve_perm_exact(
    x: np.ndarray,
    y: np.ndarray,
    p: float = 1.0,
    *,
    domain: str = "cube",
) -> np.ndarray:
    """
    Return sigma (shape (n,)) minimizing sum_i ||x_i - y_{sigma(i)}||^p.
    """
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x{ x.shape } vs y{ y.shape }")
    dmat = _pairwise_distances(x, y, domain=domain)
    if p != 1.0:
        dmat = dmat**p
    row_ind, col_ind = linear_sum_assignment(dmat)
    sigma = np.empty(x.shape[0], dtype=np.int64)
    sigma[row_ind] = col_ind
    return sigma


def eval_cost_given_perm(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    q: float,
    *,
    domain: str = "cube",
) -> float:
    """Compute sum_i ||x_i - y_{sigma(i)}||^q."""
    dist = _matched_distances(x, y, sigma, domain=domain)
    return float(np.sum(dist**q))


def edge_lengths_given_perm(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    domain: str = "cube",
) -> np.ndarray:
    return _matched_distances(x, y, sigma, domain=domain)


def bipartite_matching_cost_exact(
    x: np.ndarray,
    y: np.ndarray,
    p: float = 1.0,
    *,
    domain: str = "cube",
) -> float:
    """Convenience wrapper: solve exactly and return the optimal p-cost."""
    sigma = solve_perm_exact(x, y, p=p, domain=domain)
    return eval_cost_given_perm(x, y, sigma, q=p, domain=domain)
