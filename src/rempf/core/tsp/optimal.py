from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class TSPSolverConfig:
    held_karp_max_n: int = 20
    milp_max_n: int = 70
    milp_time_limit: float = 180.0
    allow_heuristic_fallback: bool = False
    heuristic_starts: int = 16


@dataclass(frozen=True)
class TSPSolveResult:
    tour: np.ndarray
    method: str
    optimality_certified: bool
    details: dict[str, Any]


def tsp_cost_given_tour(points: np.ndarray, tour: np.ndarray, q: float) -> float:
    """Compute cycle cost sum_i ||x_tour[i]-x_tour[i+1]||^q (including wrap-around)."""
    pts = np.asarray(points, dtype=float)
    ord_ = np.asarray(tour, dtype=np.int64)
    if pts.ndim != 2:
        raise ValueError(f"points must have shape (n,d), got {pts.shape}")
    if ord_.ndim != 1 or ord_.shape[0] != pts.shape[0]:
        raise ValueError("tour must be a permutation vector with length n")

    cyc = np.concatenate([ord_, ord_[:1]])
    dif = pts[cyc[1:]] - pts[cyc[:-1]]
    return float(np.sum(np.linalg.norm(dif, axis=1) ** float(q)))


def solve_tsp_optimal_route(
    points: np.ndarray,
    *,
    p: float,
    cfg: TSPSolverConfig | None = None,
) -> TSPSolveResult:
    """
    Compute a high-precision route for Euclidean TSP with objective exponent p.

    Strategy:
    1) Held-Karp exact DP for small n.
    2) Exact MILP (MTZ formulation) for moderate n.
    3) Optional high-quality heuristic fallback if exact path is not available.
    """
    cfg = cfg or TSPSolverConfig()
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2:
        raise ValueError(f"points must have shape (n,d), got {pts.shape}")
    n = pts.shape[0]
    if n < 3:
        raise ValueError("TSP requires n >= 3")

    w = _weighted_dist_matrix(pts, p=float(p))

    if n <= cfg.held_karp_max_n:
        tour, info = _held_karp_tour(w)
        return TSPSolveResult(
            tour=tour,
            method="held-karp",
            optimality_certified=True,
            details=info,
        )

    if n <= cfg.milp_max_n:
        milp_res = _solve_tsp_milp_mtz(w, time_limit=cfg.milp_time_limit)
        if milp_res is not None:
            tour, info = milp_res
            return TSPSolveResult(
                tour=tour,
                method="milp-mtz",
                optimality_certified=True,
                details=info,
            )

    if cfg.allow_heuristic_fallback:
        tour, info = _solve_tsp_heuristic(w, n_starts=cfg.heuristic_starts)
        return TSPSolveResult(
            tour=tour,
            method="heuristic-2opt",
            optimality_certified=False,
            details=info,
        )

    raise RuntimeError(
        "Could not certify an exact optimal TSP route for this n with current limits. "
        "Increase solver limits or enable heuristic fallback."
    )


def _weighted_dist_matrix(points: np.ndarray, p: float) -> np.ndarray:
    d = cdist(points, points, metric="euclidean")
    if p != 1.0:
        d = d**p
    np.fill_diagonal(d, np.inf)
    return d


def _held_karp_tour(w: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact DP for TSP cycle on complete graph with node 0 fixed as start."""
    n = w.shape[0]
    nodes = range(1, n)

    # DP[(mask, j)] = best cost from 0 to j visiting mask (subset of nodes) exactly once.
    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int] = {}

    for j in nodes:
        mask = 1 << (j - 1)
        dp[(mask, j)] = w[0, j]
        parent[(mask, j)] = 0

    for r in range(2, n):
        for subset in combinations(nodes, r):
            mask = 0
            for u in subset:
                mask |= 1 << (u - 1)

            for j in subset:
                prev_mask = mask & ~(1 << (j - 1))
                best_cost = np.inf
                best_prev = -1
                for k in subset:
                    if k == j:
                        continue
                    c = dp[(prev_mask, k)] + w[k, j]
                    if c < best_cost:
                        best_cost = c
                        best_prev = k
                dp[(mask, j)] = best_cost
                parent[(mask, j)] = best_prev

    full_mask = (1 << (n - 1)) - 1
    best_cycle = np.inf
    last = -1
    for j in nodes:
        c = dp[(full_mask, j)] + w[j, 0]
        if c < best_cycle:
            best_cycle = c
            last = j

    # Reconstruct Hamiltonian cycle order starting from 0.
    tour_rev = [last]
    mask = full_mask
    j = last
    while mask:
        pj = parent[(mask, j)]
        mask = mask & ~(1 << (j - 1))
        if pj == 0:
            break
        tour_rev.append(pj)
        j = pj

    tour = np.array([0] + list(reversed(tour_rev)), dtype=np.int64)
    if tour.shape[0] != n:
        # conservative fallback to brute-reconstruction from parent map
        tour = _repair_tour_from_successors(_tour_successor_from_cycle(tour, n), n)

    return tour, {"objective": float(best_cycle)}


def _solve_tsp_milp_mtz(w: np.ndarray, *, time_limit: float) -> tuple[np.ndarray, dict[str, Any]] | None:
    """
    Exact MILP with MTZ subtour elimination.

    Returns None when optimization does not reach certified optimality.
    """
    n = w.shape[0]
    arc_idx: dict[tuple[int, int], int] = {}
    arcs: list[tuple[int, int]] = []
    k = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            arc_idx[(i, j)] = k
            arcs.append((i, j))
            k += 1

    m = len(arcs)
    nu = n - 1
    total_vars = m + nu

    c = np.zeros(total_vars, dtype=float)
    for (i, j), idx in arc_idx.items():
        c[idx] = w[i, j]

    # x_ij binary; u_i continuous for i=1..n-1.
    integrality = np.zeros(total_vars, dtype=np.int8)
    integrality[:m] = 1

    lb = np.zeros(total_vars, dtype=float)
    ub = np.ones(total_vars, dtype=float)
    if nu > 0:
        lb[m:] = 1.0
        ub[m:] = float(n - 1)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    # Out-degree constraints: sum_j x_ij = 1
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rows.append(row)
            cols.append(arc_idx[(i, j)])
            vals.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        row += 1

    # In-degree constraints: sum_i x_ij = 1
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            rows.append(row)
            cols.append(arc_idx[(i, j)])
            vals.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        row += 1

    # MTZ: u_i - u_j + (n-1) x_ij <= n-2 for i,j in {1,..,n-1}, i!=j
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            rows.append(row)
            cols.append(m + (i - 1))
            vals.append(1.0)

            rows.append(row)
            cols.append(m + (j - 1))
            vals.append(-1.0)

            rows.append(row)
            cols.append(arc_idx[(i, j)])
            vals.append(float(n - 1))

            lower.append(-np.inf)
            upper.append(float(n - 2))
            row += 1

    A = coo_matrix((vals, (rows, cols)), shape=(row, total_vars)).tocsr()
    con = LinearConstraint(A, np.array(lower, dtype=float), np.array(upper, dtype=float))

    res = milp(
        c=c,
        constraints=con,
        bounds=Bounds(lb, ub),
        integrality=integrality,
        options={"time_limit": float(time_limit)},
    )

    if res.status != 0 or res.x is None:
        return None

    x = res.x[:m]
    succ = np.full(n, -1, dtype=np.int64)
    for (i, j), idx in arc_idx.items():
        if x[idx] > 0.5:
            succ[i] = j

    tour = _repair_tour_from_successors(succ, n)
    info = {
        "objective": float(res.fun),
        "status": int(res.status),
        "message": str(res.message),
        "mip_node_count": int(getattr(res, "mip_node_count", -1)),
    }
    return tour, info


def _solve_tsp_heuristic(w: np.ndarray, *, n_starts: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Nearest-neighbor multi-start + 2-opt local search."""
    n = w.shape[0]
    n_starts = max(1, min(n, int(n_starts)))

    best_tour = None
    best_cost = np.inf
    for s in range(n_starts):
        tour = _nearest_neighbor_tour(w, start=s)
        tour, cost = _two_opt_improve(tour, w)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    assert best_tour is not None
    return best_tour, {"objective": float(best_cost), "starts": int(n_starts)}


def _nearest_neighbor_tour(w: np.ndarray, *, start: int) -> np.ndarray:
    n = w.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: w[cur, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return np.array(tour, dtype=np.int64)


def _two_opt_improve(tour: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    n = tour.shape[0]
    best = tour.copy()
    best_cost = _cycle_cost_from_w(best, w)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                cand = best.copy()
                cand[i : j + 1] = cand[i : j + 1][::-1]
                c = _cycle_cost_from_w(cand, w)
                if c + 1e-15 < best_cost:
                    best, best_cost = cand, c
                    improved = True
    return best, float(best_cost)


def _cycle_cost_from_w(tour: np.ndarray, w: np.ndarray) -> float:
    cyc = np.concatenate([tour, tour[:1]])
    return float(np.sum(w[cyc[:-1], cyc[1:]]))


def _repair_tour_from_successors(succ: np.ndarray, n: int) -> np.ndarray:
    """Build an order [0, ...] from successor array and validate it's a single cycle."""
    tour = np.empty(n, dtype=np.int64)
    seen = set()
    cur = 0
    for k in range(n):
        if cur in seen or cur < 0 or cur >= n:
            raise RuntimeError("Invalid tour reconstruction from solver output")
        seen.add(cur)
        tour[k] = cur
        cur = int(succ[cur])

    if cur != 0 or len(seen) != n:
        raise RuntimeError("Solver output contains subtours or invalid cycle")
    return tour


def _tour_successor_from_cycle(tour: np.ndarray, n: int) -> np.ndarray:
    succ = np.full(n, -1, dtype=np.int64)
    cyc = np.concatenate([tour, tour[:1]])
    for i in range(n):
        succ[int(cyc[i])] = int(cyc[i + 1])
    return succ
