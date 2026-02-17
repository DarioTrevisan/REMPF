from __future__ import annotations

from itertools import permutations

import numpy as np

from rempf.core.tsp.optimal import TSPSolverConfig, solve_tsp_optimal_route, tsp_cost_given_tour


def _bruteforce_tsp_cost(points: np.ndarray, p: float) -> float:
    n = points.shape[0]
    best = np.inf
    # Fix node 0 as start to avoid rotational duplicates.
    for perm in permutations(range(1, n)):
        tour = np.array((0, *perm), dtype=np.int64)
        c = tsp_cost_given_tour(points, tour, q=p)
        if c < best:
            best = c
    return float(best)


def test_tsp_exact_matches_bruteforce_small() -> None:
    rng = np.random.default_rng(1)
    points = rng.random((8, 2))

    sol = solve_tsp_optimal_route(
        points,
        p=1.7,
        cfg=TSPSolverConfig(held_karp_max_n=20, milp_max_n=0),
    )
    exact = _bruteforce_tsp_cost(points, p=1.7)
    got = tsp_cost_given_tour(points, sol.tour, q=1.7)

    assert sol.optimality_certified
    assert np.isclose(got, exact, rtol=1e-12, atol=1e-12)


def test_tsp_q_cost_nonnegative() -> None:
    rng = np.random.default_rng(2)
    points = rng.random((10, 3))

    sol = solve_tsp_optimal_route(
        points,
        p=1.0,
        cfg=TSPSolverConfig(held_karp_max_n=20, milp_max_n=0),
    )
    c = tsp_cost_given_tour(points, sol.tour, q=3.0)

    assert c >= 0.0
