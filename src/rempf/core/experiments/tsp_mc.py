from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from rempf.core.points import sample_uniform
from rempf.core.tsp.optimal import TSPSolverConfig, solve_tsp_optimal_route, tsp_cost_given_tour


@dataclass(frozen=True)
class TSPMCConfig:
    n: int
    dim: int
    p_opt: float
    q_eval: tuple[float, ...]
    trials: int
    seed: int
    domain: str = "cube"

    held_karp_max_n: int = 20
    milp_max_n: int = 70
    milp_time_limit: float = 180.0
    allow_heuristic_fallback: bool = False
    heuristic_starts: int = 16
    progress: bool = False


def run_tsp_mc(cfg: TSPMCConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)

    costs = np.empty((cfg.trials, len(cfg.q_eval)), dtype=float)
    method_used: list[str] = []
    optimality_certified: list[bool] = []

    solver_cfg = TSPSolverConfig(
        held_karp_max_n=cfg.held_karp_max_n,
        milp_max_n=cfg.milp_max_n,
        milp_time_limit=cfg.milp_time_limit,
        allow_heuristic_fallback=cfg.allow_heuristic_fallback,
        heuristic_starts=cfg.heuristic_starts,
    )

    trial_iter = range(cfg.trials)
    if cfg.progress:
        trial_iter = tqdm(trial_iter, desc=f"tsp trials n={cfg.n}", leave=False)

    for t in trial_iter:
        x = sample_uniform(cfg.n, cfg.dim, rng=rng)
        sol = solve_tsp_optimal_route(x, p=cfg.p_opt, cfg=solver_cfg)

        for j, q in enumerate(cfg.q_eval):
            costs[t, j] = tsp_cost_given_tour(x, sol.tour, q=q)

        method_used.append(sol.method)
        optimality_certified.append(sol.optimality_certified)

    return {
        "costs": costs,
        "q_eval": np.array(cfg.q_eval, dtype=float),
        "p_opt": float(cfg.p_opt),
        "n": int(cfg.n),
        "dim": int(cfg.dim),
        "trials": int(cfg.trials),
        "seed": int(cfg.seed),
        "domain": cfg.domain,
        "method_used": np.array(method_used, dtype="U32"),
        "optimality_certified": np.array(optimality_certified, dtype=bool),
    }
