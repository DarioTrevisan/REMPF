from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from rempf.core.experiments.tsp_mc import TSPMCConfig, run_tsp_mc


@dataclass(frozen=True)
class TSPSweepConfig:
    n_list: tuple[int, ...]
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


def run_tsp_sweep(cfg: TSPSweepConfig) -> dict[str, Any]:
    all_costs = []
    method_used_all = []
    optimality_all = []

    n_iter = cfg.n_list
    if cfg.progress:
        n_iter = tqdm(n_iter, desc="tsp n-grid")

    for k, n in enumerate(n_iter):
        res = run_tsp_mc(
            TSPMCConfig(
                n=n,
                dim=cfg.dim,
                p_opt=cfg.p_opt,
                q_eval=cfg.q_eval,
                trials=cfg.trials,
                seed=cfg.seed + 1000 * k,
                domain=cfg.domain,
                held_karp_max_n=cfg.held_karp_max_n,
                milp_max_n=cfg.milp_max_n,
                milp_time_limit=cfg.milp_time_limit,
                allow_heuristic_fallback=cfg.allow_heuristic_fallback,
                heuristic_starts=cfg.heuristic_starts,
                progress=cfg.progress,
            )
        )
        all_costs.append(res["costs"])
        method_used_all.append(res["method_used"])
        optimality_all.append(res["optimality_certified"])

    return {
        "n_list": np.array(cfg.n_list, dtype=int),
        "dim": int(cfg.dim),
        "p_opt": float(cfg.p_opt),
        "q_eval": np.array(cfg.q_eval, dtype=float),
        "trials": int(cfg.trials),
        "seed": int(cfg.seed),
        "domain": cfg.domain,
        "costs": np.stack(all_costs, axis=0),
        "method_used": np.stack(method_used_all, axis=0),
        "optimality_certified": np.stack(optimality_all, axis=0),
    }
