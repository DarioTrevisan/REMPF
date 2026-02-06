from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from rempf.core.experiments.matching_mc import MatchMCConfig, run_matching_mc


@dataclass(frozen=True)
class SweepConfig:
    n_list: tuple[int, ...]
    dim: int
    p_opt: float
    q_eval: tuple[float, ...]
    trials: int
    seed: int
    domain: str = "cube"


def run_sweep(cfg: SweepConfig) -> Dict[str, Any]:
    all_costs = []

    # deterministic seed schedule per n
    for k, n in enumerate(cfg.n_list):
        res = run_matching_mc(
            MatchMCConfig(
                n=n,
                dim=cfg.dim,
                p_opt=cfg.p_opt,
                q_eval=cfg.q_eval,
                trials=cfg.trials,
                seed=cfg.seed + 1000 * k,
                domain=cfg.domain,
            )
        )
        all_costs.append(res["costs"])  # (trials, nq)

    return {
        "n_list": np.array(cfg.n_list, dtype=int),
        "dim": int(cfg.dim),
        "p_opt": float(cfg.p_opt),
        "q_eval": np.array(cfg.q_eval, dtype=float),
        "trials": int(cfg.trials),
        "seed": int(cfg.seed),
        "domain": cfg.domain,
        "costs": np.stack(all_costs, axis=0),  # (Nn, trials, nq)
    }

