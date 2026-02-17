from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from tqdm.auto import tqdm

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
    progress: bool = False


def run_sweep(cfg: SweepConfig) -> Dict[str, Any]:
    all_costs = []

    # deterministic seed schedule per n
    n_iter = cfg.n_list
    if cfg.progress:
        n_iter = tqdm(n_iter, desc="match n-grid")

    for k, n in enumerate(n_iter):
        res = run_matching_mc(
            MatchMCConfig(
                n=n,
                dim=cfg.dim,
                p_opt=cfg.p_opt,
                q_eval=cfg.q_eval,
                trials=cfg.trials,
                seed=cfg.seed + 1000 * k,
                domain=cfg.domain,
                progress=cfg.progress,
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
