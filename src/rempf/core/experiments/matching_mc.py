from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from tqdm.auto import tqdm

from rempf.core.points import sample_uniform
from rempf.core.matching.exact import solve_perm_exact, eval_cost_given_perm


@dataclass(frozen=True)
class MatchMCConfig:
    n: int
    dim: int
    p_opt: float                # exponent used to compute sigma*_p
    q_eval: tuple[float, ...]   # exponents evaluated on sigma*_p
    trials: int
    seed: int
    domain: str = "cube"        # future-proof: "cube" or "torus"
    progress: bool = False


def run_matching_mc(cfg: MatchMCConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)

    costs = np.empty((cfg.trials, len(cfg.q_eval)), dtype=float)

    trial_iter = range(cfg.trials)
    if cfg.progress:
        trial_iter = tqdm(trial_iter, desc=f"match trials n={cfg.n}", leave=False)

    for t in trial_iter:
        x = sample_uniform(cfg.n, cfg.dim, rng=rng)
        y = sample_uniform(cfg.n, cfg.dim, rng=rng)

        sigma = solve_perm_exact(x, y, p=cfg.p_opt, domain=cfg.domain)
        for j, q in enumerate(cfg.q_eval):
            costs[t, j] = eval_cost_given_perm(x, y, sigma, q=q, domain=cfg.domain)

    return {
        "costs": costs,  # (trials, nq)
        "q_eval": np.array(cfg.q_eval, dtype=float),
        "p_opt": float(cfg.p_opt),
        "n": int(cfg.n),
        "dim": int(cfg.dim),
        "trials": int(cfg.trials),
        "seed": int(cfg.seed),
        "domain": cfg.domain,
    }
