from __future__ import annotations

import argparse
import json
from pathlib import Path

from rempf.core.experiments.tsp_sweeps import TSPSweepConfig, run_tsp_sweep
from rempf.io.npz import load_npz, save_npz
from rempf.plots.match_sweep_plots import (
    plot_sweep_concentration,
    plot_sweep_normalized,
    plot_sweep_variance,
)


def _parse_csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _parse_csv_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def add_tsp_subparser(sub) -> None:
    p = sub.add_parser("tsp", help="Euclidean TSP experiments")
    sub2 = p.add_subparsers(dest="tsp_cmd", required=True)

    sweep = sub2.add_parser("sweep", help="Run MC sweep over n for p-optimal TSP route")
    sweep.add_argument("--dim", type=int, default=2)
    sweep.add_argument("--p-opt", type=float, required=True)
    sweep.add_argument("--q-eval", type=str, required=True, help="Comma-separated list, e.g. 1,2,3")
    sweep.add_argument("--n-list", type=str, required=True, help="Comma-separated n values")
    sweep.add_argument("--trials", type=int, default=20)
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--domain", choices=["cube"], default="cube")
    sweep.add_argument("--out", type=str, required=True)
    sweep.add_argument("--progress", action="store_true", help="Show progress bars")
    sweep.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional sidecar JSON path (default: same stem as --out with .json)",
    )

    sweep.add_argument("--held-karp-max-n", type=int, default=20)
    sweep.add_argument("--milp-max-n", type=int, default=70)
    sweep.add_argument("--milp-time-limit", type=float, default=180.0)
    sweep.add_argument("--heuristic-starts", type=int, default=16)
    sweep.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Allow non-certified fallback if exact solver cannot certify optimality",
    )
    sweep.set_defaults(func=cmd_sweep)

    ps = sub2.add_parser("plot-sweep", help="Plot normalized curves + concentration from TSP sweep .npz")
    ps.add_argument("npz", type=str)
    ps.add_argument("--out-norm", type=str, required=True)
    ps.add_argument("--out-conc", type=str, required=True)
    ps.add_argument("--out-var", type=str, default=None)
    ps.add_argument("--quantiles", type=str, default="0.1,0.9")
    ps.add_argument(
        "--error-bars",
        choices=["none", "std", "se"],
        default="none",
        help="Optional error bars around mean normalized curve",
    )
    ps.set_defaults(func=cmd_plot_sweep)


def cmd_sweep(args: argparse.Namespace) -> None:
    n_list = _parse_csv_ints(args.n_list)
    q_eval = _parse_csv_floats(args.q_eval)

    cfg = TSPSweepConfig(
        n_list=n_list,
        dim=args.dim,
        p_opt=float(args.p_opt),
        q_eval=q_eval,
        trials=args.trials,
        seed=args.seed,
        domain=args.domain,
        held_karp_max_n=args.held_karp_max_n,
        milp_max_n=args.milp_max_n,
        milp_time_limit=args.milp_time_limit,
        heuristic_starts=args.heuristic_starts,
        allow_heuristic_fallback=bool(args.allow_heuristic_fallback),
        progress=bool(args.progress),
    )
    res = run_tsp_sweep(cfg)

    params = {
        "problem": "euclidean_tsp_sweep",
        "dim": res["dim"],
        "p_opt": res["p_opt"],
        "q_eval": [float(x) for x in res["q_eval"]],
        "n_list": [int(x) for x in res["n_list"]],
        "trials": res["trials"],
        "seed": res["seed"],
        "domain": res["domain"],
        "held_karp_max_n": int(args.held_karp_max_n),
        "milp_max_n": int(args.milp_max_n),
        "milp_time_limit": float(args.milp_time_limit),
        "heuristic_starts": int(args.heuristic_starts),
        "allow_heuristic_fallback": bool(args.allow_heuristic_fallback),
    }

    save_npz(
        args.out,
        arrays={
            "n_list": res["n_list"],
            "q_eval": res["q_eval"],
            "costs": res["costs"],
            "method_used": res["method_used"],
            "optimality_certified": res["optimality_certified"],
        },
        params=params,
    )

    out_json = args.out_json or str(Path(args.out).with_suffix(".json"))
    summary = {
        "notes": [
            "Monte Carlo sweep over Euclidean random points in [0,1]^d.",
            "For each instance, route is optimized for p_opt then evaluated at all q_eval.",
            "Optimality certification flags indicate whether each run was solved exactly.",
        ],
        "problem": "euclidean_tsp_sweep",
        "n_values": [int(x) for x in res["n_list"]],
        "num_points_per_instance": [int(x) for x in res["n_list"]],
        "dimension_d": int(res["dim"]),
        "p_opt": float(res["p_opt"]),
        "q_eval": [float(x) for x in res["q_eval"]],
        "trials_per_n": int(res["trials"]),
        "seed_base": int(res["seed"]),
        "domain": str(res["domain"]),
        "held_karp_max_n": int(args.held_karp_max_n),
        "milp_max_n": int(args.milp_max_n),
        "milp_time_limit_sec": float(args.milp_time_limit),
        "heuristic_starts": int(args.heuristic_starts),
        "allow_heuristic_fallback": bool(args.allow_heuristic_fallback),
        "methods_used": sorted({str(x) for x in res["method_used"].flatten().tolist()}),
        "all_optimality_certified": bool(res["optimality_certified"].all()),
        "npz_output": str(args.out),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def cmd_plot_sweep(args: argparse.Namespace) -> None:
    d = load_npz(args.npz)
    n_list = d["n_list"]
    q_eval = d["q_eval"]
    costs = d["costs"]
    dim = int(d["params"]["dim"])
    p_opt = float(d["params"]["p_opt"])

    qlo, qhi = (float(x) for x in args.quantiles.split(","))

    title_norm = f"TSP normalized costs on tau*_p (dim={dim}, p_opt={p_opt:g})"
    plot_sweep_normalized(
        n_list=n_list,
        costs=costs,
        dim=dim,
        q_eval=q_eval,
        title=title_norm,
        out=args.out_norm,
        quantiles=(qlo, qhi),
        error_bars=args.error_bars,
    )

    title_conc = f"TSP concentration proxy std/mean (dim={dim}, p_opt={p_opt:g})"
    plot_sweep_concentration(
        n_list=n_list,
        costs=costs,
        dim=dim,
        q_eval=q_eval,
        title=title_conc,
        out=args.out_conc,
    )

    if args.out_var:
        title_var = f"TSP var of normalized cost (dim={dim}, p_opt={p_opt:g})"
        plot_sweep_variance(
            n_list=n_list,
            costs=costs,
            dim=dim,
            q_eval=q_eval,
            title=title_var,
            out=args.out_var,
        )
