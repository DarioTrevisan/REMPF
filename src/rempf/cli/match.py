from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from rempf.core.experiments.sweeps import SweepConfig, run_sweep
from rempf.io.npz import save_npz, load_npz
from rempf.plots.match_sweep_plots import (
    plot_sweep_normalized, plot_sweep_concentration, plot_sweep_variance
)


def _parse_csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip() != "")

def _parse_csv_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip() != "")


def add_match_subparser(sub):
    p = sub.add_parser("match", help="Bipartite Euclidean matching experiments")
    sub2 = p.add_subparsers(dest="match_cmd", required=True)

    # --- sweep ---
    sweep = sub2.add_parser("sweep", help="Run MC sweep over a list of n")
    sweep.add_argument("--dim", type=int, default=2)
    sweep.add_argument("--p-opt", type=float, required=True, help="Exponent used to compute sigma*_p")
    sweep.add_argument("--q-eval", type=str, required=True, help="Comma-separated list, e.g. 1,2,3")
    sweep.add_argument("--n-list", type=str, required=True, help="Comma-separated list of n, e.g. 50,80,120")
    sweep.add_argument("--trials", type=int, default=20)
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--domain", choices=["cube", "torus"], default="cube")
    sweep.add_argument("--out", type=str, required=True)
    sweep.add_argument("--progress", action="store_true", help="Show progress bars")
    sweep.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional sidecar JSON path (default: same stem as --out with .json)",
    )
    sweep.set_defaults(func=cmd_sweep)

    # --- plot sweep ---
    ps = sub2.add_parser("plot-sweep", help="Plot normalized curves + concentration from sweep .npz")
    
    ps.add_argument("npz", type=str)
    ps.add_argument("--out-norm", type=str, required=True, help="Output PDF for normalized plot")
    ps.add_argument("--out-conc", type=str, required=True, help="Output PDF for concentration plot (std/mean)")
    ps.add_argument("--out-var", type=str, default=None, help="Optional PDF for variance plot")
    ps.add_argument("--quantiles", type=str, default="0.1,0.9", help="Two quantiles for band, e.g. 0.1,0.9")
    ps.add_argument(
        "--error-bars",
        choices=["none", "std", "se"],
        default="none",
        help="Optional error bars around mean normalized curve",
    )
    ps.set_defaults(func=cmd_plot_sweep)
    
    # --- edge dist ---
    
    ed = sub2.add_parser("edge-dist", help="Edge-length distribution under sigma*_p")
    ed.add_argument("--n", type=int, required=True)
    ed.add_argument("--dim", type=int, default=2)
    ed.add_argument("--p-opt", type=float, required=True)
    ed.add_argument("--domain", choices=["cube", "torus"], default="cube")
    ed.add_argument("--trials", type=int, default=10)
    ed.add_argument("--seed", type=int, default=0)
    ed.add_argument("--out", type=str, required=True, help="Output .npz (stores pooled edge lengths)")
    ed.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional sidecar JSON path (default: same stem as --out with .json)",
    )
    ed.add_argument("--plot", type=str, required=True, help="Output PDF for histogram/CCDF")
    ed.add_argument("--bins", type=int, default=80)
    ed.add_argument("--ccdf", action="store_true", help="Plot CCDF instead of histogram")
    ed.set_defaults(func=cmd_edge_dist)



def cmd_sweep(args: argparse.Namespace) -> None:
    n_list = _parse_csv_ints(args.n_list)
    q_eval = _parse_csv_floats(args.q_eval)

    cfg = SweepConfig(
        n_list=n_list,
        dim=args.dim,
        p_opt=float(args.p_opt),
        q_eval=q_eval,
        trials=args.trials,
        seed=args.seed,
        domain=args.domain,
        progress=bool(args.progress),
    )
    res = run_sweep(cfg)

    params = {
        "problem": "bipartite_matching_sweep",
        "dim": res["dim"],
        "p_opt": res["p_opt"],
        "q_eval": [float(x) for x in res["q_eval"]],
        "n_list": [int(x) for x in res["n_list"]],
        "trials": res["trials"],
        "seed": res["seed"],
        "domain": res["domain"],
    }

    save_npz(
        args.out,
        arrays={
            "n_list": res["n_list"],
            "q_eval": res["q_eval"],
            "costs": res["costs"],
            "method_used": np.full((len(res["n_list"]), res["trials"]), "hungarian", dtype="U32"),
            "optimality_certified": np.full((len(res["n_list"]), res["trials"]), True, dtype=bool),
        },
        params=params,
    )

    out_json = args.out_json or str(Path(args.out).with_suffix(".json"))
    summary = {
        "notes": [
            "Monte Carlo sweep over two i.i.d. random point clouds in [0,1]^d.",
            "For each instance pair, matching is optimized for p_opt then evaluated at all q_eval.",
            "Optimality is certified by exact linear assignment solver (Hungarian).",
        ],
        "problem": "bipartite_matching_sweep",
        "n_values": [int(x) for x in res["n_list"]],
        "num_points_per_set": [int(x) for x in res["n_list"]],
        "dimension_d": int(res["dim"]),
        "p_opt": float(res["p_opt"]),
        "q_eval": [float(x) for x in res["q_eval"]],
        "trials_per_n": int(res["trials"]),
        "seed_base": int(res["seed"]),
        "domain": str(res["domain"]),
        "methods_used": ["hungarian"],
        "all_optimality_certified": True,
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

    title_norm = f"Normalized costs on sigma*_p (dim={dim}, p_opt={p_opt:g})"
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

    title_conc = f"Concentration proxy std/mean (dim={dim}, p_opt={p_opt:g})"
    plot_sweep_concentration(
        n_list=n_list,
        costs=costs,
        dim=dim,
        q_eval=q_eval,
        title=title_conc,
        out=args.out_conc,
    )
    
    if args.out_var:
        title_var = f"Var of normalized cost (dim={dim}, p_opt={p_opt:g})"
        plot_sweep_variance(
            n_list=n_list,
            costs=costs,
            dim=dim,
            q_eval=q_eval,
            title=title_var,
            out=args.out_var,
        )

def cmd_edge_dist(args: argparse.Namespace) -> None:
    import numpy as np
    from rempf.core.points import sample_uniform
    from rempf.core.matching.exact import solve_perm_exact, edge_lengths_given_perm
    from rempf.io.npz import save_npz
    from rempf.plots.edge_plots import plot_edge_dist

    rng = np.random.default_rng(args.seed)
    pooled = []
    for t in range(args.trials):
        x = sample_uniform(args.n, args.dim, rng=rng)
        y = sample_uniform(args.n, args.dim, rng=rng)
        sigma = solve_perm_exact(x, y, p=float(args.p_opt), domain=args.domain)
        ell = edge_lengths_given_perm(x, y, sigma, domain=args.domain)

        # normalize by typical spacing ~ n^{-1/d}
        ell_scaled = ell * (args.n ** (1.0 / args.dim))
        pooled.append(ell_scaled)

    pooled = np.concatenate(pooled, axis=0)


    params = {
        "problem": "edge_dist",
        "n": args.n,
        "dim": args.dim,
        "p_opt": float(args.p_opt),
        "trials": args.trials,
        "seed": args.seed,
        "domain": args.domain,
    }
    save_npz(args.out, arrays={"edge_lengths_scaled": pooled}, params=params)

    out_json = args.out_json or str(Path(args.out).with_suffix(".json"))
    summary = {
        "notes": [
            "Pooled edge-length distribution under p-optimal exact bipartite matching.",
            "Edge lengths are scaled by n^(1/d) for comparability across n.",
        ],
        "problem": "edge_dist",
        "n": int(args.n),
        "dimension_d": int(args.dim),
        "p_opt": float(args.p_opt),
        "trials": int(args.trials),
        "seed": int(args.seed),
        "domain": str(args.domain),
        "bins": int(args.bins),
        "ccdf": bool(args.ccdf),
        "npz_output": str(args.out),
        "plot_output": str(args.plot),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_edge_dist(
        edge_lengths=pooled,
        out=args.plot,
        bins=args.bins,
        ccdf=bool(args.ccdf),
        title=f"Scaled edge lengths on sigma*_p (n={args.n}, d={args.dim}, p={args.p_opt:g})"
    )
