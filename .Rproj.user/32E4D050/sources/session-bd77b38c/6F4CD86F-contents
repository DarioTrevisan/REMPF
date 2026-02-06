from __future__ import annotations
import argparse
import numpy as np
from tqdm import trange

from rempf.core.points import sample_uniform
from rempf.core.matching.exact import bipartite_matching_cost_exact
from rempf.io.npz import save_npz
from rempf.plots.match_plots import plot_cost_hist

def add_match_subparser(sub):
    p = sub.add_parser("match", help="Bipartite Euclidean matching experiments")
    sub2 = p.add_subparsers(dest="match_cmd", required=True)

    run = sub2.add_parser("run", help="Run Monte Carlo trials")
    run.add_argument("--n", type=int, required=True)
    run.add_argument("--dim", type=int, default=2)
    run.add_argument("--p", type=float, default=1.0)
    run.add_argument("--trials", type=int, default=10)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--solver", choices=["exact"], default="exact")
    run.add_argument("--out", type=str, required=True)
    run.add_argument("--plot", action="store_true")
    run.set_defaults(func=cmd_run)

    plot = sub2.add_parser("plot", help="Plot from a saved .npz")
    plot.add_argument("npz", type=str)
    plot.add_argument("--out", type=str, required=True)
    plot.set_defaults(func=cmd_plot)

def cmd_run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    costs = np.empty(args.trials, dtype=float)

    for t in trange(args.trials, desc="trials"):
        x = sample_uniform(args.n, args.dim, rng=rng)
        y = sample_uniform(args.n, args.dim, rng=rng)
        if args.solver == "exact":
            costs[t] = bipartite_matching_cost_exact(x, y, p=args.p)
        else:
            raise RuntimeError("Unknown solver")

    params = {
        "problem": "bipartite_matching",
        "n": args.n,
        "dim": args.dim,
        "p": args.p,
        "trials": args.trials,
        "seed": args.seed,
        "solver": args.solver,
    }
    save_npz(args.out, arrays={"costs": costs}, params=params)

    if args.plot:
        out_pdf = args.out.replace(".npz", ".pdf")
        plot_cost_hist(costs, title=f"Matching costs (n={args.n}, dim={args.dim}, solver={args.solver})", out=out_pdf)

def cmd_plot(args: argparse.Namespace) -> None:
    from rempf.io.npz import load_npz
    d = load_npz(args.npz)
    costs = d["costs"]
    title = f"Costs | {d['params']}"
    plot_cost_hist(costs, title=title, out=args.out)
