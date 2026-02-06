# REMPF — Random Euclidean Matching Problem & Friends

Numerical experiments for Random Euclidean (bipartite) Matching and related problems
(e.g. TSP), with an emphasis on **reproducibility**, **validation**, and
cross-checking **exact solvers vs heuristics**.

## Goals

- Reproducible Monte Carlo experiments (fixed seeds, saved configs + metadata).
- Multiple solvers per problem:
  - exact / robust reference solvers (when feasible)
  - fast heuristics for larger n
  - cross-validation between approaches
- Standardized output format (`.npz`) + plotting scripts.

## Installation (dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```


## Quick start
Bipartite matching (smoke test)

```bash
rempf match run \
  --n 200 --dim 2 --trials 5 --seed 0 \
  --solver exact \
  --out results/match_exact_smoke.npz
```

Plot a saved run

```bash
rempf match plot results/match_exact_smoke.npz --out results/match_exact_smoke.pdf
```

## Output format

All commands save a .npz that contains:

- costs: array of per-trial costs
- params: JSON-serializable dict of CLI parameters
- meta: git commit hash, timestamp, platform, python version, etc.

## Project structure

- src/rempf/core/: algorithms and problem definitions
- src/rempf/cli/: command-line entrypoints
- src/rempf/plots/: plotting utilities
- tests/: correctness/invariants tests
- experiments/: scripts used to generate figures


## Citation

If you use this code in academic work, please cite (TBD).