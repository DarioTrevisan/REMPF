# REMPF — Random Euclidean Matching Problem & Friends

Numerical experiments for Random Euclidean bipartite matching and Euclidean TSP,
with emphasis on reproducibility and scaling comparisons across exponents.

## Installation (dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Quickstart (local scripts)

Two local-friendly bash scripts are included (no `sbatch`/Slurm):

```bash
./experiments/matching/run_sweep.sh
./experiments/tsp/run_sweep.sh
```

Each script exposes simple overrides via env vars (e.g. `N_LIST`, `TRIALS`, `OUT`):

```bash
N_LIST=64,128,256 TRIALS=50 ./experiments/matching/run_sweep.sh
DIM=5 Q_EVAL=4,5,6 ./experiments/tsp/run_sweep.sh
```

## CLI overview

```bash
rempf match {sweep,plot-sweep,edge-dist}
rempf tsp {sweep,plot-sweep}
```

## Matching sweep

Compute p-optimal matching, then evaluate q-costs on the same optimizer.

```bash
rempf match sweep \
  --dim 3 --p-opt 1 --q-eval 1,2,3 \
  --n-list 64,128,256,512 --trials 100 --seed 0 \
  --progress \
  --out results/match_sweep_d3_p1_q123.npz
```

Plot with error bars at sampled points:

```bash
rempf match plot-sweep results/match_sweep_d3_p1_q123.npz \
  --out-norm results/match_sweep_d3_p1_q123_norm.pdf \
  --out-conc results/match_sweep_d3_p1_q123_conc.pdf \
  --out-var results/match_sweep_d3_p1_q123_var.pdf \
  --error-bars se
```

## TSP sweep

Compute a p-optimal route (exact when feasible, heuristic fallback if enabled),
then evaluate q-costs on that same route.

```bash
rempf tsp sweep \
  --dim 4 --p-opt 4 --q-eval 4,5,6 \
  --n-list 64,128,256 \
  --trials 6 --seed 0 \
  --milp-max-n 0 \
  --heuristic-starts 32 \
  --allow-heuristic-fallback \
  --progress \
  --out results/tsp_sweep_d4_p4_q456.npz
```

```bash
rempf tsp plot-sweep results/tsp_sweep_d4_p4_q456.npz \
  --out-norm results/tsp_sweep_d4_p4_q456_norm.pdf \
  --out-conc results/tsp_sweep_d4_p4_q456_conc.pdf \
  --out-var results/tsp_sweep_d4_p4_q456_var.pdf \
  --error-bars se
```

## Output files

`*sweep` commands write:

- `.npz` arrays (cost tensors, parameters, metadata)
- `.json` sidecar summary (n values, exponents, trials, solver settings, methods used)

By default the sidecar path is the same as `--out` with `.json` extension;
you can override it with `--out-json`.

`plot-sweep` writes `.pdf` figures:

- normalized scaling plot
- concentration plot (`std/mean`)
- optional variance plot

## Notes on TSP accuracy

- For small `n`, exact methods are used (Held-Karp / MILP when available).
- For larger `n`, enable `--allow-heuristic-fallback`.
- Increase `--heuristic-starts` to improve route quality (higher runtime).

## Project structure

- `src/rempf/core/`: algorithms and experiment runners
- `src/rempf/cli/`: command-line interfaces
- `src/rempf/plots/`: plotting utilities
- `src/rempf/io/`: save/load helpers for `.npz`
- `experiments/matching/`: local matching sweep scripts
- `experiments/tsp/`: local TSP sweep scripts
- `tests/`: tests
