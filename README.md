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

The matching ratio-check pipeline can also be run from an existing raw table:

```bash
python analyze_matching_ratios.py \
  --input results/matching_ratio_checks/raw_matching_ratios.csv \
  --outdir figures_ratio_checks \
  --paired true \
  --bootstrap 5000 \
  --alpha 0.05 \
  --target-a 1.80788
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

Use `--domain torus` for periodic flat-torus matching. Without this option,
the matching CLI uses the square/cube domain.

## Matching p=2 versus p=4 ratio checks

The script `analyze_matching_ratios.py` implements the diagnostics in
`SPECS_matching_ratio_checks.md`. It compares the self-normalized p=4 ratio
with the cross-normalized ratio relevant to the formal flow prediction:

- `R22 = E[M4^(2)] / E[M2^(2)]^2`
- `R44_self = E[M4^(4)] / E[M2^(4)]^2`
- `R44_cross = E[M4^(4)] / E[M2^(2)]^2`
- `Q2 = E[M2^(4)] / E[M2^(2)]`
- `I4 = 1 - E[M4^(4)] / E[M4^(2)]`

Input must be CSV, Parquet, or Feather with at least:

```text
n, seed, p_opt, M2, M4
```

Optional columns such as `status`, `runtime_sec`, and `solver_gap` are allowed.
The analyzer defaults to paired mode, retaining only seeds available for both
`p_opt=2` and `p_opt=4`.

Outputs include:

```text
summary_by_n.csv
summary_by_n.parquet
bootstrap_samples.parquet
ratio_curves.pdf
ratio_curves.png
cross_vs_self.pdf
quadratic_inflation.pdf
quartic_improvement.pdf
finite_size_fits.pdf
finite_size_fits.csv
report.md
raw_qc.md
```

The identity

```text
R44_cross = R44_self * Q2^2
```

is checked numerically and reported.

## Remote Slurm run to n=8192

`rates.slurm` runs the full periodic matching ratio experiment on a remote
Slurm cluster:

1. runs p=2 exact matching sweeps;
2. runs p=4 exact matching sweeps on the same seed schedule;
3. converts both `.npz` files into the raw analyzer table;
4. runs `analyze_matching_ratios.py`;
5. writes the final report and plots.

Default settings:

```text
DIM=2
DOMAIN=torus
Q_EVAL=2,4
N_LIST=16,32,64,128,256,512,1024,2048,4096,8192
TRIALS=200
BOOTSTRAP=5000
TARGET_A=1.80788
```

Submit from the repository root:

```bash
sbatch rates.slurm
```

Useful overrides:

```bash
TRIALS=100 sbatch rates.slurm
```

```bash
TRIALS=200 N_LIST=16,32,64,128,256,512,1024,2048,4096,8192 sbatch rates.slurm
```

If the remote environment needs dependencies installed into the active Python
environment:

```bash
INSTALL_DEPS=1 sbatch rates.slurm
```

If the conda environment path differs from `$SCRATCH/REMPF`:

```bash
CONDA_ENV=/path/to/env sbatch rates.slurm
```

Outputs are written by default to:

```text
results/matching_ratio_checks_slurm/
figures_ratio_checks_slurm/
logs/
```

These output directories are ignored by git, except for `logs/.gitkeep` so that
Slurm has a valid log directory after cloning or pushing the repository.

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

`analyze_matching_ratios.py` writes `.csv`, optional `.parquet`, `.pdf`, `.png`,
and Markdown report files for the p=2 versus p=4 diagnostics.

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
- `analyze_matching_ratios.py`: table-driven matching ratio diagnostics
- `rates.slurm`: Slurm batch workflow for periodic p=2/p=4 matching checks
- `tests/`: tests
