#!/bin/bash
#SBATCH --job-name=match-rates                  # Nome del job
#SBATCH --mail-type=END,FAIL                    # Eventi E-mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dario.trevisan@unipi.it           # Dove inviare la mail con gli esiti
#SBATCH --nodes=1                               # Richiediamo un singolo nodo
#SBATCH --ntasks=1                              # Che deve eseguire una sola task
#SBATCH --cpus-per-task=2                       # Utilizzando X CPUs
#SBATCH --time=24:00:00                         # Per cui richiediamo 24 ore
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Run from the directory where sbatch was submitted, so relative output paths
# remain valid after pushing the project folder to a remote server.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Si caricano sul nodo tutti i moduli necessari ad eseguire il codice in oggetto.
# Override SKIP_CONDA=1 if the remote environment is already active.
if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load "${ANACONDA_MODULE:-anaconda3}"
  fi
  if command -v conda >/dev/null 2>&1; then
    if [[ -n "${CONDA_ENV:-}" ]]; then
      conda activate "$CONDA_ENV"
    elif [[ -n "${SCRATCH:-}" ]]; then
      conda activate "$SCRATCH/REMPF"
    else
      echo "CONDA_ENV and SCRATCH are unset; leaving current Python environment active."
    fi
  fi
fi

# Il numero riportato qui deve essere minore o uguale a quello in --cpus-per-task
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:-src}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-rempf-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

# Matching ratio experiment defaults. Override any of these at submission time:
#
#   TRIALS=100 N_LIST=16,32,64,128,256,512,1024,2048,4096,8192 sbatch rates.slurm
#
# Exact n=8192 runs are expensive. On the local workstation, 10 paired trials
# took about 15 minutes total; 200 trials should be multi-hour but plausible on
# a single well-provisioned node.
PYTHON_BIN="${PYTHON_BIN:-python}"
DIM="${DIM:-2}"
DOMAIN="${DOMAIN:-torus}"
Q_EVAL="${Q_EVAL:-2,4}"
N_LIST="${N_LIST:-16,32,64,128,256,512,1024,2048,4096,8192}"
TRIALS="${TRIALS:-200}"
SEED="${SEED:-0}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
ALPHA="${ALPHA:-0.05}"
TARGET_A="${TARGET_A:-1.80788}"
OUT_DIR="${OUT_DIR:-results/matching_ratio_checks_slurm}"
FIG_DIR="${FIG_DIR:-figures_ratio_checks_slurm}"
TAG="${TAG:-d${DIM}_${DOMAIN}_q${Q_EVAL//,/}_n${N_LIST//,/x}_t${TRIALS}_seed${SEED}}"

mkdir -p "$OUT_DIR" "$FIG_DIR"

P2_SWEEP="${OUT_DIR}/match_sweep_${TAG}_p2.npz"
P4_SWEEP="${OUT_DIR}/match_sweep_${TAG}_p4.npz"
RAW_CSV="${OUT_DIR}/raw_matching_ratios_${TAG}.csv"
export P2_SWEEP P4_SWEEP RAW_CSV

echo "==== JOB ENVIRONMENT ===="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" --version
echo "DIM=$DIM DOMAIN=$DOMAIN Q_EVAL=$Q_EVAL"
echo "N_LIST=$N_LIST TRIALS=$TRIALS SEED=$SEED"
echo "OUT_DIR=$OUT_DIR FIG_DIR=$FIG_DIR"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -e ".[dev]"
fi

echo "==== RUN p=2 MATCHING SWEEP ===="
"$PYTHON_BIN" -m rempf.cli.rempf match sweep \
  --dim "$DIM" \
  --domain "$DOMAIN" \
  --p-opt 2 \
  --q-eval "$Q_EVAL" \
  --n-list "$N_LIST" \
  --trials "$TRIALS" \
  --seed "$SEED" \
  --progress \
  --out "$P2_SWEEP"

echo "==== RUN p=4 MATCHING SWEEP ===="
"$PYTHON_BIN" -m rempf.cli.rempf match sweep \
  --dim "$DIM" \
  --domain "$DOMAIN" \
  --p-opt 4 \
  --q-eval "$Q_EVAL" \
  --n-list "$N_LIST" \
  --trials "$TRIALS" \
  --seed "$SEED" \
  --progress \
  --out "$P4_SWEEP"

echo "==== BUILD RAW RATIO TABLE ===="
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os

import pandas as pd

from rempf.io.npz import load_npz

p2_path = Path(os.environ["P2_SWEEP"])
p4_path = Path(os.environ["P4_SWEEP"])
out = Path(os.environ["RAW_CSV"])
out.parent.mkdir(parents=True, exist_ok=True)

rows = []
for path in (p2_path, p4_path):
    data = load_npz(path)
    q_eval = [float(q) for q in data["q_eval"]]
    try:
        q2_idx = q_eval.index(2.0)
        q4_idx = q_eval.index(4.0)
    except ValueError as exc:
        raise ValueError(f"{path} must contain q_eval=2,4; got {q_eval}") from exc

    p_opt = float(data["params"]["p_opt"])
    seed_base = int(data["params"]["seed"])
    costs = data["costs"]
    for k, n_value in enumerate(data["n_list"].astype(int)):
        # run_sweep uses seed + 1000*k for the kth n. Trial id gives a paired
        # sample label across p=2 and p=4 because both sweeps use the same seed.
        seed_n = seed_base + 1000 * k
        for trial in range(costs.shape[1]):
            rows.append(
                {
                    "n": int(n_value),
                    "seed": f"{seed_n}:{trial}",
                    "p_opt": p_opt,
                    "M2": float(costs[k, trial, q2_idx] / n_value),
                    "M4": float(costs[k, trial, q4_idx] / n_value),
                    "status": "ok",
                    "source_npz": str(path),
                    "cost_convention": "per_point_total_cost_divided_by_n",
                    "domain": str(data["params"]["domain"]),
                    "dim": int(data["params"]["dim"]),
                }
            )

raw = pd.DataFrame(rows)
raw.to_csv(out, index=False)
print(out)
print(raw.groupby(["n", "p_opt"]).size().unstack().to_string())
PY

echo "==== ANALYZE MATCHING RATIOS ===="
"$PYTHON_BIN" analyze_matching_ratios.py \
  --input "$RAW_CSV" \
  --outdir "$FIG_DIR" \
  --paired true \
  --bootstrap "$BOOTSTRAP" \
  --alpha "$ALPHA" \
  --target-a "$TARGET_A" \
  --rng-seed 12345

echo "==== OUTPUTS ===="
echo "p=2 sweep: $P2_SWEEP"
echo "p=4 sweep: $P4_SWEEP"
echo "raw csv: $RAW_CSV"
echo "summary: ${FIG_DIR}/summary_by_n.csv"
echo "report: ${FIG_DIR}/report.md"
echo "plot: ${FIG_DIR}/ratio_curves.png"

echo "==== SLURM ACCOUNTING ===="
if command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,CPUTime,MaxRSS,AllocCPUS,AllocGRES
fi
