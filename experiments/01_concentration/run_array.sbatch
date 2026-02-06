#!/bin/bash
#SBATCH --job-name=rempf
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dario.trevisan@unipi.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j_%a.out
#SBATCH --error=logs/%x-%j_%a.err
#SBATCH --array=1-5%2

# -------------------------
# strict mode (careful!)
# -------------------------
set -eo pipefail   # do NOT use -u yet

# -------------------------
# directories
# -------------------------
ROOT_DIR="$SLURM_SUBMIT_DIR"
JOBS_FILE="experiments/01_concentration/jobs.txt"
RESULTS_DIR="results/01_concentration"
LOGS_DIR="logs"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# -------------------------
# diagnostics
# -------------------------
echo "HOST: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-<unset>}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"

# Robust task id resolution
TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-1}}"
echo "Using TASK_ID=${TASK_ID}"

# -------------------------
# load conda
# -------------------------
module purge
module load anaconda3 || module load Anaconda3 || module load anaconda

# IMPORTANT: initialize conda for non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# >>> CHANGE THIS <<<
conda activate rempf
# or: conda activate ldp-nn / base / whatever you use

echo "Python: $(which python)"
echo "rempf: $(which rempf)"

set -euo pipefail   # now it's safe

# -------------------------
# select job line
# -------------------------
LINE=$(awk 'NF && $1 !~ /^#/' "$JOBS_FILE" | sed -n "${TASK_ID}p" || true)
if [[ -z "$LINE" ]]; then
  echo "No job line for TASK_ID=${TASK_ID}. Exiting."
  exit 0
fi

echo "Job line: $LINE"

# -------------------------
# parse fields
# -------------------------
TYPE=$(echo "$LINE" | awk '{print $1}')
DIM=$(echo "$LINE"  | awk '{print $2}')
P_OPT=$(echo "$LINE"| awk '{print $3}')
Q_EVAL=$(echo "$LINE"| awk '{print $4}')
N_LIST=$(echo "$LINE"| awk '{print $5}')
TRIALS=$(echo "$LINE"| awk '{print $6}')
SEED=$(echo "$LINE"| awk '{print $7}')

echo "Parsed:"
echo "  TYPE=$TYPE"
echo "  DIM=$DIM"
echo "  P_OPT=$P_OPT"
echo "  Q_EVAL=$Q_EVAL"
echo "  N_LIST=$N_LIST"
echo "  TRIALS=$TRIALS"
echo "  SEED=$SEED"

STAMP="${TYPE}_d${DIM}_p${P_OPT}_seed${SEED}_task${TASK_ID}"

# -------------------------
# run job
# -------------------------
set -x   # echo commands

if [[ "$TYPE" == "sweep" ]]; then
  OUT_NPZ="${RESULTS_DIR}/${STAMP}.npz"

  rempf match sweep \
    --dim "$DIM" --p-opt "$P_OPT" \
    --q-eval "$Q_EVAL" \
    --n-list "$N_LIST" \
    --trials "$TRIALS" --seed "$SEED" \
    --out "$OUT_NPZ"

  rempf match plot-sweep "$OUT_NPZ" \
    --out-norm "${RESULTS_DIR}/${STAMP}_norm.pdf" \
    --out-conc "${RESULTS_DIR}/${STAMP}_conc.pdf" \
    --out-var  "${RESULTS_DIR}/${STAMP}_var.pdf"

elif [[ "$TYPE" == "edge" ]]; then
  N="$N_LIST"

  rempf match edge-dist \
    --dim "$DIM" --p-opt "$P_OPT" --n "$N" \
    --trials "$TRIALS" --seed "$SEED" \
    --out "${RESULTS_DIR}/${STAMP}_n${N}.npz" \
    --plot "${RESULTS_DIR}/${STAMP}_n${N}_ccdf.pdf" \
    --ccdf
else
  echo "Unknown TYPE='$TYPE'"
  exit 2
fi

set +x
echo "Done: $STAMP"
