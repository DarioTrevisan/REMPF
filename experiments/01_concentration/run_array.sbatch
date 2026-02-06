#!/bin/bash
#SBATCH --job-name=rempf                   # Nome del job
#SBATCH --mail-type=END,FAIL                    # Eventi E-mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dario.trevisan@unipi.it           # Dove inviare la mail con gli esiti
#SBATCH --nodes=1                               # Richiediamo un singolo nodo
#SBATCH --ntasks=1                              # Che deve eseguire una sola task
#SBATCH --cpus-per-task=8                  # Utilizzando X CPUs
#SBATCH --time=24:00:00                         # Per cui richiediamo 24 ore
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=1-5%2


set -euo pipefail

echo "Dario"

# ---------- config ----------
JOBS_FILE="experiments/01_concentration/jobs.txt"
RESULTS_DIR="results/01_concentration"
LOGS_DIR="logs"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Stop if array index exceeds number of non-comment lines
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
LINE=$(awk 'NF && $1 !~ /^#/ {print NR ":" $0}' "$JOBS_FILE" | sed -n "${TASK_ID}p" || true)

if [[ -z "${LINE}" ]]; then
  echo "No job line for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}. Exiting."
  exit 0
fi

JOB=$(echo "$LINE" | cut -d: -f2-)
echo "Job line: $JOB"

# Parse fields
TYPE=$(echo "$JOB" | awk '{print $1}')
DIM=$(echo "$JOB"  | awk '{print $2}')
P_OPT=$(echo "$JOB"| awk '{print $3}')
Q_EVAL=$(echo "$JOB"| awk '{print $4}')
N_LIST=$(echo "$JOB"| awk '{print $5}')
TRIALS=$(echo "$JOB"| awk '{print $6}')
SEED=$(echo "$JOB"| awk '{print $7}')

echo "Parsed: type=$TYPE dim=$DIM p_opt=$P_OPT q_eval=$Q_EVAL n_list=$N_LIST trials=$TRIALS seed=$SEED"

# ---------- activate env ----------
# Option A: use venv stored in repo
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  # Option B: module+conda (edit to your cluster)
  module load anaconda3 || true
  # conda activate <ENVNAME>
fi

# Ensure rempf is available
which rempf
rempf --help >/dev/null

# ---------- run ----------
STAMP="${TYPE}_d${DIM}_p${P_OPT}_seed${SEED}_task${TASK_ID}"


if [[ "$TYPE" == "sweep" ]]; then
  OUT_NPZ="${RESULTS_DIR}/${STAMP}.npz"
  OUT_NORM="${RESULTS_DIR}/${STAMP}_norm.pdf"
  OUT_CONC="${RESULTS_DIR}/${STAMP}_conc.pdf"
  OUT_VAR="${RESULTS_DIR}/${STAMP}_var.pdf"

  rempf match sweep \
    --dim "$DIM" --p-opt "$P_OPT" \
    --q-eval "$Q_EVAL" \
    --n-list "$N_LIST" \
    --trials "$TRIALS" --seed "$SEED" \
    --out "$OUT_NPZ"

  rempf match plot-sweep "$OUT_NPZ" \
    --out-norm "$OUT_NORM" \
    --out-conc "$OUT_CONC" \
    --out-var  "$OUT_VAR"

elif [[ "$TYPE" == "edge" ]]; then
  # N_LIST is a single integer here
  N="$N_LIST"
  OUT_NPZ="${RESULTS_DIR}/${STAMP}_n${N}.npz"
  OUT_CCDF="${RESULTS_DIR}/${STAMP}_n${N}_ccdf.pdf"

  rempf match edge-dist \
    --dim "$DIM" --p-opt "$P_OPT" --n "$N" \
    --trials "$TRIALS" --seed "$SEED" \
    --out "$OUT_NPZ" \
    --plot "$OUT_CCDF" \
    --ccdf

else
  echo "Unknown TYPE='$TYPE'. Expected 'sweep' or 'edge'."
  exit 2
fi

echo "Done: $STAMP"
