#!/usr/bin/env bash
set -euo pipefail

# Reusable local runner for the observable
#   E[(1/n) sum |X_i-Y_sigma(i)|^4] / (E[(1/n) sum |X_i-Y_sigma(i)|^2])^2
# where sigma is the p=4 optimal matching.

DIM="${DIM:-2}"
DOMAIN="${DOMAIN:-torus}"
P_OPT="${P_OPT:-4}"
Q_EVAL="${Q_EVAL:-2,4}"
N_LIST="${N_LIST:-16,32,64,128,256}"
TRIALS="${TRIALS:-100}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-results/matching_p4_ratio}"
TAG="${TAG:-d${DIM}_${DOMAIN}_p${P_OPT}_q${Q_EVAL//,/}_n${N_LIST//,/x}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

SWEEP_OUT="${SWEEP_OUT:-${OUT_DIR}/match_sweep_${TAG}.npz}"
PLOT_OUT="${PLOT_OUT:-${OUT_DIR}/match_ratio_${TAG}.pdf}"
DATA_OUT="${DATA_OUT:-${OUT_DIR}/match_ratio_${TAG}.npz}"

mkdir -p "$OUT_DIR"
export PYTHONPATH="${PYTHONPATH:-src}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-rempf}"

"$PYTHON_BIN" -m rempf.cli.rempf match sweep \
  --dim "$DIM" --domain "$DOMAIN" --p-opt "$P_OPT" --q-eval "$Q_EVAL" \
  --n-list "$N_LIST" --trials "$TRIALS" --seed "$SEED" \
  --progress \
  --out "$SWEEP_OUT"

"$PYTHON_BIN" experiments/matching/p4_ratio_plot.py \
  "$SWEEP_OUT" \
  --out "$PLOT_OUT" \
  --out-data "$DATA_OUT"

echo "Wrote sweep: $SWEEP_OUT"
echo "Wrote ratio plot: $PLOT_OUT"
echo "Wrote ratio data: $DATA_OUT"
