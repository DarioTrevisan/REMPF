#!/usr/bin/env bash
set -euo pipefail

# Local-friendly defaults (override via env vars).
DIM="${DIM:-4}"
P_OPT="${P_OPT:-4}"
Q_EVAL="${Q_EVAL:-4,5}"
N_LIST="${N_LIST:-32,64}"
TRIALS="${TRIALS:-6}"
SEED="${SEED:-0}"
MILP_MAX_N="${MILP_MAX_N:-0}"
HEURISTIC_STARTS="${HEURISTIC_STARTS:-8}"
OUT="${OUT:-results/tsp_sweep_d${DIM}_p${P_OPT}_q${Q_EVAL//,/}.npz}"

mkdir -p "$(dirname "$OUT")"

rempf tsp sweep \
  --dim "$DIM" --p-opt "$P_OPT" --q-eval "$Q_EVAL" \
  --n-list "$N_LIST" --trials "$TRIALS" --seed "$SEED" \
  --milp-max-n "$MILP_MAX_N" \
  --heuristic-starts "$HEURISTIC_STARTS" \
  --allow-heuristic-fallback \
  --progress \
  --out "$OUT"

echo "Wrote: $OUT"
