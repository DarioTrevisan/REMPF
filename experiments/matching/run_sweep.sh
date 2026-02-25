#!/usr/bin/env bash
set -euo pipefail

# Local-friendly defaults (override via env vars).
DIM="${DIM:-3}"
P_OPT="${P_OPT:-1}"
Q_EVAL="${Q_EVAL:-1,2,3}"
N_LIST="${N_LIST:-32,64,128}"
TRIALS="${TRIALS:-20}"
SEED="${SEED:-0}"
OUT="${OUT:-results/match_sweep_d${DIM}_p${P_OPT}_q${Q_EVAL//,/}.npz}"

mkdir -p "$(dirname "$OUT")"

rempf match sweep \
  --dim "$DIM" --p-opt "$P_OPT" --q-eval "$Q_EVAL" \
  --n-list "$N_LIST" --trials "$TRIALS" --seed "$SEED" \
  --progress \
  --out "$OUT"

echo "Wrote: $OUT"
