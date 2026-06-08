#!/usr/bin/env bash
# Convenience wrapper: run a NIAH context-length sweep over a checkpoint and chart it.
# run_niah.py already handles multiple context lengths in one process.
#
#   ./scripts/run_sweep.sh out-niah-demo/ckpt.pt
#   CONTEXTS="64 128 256" DEVICE=cpu ./scripts/run_sweep.sh out-niah-demo/ckpt.pt
set -euo pipefail

CKPT="${1:?usage: run_sweep.sh <ckpt.pt>}"
CONTEXTS="${CONTEXTS:-64 128 256}"
SAMPLES="${SAMPLES:-5}"
CHOICES="${CHOICES:-4}"
SEED="${SEED:-42}"
TOKENIZER="${TOKENIZER:-gpt2}"
OUTDIR="${OUTDIR:-results}"

DEVICE_ARG=()
[ -n "${DEVICE:-}" ] && DEVICE_ARG=(--device "$DEVICE")

python scripts/run_niah.py \
  --ckpt "$CKPT" \
  --context-length $CONTEXTS \
  --samples "$SAMPLES" \
  --num-needles "$CHOICES" \
  --seed "$SEED" \
  --tokenizer "$TOKENIZER" \
  --csv-out "$OUTDIR/niah_sweep.csv" \
  --chart "$OUTDIR" \
  "${DEVICE_ARG[@]}"
