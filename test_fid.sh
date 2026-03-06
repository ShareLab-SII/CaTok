#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-./output/catok_b_256}"
STEP="${2:-250000}"
CFG_VALUE="${3:-2.0}"
NUM_SLOTS="${4:-256}"
NUM_STEPS="${5:-25}"
MASTER_PORT="${MASTER_PORT:-29506}"

torchrun --nproc-per-node=8 --master_port="${MASTER_PORT}" test_net.py \
  --model "${MODEL_DIR}" \
  --step "${STEP}" \
  --cfg_value "${CFG_VALUE}" \
  --test_num_slots "${NUM_SLOTS}" \
  --test_num_steps "${NUM_STEPS}"
