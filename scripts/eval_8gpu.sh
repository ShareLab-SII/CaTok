#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-./output/catok_b_256}
STEP=${2:-250000}
CFG=${3:-configs/catok_b_256.yaml}
SLOTS=${4:-256}
SAMPLE_STEPS=${5:-25}

torchrun --nproc-per-node=8 test_net.py \
  --model "${MODEL}" \
  --step "${STEP}" \
  --cfg "${CFG}" \
  --cfg_value 1.0 \
  --test_num_slots "${SLOTS}" \
  --test_num_steps "${SAMPLE_STEPS}"
