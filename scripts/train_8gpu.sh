#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/catok_b_256.yaml}

torchrun --nproc-per-node=8 train_net.py --cfg "${CFG}"
