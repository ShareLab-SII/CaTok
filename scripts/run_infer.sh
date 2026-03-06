#!/usr/bin/env bash
set -euo pipefail

# One-click wrapper for scripts/infer_recon.py
#
# Usage:
#   bash scripts/run_infer.sh <checkpoint|auto> <image_path> [cfg] [num_tokens] [start_token] [sample_steps] [output_dir] [config_path]
#
# Example:
#   bash scripts/run_infer.sh \
#     /path/to/models/step50040/custom_checkpoint_1.pkl \
#     /path/to/your/input_image.webp \
#     2.0 256 0 25 ./infer_outputs
#
# Interpreter:
#   Set PYTHON_BIN to force a specific CPython interpreter, e.g.
#   PYTHON_BIN=/usr/bin/python bash scripts/run_infer.sh ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/output/catok_b_256}"
CONFIG_PATH="${PROJECT_ROOT}/configs/catok_b_256.yaml"
DEFAULT_IMAGE="/path/to/your/input_image.webp"

CHECKPOINT="${1:-auto}"
IMAGE_PATH="${2:-${DEFAULT_IMAGE}}"
CFG="${3:-2.0}"
NUM_TOKENS="${4:-256}"
START_TOKEN="${5:-0}"
SAMPLE_STEPS="${6:-25}"
OUTPUT_DIR="${7:-${PROJECT_ROOT}/infer_outputs}"
CONFIG_OVERRIDE="${8:-}"

if [[ -n "${CONFIG_OVERRIDE}" ]]; then
  CONFIG_PATH="${CONFIG_OVERRIDE}"
fi

PY_EXE="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "[INFO] Using Python: ${PY_EXE}"

if ! "${PYTHON_BIN}" -c 'import omegaconf' >/dev/null 2>&1; then
  cat <<EOF
[ERROR] omegaconf is not installed in this interpreter:
        ${PY_EXE}
        Please install with:
        ${PY_EXE} -m pip install omegaconf
EOF
  exit 3
fi

CKPT_ARGS=()
if [[ "${CHECKPOINT}" != "auto" ]]; then
  CKPT_ARGS=(--checkpoint "${CHECKPOINT}")
fi

if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sys
text = sys.version.lower()
raise SystemExit(0 if ("graalpy" in text or "graalvm" in text) else 1)
PY
then
  cat <<EOF
[ERROR] PYTHON_BIN points to GraalPy, which is incompatible with this torch/numpy inference script.
        Current: ${PYTHON_BIN}
        Please use CPython, for example:
        PYTHON_BIN=/usr/bin/python bash ${PROJECT_ROOT}/scripts/run_infer.sh ...
EOF
  exit 2
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/infer_recon.py" \
  --model-dir "${MODEL_DIR}" \
  --config "${CONFIG_PATH}" \
  --image "${IMAGE_PATH}" \
  --cfg "${CFG}" \
  --num-tokens "${NUM_TOKENS}" \
  --start-token "${START_TOKEN}" \
  --sample-steps "${SAMPLE_STEPS}" \
  --output-dir "${OUTPUT_DIR}" \
  "${CKPT_ARGS[@]}"
