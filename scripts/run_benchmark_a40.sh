#!/usr/bin/env bash
# Re-run CUDA vs static CPU/NPU benchmark (after offline install).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/libs/pointops_torch/bench_compare_static.py" ]]; then
  ROOT="${SCRIPT_DIR}"
else
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

DEVICE_ID="${DEVICE_ID:-0}"
OUT_DIR="${OUT_DIR:-${ROOT}/artifacts}"
mkdir -p "${OUT_DIR}"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd "${ROOT}/libs/pointops_torch"

python3 bench_compare_static.py \
  --device-id "${DEVICE_ID}" \
  --markdown-out "${OUT_DIR}/PERF_CUDA_COMPARE.md" \
  | tee "${OUT_DIR}/PERF_CUDA_COMPARE.log"

echo "Report: ${OUT_DIR}/PERF_CUDA_COMPARE.md"
