#!/usr/bin/env bash
# Offline install Pointcept CUDA libs + run benchmark prep (no network).
#
# Expects a bundle layout created by package_for_a40_offline.sh:
#   <bundle>/
#     README_A40_OFFLINE.md
#     install.sh                 -> symlink or copy of this script
#     libs/pointops/
#     libs/pointops2/
#     libs/pointops_torch/
#     wheels/                    optional (ninja)
#     VERSION.txt
#
# Prerequisites on A40 host (must exist before running):
#   - Python 3.8+ with PyTorch (CUDA build matching nvcc)
#   - CUDA toolkit, NVIDIA driver, gcc/g++
#   - ninja (or bundled wheel under wheels/)
#
# Usage (on offline A40 machine):
#   tar xzf pointcept_a40_offline_*.tar.gz
#   cd pointcept_a40_offline_*
#   bash install.sh
#   bash run_benchmark.sh

set -euo pipefail

TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 4)}"
INSTALL_POINTGROUP="${INSTALL_POINTGROUP:-0}"
SKIP_BENCHMARK="${SKIP_BENCHMARK:-0}"
DEVICE_ID="${DEVICE_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Bundle root: parent of scripts/ OR current dir if install.sh lives at root
if [[ -f "${SCRIPT_DIR}/libs/pointops/setup.py" ]]; then
  BUNDLE_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../libs/pointops/setup.py" ]]; then
  BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  BUNDLE_ROOT="${SCRIPT_DIR}"
fi

LIBS="${BUNDLE_ROOT}/libs"
WHEELS="${BUNDLE_ROOT}/wheels"
ARTIFACTS="${BUNDLE_ROOT}/artifacts"

log() { printf '[offline-install] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 1; }

preflight() {
  log "Bundle root: ${BUNDLE_ROOT}"
  [[ -f "${LIBS}/pointops/setup.py" ]] || die "missing ${LIBS}/pointops/setup.py"
  [[ -f "${LIBS}/pointops2/setup.py" ]] || die "missing ${LIBS}/pointops2/setup.py"
  [[ -f "${LIBS}/pointops_torch/bench_compare_static.py" ]] \
    || die "missing bench_compare_static.py"

  command -v python3 >/dev/null || die "python3 not found"
  command -v nvcc >/dev/null || die "nvcc not found"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"

  export TORCH_DEVICE_BACKEND_AUTOLOAD=0

  python3 - <<'PY' || die "PyTorch CUDA not available"
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False")
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

  export MAX_JOBS TORCH_CUDA_ARCH_LIST
  log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS}"
}

install_ninja_offline() {
  if python3 -c "import ninja" 2>/dev/null; then
    log "ninja already available"
    return
  fi
  shopt -s nullglob
  local whls=("${WHEELS}"/ninja-*.whl)
  shopt -u nullglob
  if [[ ${#whls[@]} -eq 0 ]]; then
    die "ninja not installed and no wheel in ${WHEELS}/ — install ninja manually"
  fi
  log "Installing ninja from ${whls[0]}"
  python3 -m pip install --no-index --find-links="${WHEELS}" "${whls[0]}"
}

install_extension() {
  local name="$1"
  shift
  log "Building ${name} (offline source) ..."
  cd "${LIBS}/${name}"
  python3 setup.py install "$@"
  cd "${BUNDLE_ROOT}"
}

record_provenance() {
  mkdir -p "${ARTIFACTS}"
  {
    [[ -f "${BUNDLE_ROOT}/VERSION.txt" ]] && cat "${BUNDLE_ROOT}/VERSION.txt"
    echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
    python3 - <<'PY'
import torch
print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
PY
    date -u + "installed_utc=%Y-%m-%dT%H:%M:%SZ"
  } >> "${ARTIFACTS}/offline_install.txt"
  log "Provenance: ${ARTIFACTS}/offline_install.txt"
}

verify() {
  log "Verify imports + CUDA smoke test"
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  python3 - <<'PY'
import torch
import pointops
from pointops import knn_query, farthest_point_sampling, grouping, interpolation
import pointops2
from pointops2 import furthestsampling

xyz = torch.randn(128, 3, device="cuda")
offset = torch.tensor([64, 128], dtype=torch.int32, device="cuda")
new_offset = torch.tensor([8, 16], dtype=torch.int32, device="cuda")
idx = furthestsampling(xyz, offset, new_offset)
assert idx.is_cuda
print("OK: pointops + pointops2 CUDA extensions")
PY
}

run_benchmark() {
  [[ "${SKIP_BENCHMARK}" == "1" ]] && return
  log "Running CUDA vs static CPU/NPU benchmark ..."
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  cd "${LIBS}/pointops_torch"
  python3 bench_compare_static.py \
    --device-id "${DEVICE_ID}" \
    --markdown-out "${ARTIFACTS}/PERF_CUDA_COMPARE.md" \
    | tee "${ARTIFACTS}/PERF_CUDA_COMPARE.log"
  log "Report: ${ARTIFACTS}/PERF_CUDA_COMPARE.md"
}

main() {
  preflight
  install_ninja_offline
  install_extension pointops
  install_extension pointops2

  if [[ "${INSTALL_POINTGROUP}" == "1" ]]; then
    [[ -f "${LIBS}/pointgroup_ops/setup.py" ]] || die "pointgroup_ops source missing"
    pg_args=()
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -d "${CONDA_PREFIX}/include" ]]; then
      pg_args+=(--include_dirs="${CONDA_PREFIX}/include")
    fi
    install_extension pointgroup_ops "${pg_args[@]:-}"
  fi

  record_provenance
  verify
  run_benchmark
  log "Done."
}

main "$@"
