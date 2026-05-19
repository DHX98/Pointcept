#!/usr/bin/env bash
# Install Pointcept CUDA libs (pointops / pointops2 / pointgroup_ops) on x86 + A40.
#
# Source tree (official):
#   https://github.com/Pointcept/Pointcept/tree/df36980119f4636beb2d02d04ef3b2fec0fddfba/libs
#
# Target GPU: NVIDIA A40 (compute capability 8.6).
#
# Prerequisites (install separately before running this script):
#   - Python 3.8+ with PyTorch built for the same CUDA as nvcc
#   - CUDA toolkit + driver (README suggests CUDA 11.3+; match your torch build)
#   - gcc/g++, ninja
#
# Usage:
#   # Use existing checkout (default: directory containing this script's parent repo)
#   bash scripts/install_pointcept_libs_x86_a40.sh
#
#   # Clone fresh at pinned commit into $WORKDIR
#   POINTCEPT_ROOT= WORKDIR=/tmp/pointcept bash scripts/install_pointcept_libs_x86_a40.sh
#
#   # Skip optional pointgroup_ops (needs google-sparsehash headers)
#   INSTALL_POINTGROUP=0 bash scripts/install_pointcept_libs_x86_a40.sh
#
# Environment:
#   POINTCEPT_COMMIT   default df36980119f4636beb2d02d04ef3b2fec0fddfba
#   POINTCEPT_ROOT     repo root; auto-detected if already at pinned commit
#   WORKDIR            clone destination when POINTCEPT_ROOT is empty
#   TORCH_CUDA_ARCH_LIST  default 8.6 (A40)
#   USE_GHFAST         default 1 — prefix GitHub clone URL with https://ghfast.top/
#   PIP_INDEX_URL      optional PyPI mirror (e.g. Tsinghua)
#   INSTALL_POINTGROUP default 1
#   MAX_JOBS           parallel compile jobs (default nproc)
#   ARTIFACTS_DIR      provenance output (default $POINTCEPT_ROOT/artifacts)

set -euo pipefail

POINTCEPT_COMMIT="${POINTCEPT_COMMIT:-df36980119f4636beb2d02d04ef3b2fec0fddfba}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
USE_GHFAST="${USE_GHFAST:-1}"
INSTALL_POINTGROUP="${INSTALL_POINTGROUP:-1}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 4)}"
WORKDIR="${WORKDIR:-/tmp/pointcept_install}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
POINTCEPT_ROOT="${POINTCEPT_ROOT:-${DEFAULT_ROOT}}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${POINTCEPT_ROOT}/artifacts}"

GITHUB_REPO="https://github.com/Pointcept/Pointcept.git"
if [[ "${USE_GHFAST}" == "1" ]]; then
  CLONE_URL="https://ghfast.top/${GITHUB_REPO}"
else
  CLONE_URL="${GITHUB_REPO}"
fi

log() { printf '[install_pointcept_libs] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 1; }

pip_cmd() {
  if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    python3 -m pip install "$@" -i "${PIP_INDEX_URL}" --trusted-host "$(echo "${PIP_INDEX_URL}" | sed -E 's|https?://([^/]+).*|\1|')"
  else
    python3 -m pip install "$@"
  fi
}

ensure_repo() {
  if [[ -d "${POINTCEPT_ROOT}/.git" ]]; then
    log "Using existing git repo: ${POINTCEPT_ROOT}"
    cd "${POINTCEPT_ROOT}"
    current="$(git rev-parse HEAD 2>/dev/null || true)"
    if [[ "${current}" != "${POINTCEPT_COMMIT}" ]]; then
      log "Checking out ${POINTCEPT_COMMIT} (was ${current:-unknown})"
      git fetch --depth 1 origin "${POINTCEPT_COMMIT}" || git fetch origin
      git checkout "${POINTCEPT_COMMIT}"
    fi
  elif [[ -f "${POINTCEPT_ROOT}/libs/pointops/setup.py" ]]; then
    log "Using existing tree (no .git): ${POINTCEPT_ROOT}"
    cd "${POINTCEPT_ROOT}"
  else
    log "Cloning Pointcept into ${WORKDIR}"
    mkdir -p "${WORKDIR}"
    if [[ -d "${WORKDIR}/Pointcept" ]]; then
      die "Target exists: ${WORKDIR}/Pointcept — remove it or set POINTCEPT_ROOT"
    fi
    git clone --filter=blob:none --depth 1 --branch "${POINTCEPT_COMMIT}" "${CLONE_URL}" "${WORKDIR}/Pointcept" 2>/dev/null \
      || {
        log "Shallow branch clone failed; full clone + fetch"
        git clone --filter=blob:none "${CLONE_URL}" "${WORKDIR}/Pointcept"
        cd "${WORKDIR}/Pointcept"
        git fetch --depth 1 origin "${POINTCEPT_COMMIT}"
        git checkout "${POINTCEPT_COMMIT}"
        cd - >/dev/null
      }
    POINTCEPT_ROOT="${WORKDIR}/Pointcept"
    cd "${POINTCEPT_ROOT}"
  fi

  [[ -f "${POINTCEPT_ROOT}/libs/pointops/setup.py" ]] \
    || die "libs/pointops/setup.py not found under ${POINTCEPT_ROOT}"
}

preflight() {
  log "Preflight checks (x86 + CUDA + PyTorch)"

  arch="$(uname -m)"
  [[ "${arch}" == "x86_64" ]] || log "WARN: expected x86_64, got ${arch}"

  command -v python3 >/dev/null || die "python3 not found"
  command -v nvcc >/dev/null || die "nvcc not found — install CUDA toolkit"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found — install NVIDIA driver"

  python3 - <<'PY' || die "PyTorch CUDA not available"
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {name}, capability {cap[0]}.{cap[1]}, torch {torch.__version__}, cuda {torch.version.cuda}")
PY

  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)"
  log "nvidia-smi GPU: ${gpu_name}"
  if [[ "${gpu_name}" != *"A40"* ]]; then
    log "WARN: GPU is not A40; TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (override if needed)"
  fi

  pip_cmd -q ninja
  export MAX_JOBS TORCH_CUDA_ARCH_LIST
  log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, MAX_JOBS=${MAX_JOBS}"
}

install_extension() {
  local name="$1"
  local dir="${POINTCEPT_ROOT}/libs/${name}"
  [[ -d "${dir}" ]] || die "missing lib dir: ${dir}"
  log "Building ${name} ..."
  cd "${dir}"
  python3 setup.py install "$@"
  cd "${POINTCEPT_ROOT}"
}

record_provenance() {
  mkdir -p "${ARTIFACTS_DIR}"
  {
    echo "pointcept_commit=${POINTCEPT_COMMIT}"
    if [[ -d "${POINTCEPT_ROOT}/.git" ]]; then
      echo "git_head=$(git -C "${POINTCEPT_ROOT}" rev-parse HEAD)"
    fi
    echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)"
    python3 - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
PY
    date -u + "installed_utc=%Y-%m-%dT%H:%M:%SZ"
  } >> "${ARTIFACTS_DIR}/pointcept_libs_install.txt"

  if [[ -d "${POINTCEPT_ROOT}/.git" ]]; then
    git -C "${POINTCEPT_ROOT}" rev-parse HEAD >> "${ARTIFACTS_DIR}/commits.txt" 2>/dev/null || true
  fi
  log "Provenance written to ${ARTIFACTS_DIR}/pointcept_libs_install.txt"
}

verify_imports() {
  log "Verifying imports ..."
  python3 - <<'PY'
import torch
import pointops
from pointops import knn_query, farthest_point_sampling, grouping, interpolation
import pointops2
from pointops2 import furthestsampling

assert torch.cuda.is_available()
print("pointops:", knn_query, farthest_point_sampling, grouping, interpolation)
print("pointops2:", furthestsampling)

# Minimal CUDA smoke test
xyz = torch.randn(128, 3, device="cuda")
offset = torch.tensor([64, 128], dtype=torch.int32, device="cuda")
new_offset = torch.tensor([8, 16], dtype=torch.int32, device="cuda")
idx = furthestsampling(xyz, offset, new_offset)
assert idx.is_cuda and idx.numel() == 16
print("smoke test OK:", idx.shape, idx.dtype)
PY

  if [[ "${INSTALL_POINTGROUP}" == "1" ]]; then
    python3 - <<'PY'
import pointgroup_ops
from pointgroup_ops import bfs_cluster
print("pointgroup_ops:", bfs_cluster)
PY
  fi
}

main() {
  ensure_repo
  preflight

  install_extension pointops
  install_extension pointops2

  if [[ "${INSTALL_POINTGROUP}" == "1" ]]; then
    pg_args=()
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -d "${CONDA_PREFIX}/include" ]]; then
      if [[ -f "${CONDA_PREFIX}/include/sparsehash/dense_hash_map" ]] \
        || [[ -f "${CONDA_PREFIX}/include/google/dense_hash_map" ]]; then
        pg_args+=(--include_dirs="${CONDA_PREFIX}/include")
        log "pointgroup_ops: using include_dirs=${CONDA_PREFIX}/include"
      else
        log "WARN: google-sparsehash headers not found under CONDA_PREFIX"
        log "  Install: conda install -c bioconda google-sparsehash"
        log "  Or: INSTALL_POINTGROUP=0 to skip"
        die "pointgroup_ops requires sparsehash headers"
      fi
    else
      log "WARN: CONDA_PREFIX not set; trying default system include paths"
    fi
    install_extension pointgroup_ops "${pg_args[@]:-}"
  else
    log "Skipping pointgroup_ops (INSTALL_POINTGROUP=0)"
  fi

  record_provenance
  verify_imports

  log "Done. Run CUDA vs CPU/NPU benchmark (static NPU data):"
  log "  cd ${POINTCEPT_ROOT}/libs/pointops_torch"
  log "  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 bench_compare_static.py --device-id 0"
}

main "$@"
