#!/usr/bin/env bash
# Pack Pointcept libs + offline install scripts for air-gapped A40 (x86_64).
#
# Run on a machine WITH network (or existing checkout). Produces:
#   <output>/pointcept_a40_offline_<commit>_<date>.tar.gz
#
# Usage:
#   bash scripts/package_for_a40_offline.sh
#   OUTPUT_DIR=/data/scp bash scripts/package_for_a40_offline.sh
#
# Transfer the .tar.gz to A40, then:
#   tar xzf pointcept_a40_offline_*.tar.gz && cd pointcept_a40_offline_* && bash install.sh

set -euo pipefail

POINTCEPT_COMMIT="${POINTCEPT_COMMIT:-df36980119f4636beb2d02d04ef3b2fec0fddfba}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/releases/a40_offline}"
DATE_TAG="$(date +%Y%m%d)"
STAGING="${OUTPUT_DIR}/.staging_pointcept_a40"
# BUNDLE_NAME set in main() after resolve_commit()

log() { printf '[package-a40] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 1; }

resolve_commit() {
  if [[ -d "${REPO_ROOT}/.git" ]]; then
    git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "${POINTCEPT_COMMIT}"
  else
    echo "${POINTCEPT_COMMIT}"
  fi
}

copy_tree() {
  local src="$1"
  local dst="$2"
  [[ -d "${src}" ]] || die "missing ${src}"
  mkdir -p "${dst}"
  rsync -a \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'build/' \
    --exclude 'dist/' \
    --exclude '*.egg-info' \
    --exclude '.git' \
    "${src}/" "${dst}/"
}

download_wheels() {
  local wheel_dir="$1"
  mkdir -p "${wheel_dir}"
  log "Downloading ninja wheel for x86_64 A40 host (offline pip install) ..."
  # Packaging host may be aarch64; force manylinux x86_64 wheel for target A40 machine.
  local pyver
  pyver="$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
  if python3 -m pip download \
    -d "${wheel_dir}" \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --platform manylinux_2_17_x86_64 \
    --python-version "${pyver}" \
    --implementation cp \
    ninja 2>/dev/null; then
    log "Wheels: $(ls "${wheel_dir}")"
  elif python3 -m pip download -d "${wheel_dir}" \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --python-version "${pyver}" \
    ninja 2>/dev/null; then
    log "Wheels: $(ls "${wheel_dir}")"
  else
    log "WARN: could not fetch x86_64 ninja wheel — A40 host must have ninja pre-installed"
    rm -f "${wheel_dir}"/ninja-*aarch64*.whl 2>/dev/null || true
  fi
}

write_version() {
  local commit="$1"
  local out="$2/VERSION.txt"
  {
    echo "pointcept_commit=${commit}"
    echo "official_url=https://github.com/Pointcept/Pointcept/tree/${commit}/libs"
    echo "packed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "packed_from=${REPO_ROOT}"
    uname -a 2>/dev/null || true
  } > "${out}"
}

write_manifest() {
  local root="$1"
  local manifest="${root}/MANIFEST.sha256"
  (cd "${root}" && find . -type f ! -name 'MANIFEST.sha256' -print0 | sort -z \
    | xargs -0 sha256sum) > "${manifest}"
  log "Manifest: ${manifest} ($(wc -l < "${manifest}") files)"
}

main() {
  [[ -f "${REPO_ROOT}/libs/pointops/setup.py" ]] || die "run from Pointcept repo root"
  commit="$(resolve_commit)"
  BUNDLE_NAME="pointcept_a40_offline_${commit:0:7}_${DATE_TAG}"
  bundle="${STAGING}/${BUNDLE_NAME}"

  log "Commit: ${commit}"
  log "Staging: ${bundle}"

  rm -rf "${bundle}"
  mkdir -p "${bundle}/libs" "${bundle}/wheels" "${bundle}/artifacts"

  for lib in pointops pointops2 pointops_torch; do
    copy_tree "${REPO_ROOT}/libs/${lib}" "${bundle}/libs/${lib}"
  done

  # Optional: pointgroup_ops source (not built by default offline)
  if [[ -d "${REPO_ROOT}/libs/pointgroup_ops" ]]; then
    copy_tree "${REPO_ROOT}/libs/pointgroup_ops" "${bundle}/libs/pointgroup_ops"
  fi

  cp "${SCRIPT_DIR}/install_pointcept_libs_offline.sh" "${bundle}/install.sh"
  cp "${SCRIPT_DIR}/run_benchmark_a40.sh" "${bundle}/run_benchmark.sh"
  cp "${SCRIPT_DIR}/README_A40_OFFLINE.md" "${bundle}/README_A40_OFFLINE.md"

  write_version "${commit}" "${bundle}"
  download_wheels "${bundle}/wheels"
  write_manifest "${bundle}"

  mkdir -p "${OUTPUT_DIR}"
  tarball="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
  tar -czf "${tarball}" -C "${STAGING}" "${BUNDLE_NAME}"

  rm -rf "${STAGING}"
  log "Created: ${tarball}"
  log "Size: $(du -h "${tarball}" | cut -f1)"
  (cd "${OUTPUT_DIR}" && sha256sum "$(basename "${tarball}")") | tee "${tarball}.sha256"
  log ""
  log "Copy to A40:"
  log "  scp ${tarball} ${tarball}.sha256 user@a40-host:/path/"
  log "On A40:"
  log "  sha256sum -c $(basename "${tarball}").sha256"
  log "  tar xzf ${BUNDLE_NAME}.tar.gz && cd ${BUNDLE_NAME} && bash install.sh"
}

main "$@"
