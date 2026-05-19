#!/usr/bin/env bash
# Push npu-pointops-torch commit to https://github.com/DHX98/Pointcept.git
# Run inside cuda_migration_test (or any host with this repo):
#
#   export GITHUB_TOKEN='ghp_xxxx'   # DHX98 account, repo scope
#   bash scripts/push_to_dhx98.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_NAME="${REMOTE_NAME:-dhx98}"
BRANCH_LOCAL="${BRANCH_LOCAL:-npu-pointops-torch}"
BRANCH_REMOTE="${BRANCH_REMOTE:-main}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: Set GITHUB_TOKEN (Personal Access Token for GitHub user DHX98)." >&2
  exit 1
fi

cd "${REPO_ROOT}"

if ! git rev-parse --verify "${BRANCH_LOCAL}" >/dev/null 2>&1; then
  echo "ERROR: branch ${BRANCH_LOCAL} not found. Commit your changes first." >&2
  exit 1
fi

git remote remove "${REMOTE_NAME}" 2>/dev/null || true
git remote add "${REMOTE_NAME}" "https://${GITHUB_TOKEN}@github.com/DHX98/Pointcept.git"

echo "Pushing ${BRANCH_LOCAL} -> ${REMOTE_NAME}/${BRANCH_REMOTE} ..."
git push "${REMOTE_NAME}" "${BRANCH_LOCAL}:${BRANCH_REMOTE}"

# Restore remote URL without token in config
git remote set-url "${REMOTE_NAME}" "https://github.com/DHX98/Pointcept.git"
echo "Done: https://github.com/DHX98/Pointcept/tree/${BRANCH_REMOTE}"
