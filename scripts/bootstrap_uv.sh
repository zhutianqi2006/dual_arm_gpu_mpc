#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

find_workspace_root() {
  if [[ -n "${WORKSPACE_ROOT:-}" && -f "${WORKSPACE_ROOT}/workspace.toml" ]]; then
    printf '%s\n' "${WORKSPACE_ROOT}"
    return
  fi

  local search_dir="${PROJECT_DIR}"
  while [[ "${search_dir}" != "/" ]]; do
    if [[ -f "${search_dir}/workspace.toml" ]]; then
      printf '%s\n' "${search_dir}"
      return
    fi
    search_dir="$(dirname "${search_dir}")"
  done

  printf '%s\n' "$(cd "${PROJECT_DIR}/.." && pwd)"
}

REPO_ROOT="$(find_workspace_root)"
CUDA_DQ_DIR="${REPO_ROOT}/cuda_dq_kernel"
CUROBO_DIR="${REPO_ROOT}/third_party/curobo"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export WORKSPACE_ROOT="${REPO_ROOT}"

if [[ ! -d "${CUDA_DQ_DIR}" ]]; then
  echo "[bootstrap_uv] expected dq_torch source at ${CUDA_DQ_DIR}, but it was not found" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

echo "[bootstrap_uv] syncing uv environment"
uv sync "$@"

VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "[bootstrap_uv] expected virtualenv interpreter at ${VENV_PYTHON}, but it was not found" >&2
  exit 1
fi

echo "[bootstrap_uv] installing dual_arm_gpu_mpc in editable mode"
uv pip install --python "${VENV_PYTHON}" -e "${PROJECT_DIR}"

echo "[bootstrap_uv] installing dq_torch from ${CUDA_DQ_DIR}"
uv pip install --python "${VENV_PYTHON}" "${CUDA_DQ_DIR}" --no-build-isolation

if [[ -d "${CUROBO_DIR}" ]]; then
  echo "[bootstrap_uv] installing curobo from ${CUROBO_DIR}"
  uv pip install --python "${VENV_PYTHON}" -e "${CUROBO_DIR}" --no-build-isolation
else
  echo "[bootstrap_uv] skipping curobo; clone it to ${CUROBO_DIR} first if you need collision checking"
fi

echo "[bootstrap_uv] done"
