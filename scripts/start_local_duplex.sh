#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv/local-duplex311"
RUNTIME_DIR="${ROOT_DIR}/.local_duplex"
LOCAL_HF_HOME="${ROOT_DIR}/third_party/models/huggingface"
CONFIG_PATH="${LOCAL_DUPLEX_CONFIG:-${ROOT_DIR}/configs/local_duplex.json}"

MODE="omni"
FORWARD_ARGS=()

if [ "$#" -gt 0 ]; then
  case "${1}" in
    audio|omni)
      MODE="${1}"
      shift
      ;;
  esac
fi

if [ "$#" -gt 0 ]; then
  FORWARD_ARGS=("$@")
fi

mkdir -p "${RUNTIME_DIR}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Missing ${VENV_DIR}. Run scripts/setup_duplex_env.sh first." >&2
  exit 1
fi

if [ "${MODE}" != "audio" ] && [ "${MODE}" != "omni" ]; then
  echo "Mode must be 'audio' or 'omni'." >&2
  exit 1
fi

if [ -f "${RUNTIME_DIR}/runtime.pid" ] && kill -0 "$(cat "${RUNTIME_DIR}/runtime.pid")" 2>/dev/null; then
  echo "Local duplex runtime is already running." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${LOCAL_HF_HOME}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${LOCAL_HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/MiniCPM-o-Demo:${PYTHONPATH:-}"

nohup "${VENV_DIR}/bin/python" -u -m local_duplex.cli \
  "${MODE}" \
  "${FORWARD_ARGS[@]}" \
  --config "${CONFIG_PATH}" \
  > "${RUNTIME_DIR}/runtime.log" 2>&1 &

echo $! > "${RUNTIME_DIR}/runtime.pid"
echo "Started local duplex runtime (mode=${MODE}, pid=$(cat "${RUNTIME_DIR}/runtime.pid"))"
