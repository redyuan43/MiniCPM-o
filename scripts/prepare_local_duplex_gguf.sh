#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/third_party/models/modelscope/MiniCPM-o-4_5-gguf"
REPO_ID="OpenBMB/MiniCPM-o-4_5-gguf"
VARIANT="${1:-Q4_K_M}"

if command -v "modelscope" >/dev/null 2>&1; then
  MODELSCOPE_BIN="$(command -v "modelscope")"
elif [ -x "${HOME}/.local/bin/modelscope" ]; then
  MODELSCOPE_BIN="${HOME}/.local/bin/modelscope"
else
  echo "Missing modelscope CLI. Install it first, or run scripts/setup_duplex_env.sh." >&2
  exit 1
fi

case "${VARIANT}" in
  Q4_K_M)
    MAIN_FILE="MiniCPM-o-4_5-Q4_K_M.gguf"
    ;;
  *)
    echo "Unsupported GGUF variant: ${VARIANT}" >&2
    echo "Currently supported variants: Q4_K_M" >&2
    exit 1
    ;;
esac

mkdir -p "${MODEL_DIR}"

"${MODELSCOPE_BIN}" download \
  --model "${REPO_ID}" \
  --local_dir "${MODEL_DIR}" \
  --include \
  "${MAIN_FILE}" \
  "audio/*" \
  "vision/*" \
  "tts/*" \
  "token2wav-gguf/*"

echo "GGUF model prepared at ${MODEL_DIR}"
