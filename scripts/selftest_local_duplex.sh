#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv/local-duplex311"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Missing ${VENV_DIR}. Run scripts/setup_duplex_env.sh first." >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/MiniCPM-o-Demo:${PYTHONPATH:-}"

"${VENV_DIR}/bin/python" -m local_duplex.selftest "$@"
