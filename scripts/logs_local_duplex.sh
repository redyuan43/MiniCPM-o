#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${ROOT_DIR}/.local_duplex/runtime.log"

if [ ! -f "${LOG_FILE}" ]; then
  echo "No runtime log file found."
  exit 1
fi

tail -n 200 -f "${LOG_FILE}"

