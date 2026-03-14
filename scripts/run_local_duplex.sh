#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-omni}"

"${ROOT_DIR}/scripts/stop_local_duplex.sh" >/dev/null 2>&1 || true
"${ROOT_DIR}/scripts/start_local_duplex.sh" "${MODE}"

sleep 3
echo
echo "Recent runtime log:"
tail -n 80 "${ROOT_DIR}/.local_duplex/runtime.log"
