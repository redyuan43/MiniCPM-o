#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARGS=("$@")
if [ "${#ARGS[@]}" -eq 0 ]; then
  ARGS=("omni")
fi

"${ROOT_DIR}/scripts/stop_local_duplex.sh" >/dev/null 2>&1 || true
"${ROOT_DIR}/scripts/start_local_duplex.sh" "${ARGS[@]}"

sleep 3
echo
echo "Recent runtime log:"
tail -n 80 "${ROOT_DIR}/.local_duplex/runtime.log"
echo
echo "Latest interaction summary:"
"${ROOT_DIR}/scripts/show_local_duplex_session.sh" || true
