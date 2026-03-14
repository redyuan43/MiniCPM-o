#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT_DIR}/.local_duplex/runtime.pid"
VENV_PYTHON="${ROOT_DIR}/.venv/local-duplex311/bin/python"

declare -a PIDS=()

append_pid() {
  local pid="$1"
  if [ -z "${pid}" ]; then
    return
  fi
  if ! [[ "${pid}" =~ ^[0-9]+$ ]]; then
    return
  fi
  local existing
  for existing in "${PIDS[@]:-}"; do
    if [ "${existing}" = "${pid}" ]; then
      return
    fi
  done
  PIDS+=("${pid}")
}

if [ -f "${PID_FILE}" ]; then
  append_pid "$(cat "${PID_FILE}")"
fi

while IFS= read -r pid; do
  append_pid "${pid}"
done < <(pgrep -f "${VENV_PYTHON} -u -m local_duplex.cli" || true)

if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "No local duplex runtime process found."
  rm -f "${PID_FILE}"
  exit 0
fi

kill -TERM "${PIDS[@]}" 2>/dev/null || true

for _ in 1 2 3 4 5 6 7 8 9 10; do
  sleep 0.5
  declare -a REMAINING=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      REMAINING+=("${pid}")
    fi
  done
  if [ "${#REMAINING[@]}" -eq 0 ]; then
    rm -f "${PID_FILE}"
    echo "Stopped local duplex runtime pid(s): ${PIDS[*]}"
    exit 0
  fi
  PIDS=("${REMAINING[@]}")
done

kill -KILL "${PIDS[@]}" 2>/dev/null || true
rm -f "${PID_FILE}"
echo "Force-stopped local duplex runtime pid(s): ${PIDS[*]}"
