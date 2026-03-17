#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${ROOT_DIR}/third_party/minicpm-o-4_5-pytorch-simple-demo"
VENV_PYTHON="${DEMO_DIR}/.venv/base/bin/python"
CONFIG_PATH="${DEMO_DIR}/config.json"
PROTO_MARKER="${DEMO_DIR}/tmp/gateway.proto"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing demo virtualenv: ${VENV_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing demo config: ${CONFIG_PATH}" >&2
  exit 1
fi

GATEWAY_PORT="$("${VENV_PYTHON}" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path(${CONFIG_PATH@Q}).read_text())
print(cfg["service"]["gateway_port"])
PY
)"
WORKER_PORT="$("${VENV_PYTHON}" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path(${CONFIG_PATH@Q}).read_text())
print(cfg["service"]["worker_base_port"])
PY
)"

if [[ -f "${PROTO_MARKER}" ]]; then
  GATEWAY_PROTO="$(tr -d '[:space:]' < "${PROTO_MARKER}")"
else
  GATEWAY_PROTO="https"
fi

if [[ "${GATEWAY_PROTO}" == "https" ]]; then
  CURL_GATEWAY=(curl -sk)
else
  CURL_GATEWAY=(curl -s)
fi

echo "==> Gateway health"
"${CURL_GATEWAY[@]}" "${GATEWAY_PROTO}://127.0.0.1:${GATEWAY_PORT}/health"
echo
echo "==> Worker health"
curl -s "http://127.0.0.1:${WORKER_PORT}/health"
echo
echo "==> Simple chat smoke"
"${CURL_GATEWAY[@]}" "${GATEWAY_PROTO}://127.0.0.1:${GATEWAY_PORT}/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"1+1等于几？只回答数字。"}],"generation":{"max_new_tokens":10,"do_sample":false}}'
echo
