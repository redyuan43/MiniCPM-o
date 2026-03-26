#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${ROOT_DIR}/third_party/minicpm-o-4_5-pytorch-simple-demo"
HF_HOME_DIR="${ROOT_DIR}/third_party/models/huggingface"
CONFIG_PATH="${DEMO_DIR}/config.json"
VENV_PYTHON="${DEMO_DIR}/.venv/base/bin/python"
DEFAULT_CERT_FILE="${DEMO_DIR}/certs/cert.pem"
DEFAULT_KEY_FILE="${DEMO_DIR}/certs/key.pem"
GENERATE_CERT_SCRIPT="${ROOT_DIR}/scripts/generate_minicpmo45_lan_cert.sh"
PROTO_MARKER="${DEMO_DIR}/tmp/gateway.proto"
WORKER_READY_TIMEOUT_S="${WORKER_READY_TIMEOUT_S:-240}"
GATEWAY_READY_TIMEOUT_S="${GATEWAY_READY_TIMEOUT_S:-60}"

GATEWAY_HOST="${MINICPMO_GATEWAY_HOST:-0.0.0.0}"
PUBLIC_HOST="${MINICPMO_PUBLIC_HOST:-localhost}"
TLS_EXTRA_DNS="${MINICPMO_TLS_EXTRA_DNS:-}"
TLS_EXTRA_IPS="${MINICPMO_TLS_EXTRA_IPS:-}"
CERT_FILE="${MINICPMO_SSL_CERTFILE:-${DEFAULT_CERT_FILE}}"
KEY_FILE="${MINICPMO_SSL_KEYFILE:-${DEFAULT_KEY_FILE}}"
REGENERATE_CERT=0
use_http=0

usage() {
  cat <<'EOF'
Usage: bash scripts/start_minicpmo45_pytorch_demo_spark.sh [options]

Start the official MiniCPM-o 4.5 PyTorch realtime demo for local or LAN access.

Options:
  --http                  Run without HTTPS/TLS
  --gateway-host HOST     Bind host for the Gateway process. Default: 0.0.0.0
  --public-host HOST      Hostname or IP remote browsers should open. Default: localhost
  --tls-extra-dns LIST    Comma-separated extra DNS SAN entries for auto-generated cert
  --tls-extra-ip LIST     Comma-separated extra IP SAN entries for auto-generated cert
  --ssl-certfile PATH     TLS cert path. Relative paths resolve from repo root
  --ssl-keyfile PATH      TLS key path. Relative paths resolve from repo root
  --regen-cert            Regenerate the managed self-signed cert before startup
  -h, --help              Show this help

Environment overrides:
  MINICPMO_GATEWAY_HOST
  MINICPMO_PUBLIC_HOST
  MINICPMO_TLS_EXTRA_DNS
  MINICPMO_TLS_EXTRA_IPS
  MINICPMO_SSL_CERTFILE
  MINICPMO_SSL_KEYFILE
  WORKER_READY_TIMEOUT_S
  GATEWAY_READY_TIMEOUT_S
EOF
}

resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${ROOT_DIR}/${value}"
  fi
}

is_ip_literal() {
  local value="$1"
  [[ "${value}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ || "${value}" == *:* ]]
}

cert_covers_host() {
  local cert_path="$1"
  local host="$2"
  local san_output

  if ! command -v openssl >/dev/null 2>&1; then
    return 1
  fi
  if [[ ! -f "${cert_path}" ]]; then
    return 1
  fi

  san_output="$(openssl x509 -in "${cert_path}" -noout -ext subjectAltName 2>/dev/null || true)"
  if [[ -z "${san_output}" ]]; then
    return 1
  fi

  if is_ip_literal "${host}"; then
    grep -Fq "IP Address:${host}" <<< "${san_output}"
  else
    grep -Eq "DNS:${host}([[:space:],]|$)" <<< "${san_output}"
  fi
}

generate_gateway_cert() {
  local reason="$1"
  echo "==> Generating TLS cert for MiniCPM-o gateway (${reason})"
  bash "${GENERATE_CERT_SCRIPT}" \
    --public-host "${PUBLIC_HOST}" \
    --tls-extra-dns "${TLS_EXTRA_DNS}" \
    --tls-extra-ip "${TLS_EXTRA_IPS}" \
    --certfile "${CERT_FILE}" \
    --keyfile "${KEY_FILE}" \
    --force
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --http)
      use_http=1
      shift
      ;;
    --gateway-host)
      GATEWAY_HOST="$2"
      shift 2
      ;;
    --public-host)
      PUBLIC_HOST="$2"
      shift 2
      ;;
    --tls-extra-dns)
      TLS_EXTRA_DNS="$2"
      shift 2
      ;;
    --tls-extra-ip|--tls-extra-ips)
      TLS_EXTRA_IPS="$2"
      shift 2
      ;;
    --ssl-certfile)
      CERT_FILE="$2"
      shift 2
      ;;
    --ssl-keyfile)
      KEY_FILE="$2"
      shift 2
      ;;
    --regen-cert)
      REGENERATE_CERT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unsupported argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${PUBLIC_HOST}" ]]; then
  echo "--public-host must not be empty" >&2
  exit 1
fi

CERT_FILE="$(resolve_path "${CERT_FILE}")"
KEY_FILE="$(resolve_path "${KEY_FILE}")"
MANAGED_CERT=0
if [[ "${CERT_FILE}" == "${DEFAULT_CERT_FILE}" && "${KEY_FILE}" == "${DEFAULT_KEY_FILE}" ]]; then
  MANAGED_CERT=1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  bash "${ROOT_DIR}/scripts/bootstrap_minicpmo45_pytorch_demo_spark.sh"
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing demo virtualenv python: ${VENV_PYTHON}" >&2
  exit 1
fi

export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}/hub"

mkdir -p "${DEMO_DIR}/tmp"

if [[ "${use_http}" == "0" ]]; then
  if [[ ! -x "${GENERATE_CERT_SCRIPT}" ]]; then
    echo "Missing TLS helper script: ${GENERATE_CERT_SCRIPT}" >&2
    exit 1
  fi

  if [[ "${REGENERATE_CERT}" == "1" ]]; then
    generate_gateway_cert "forced"
  elif [[ ! -f "${CERT_FILE}" || ! -f "${KEY_FILE}" ]]; then
    generate_gateway_cert "missing files"
  elif ! cert_covers_host "${CERT_FILE}" "${PUBLIC_HOST}"; then
    if [[ "${MANAGED_CERT}" == "1" ]]; then
      generate_gateway_cert "public host ${PUBLIC_HOST} missing from SAN"
    else
      echo "Provided TLS certificate does not cover public host '${PUBLIC_HOST}'." >&2
      echo "Pass a matching --ssl-certfile/--ssl-keyfile, or re-run with --regen-cert." >&2
      exit 1
    fi
  fi
fi

if [[ "${use_http}" == "1" ]]; then
  printf 'http\n' > "${PROTO_MARKER}"
else
  printf 'https\n' > "${PROTO_MARKER}"
fi

GATEWAY_PORT="$("${VENV_PYTHON}" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path(${CONFIG_PATH@Q}).read_text())
print(cfg["service"]["gateway_port"])
PY
)"

WORKER_BASE_PORT="$("${VENV_PYTHON}" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path(${CONFIG_PATH@Q}).read_text())
print(cfg["service"]["worker_base_port"])
PY
)"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_LIST="${CUDA_VISIBLE_DEVICES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
else
  GPU_LIST="0"
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_GPUS="${#GPU_IDS[@]}"

if [[ "${NUM_GPUS}" -eq 0 ]]; then
  echo "No GPUs selected." >&2
  exit 1
fi

wait_for_worker_ready() {
  local worker_port="$1"
  local worker_index="$2"
  local waited=0

  while (( waited < WORKER_READY_TIMEOUT_S )); do
    if curl -fsS "http://127.0.0.1:${worker_port}/health" 2>/dev/null | \
      "${VENV_PYTHON}" -c 'import json, sys; d = json.load(sys.stdin); sys.exit(0 if d.get("model_loaded") else 1)' >/dev/null 2>&1
    then
      echo "[Worker ${worker_index}] Ready ✓ (port ${worker_port})"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done

  echo "[Worker ${worker_index}] FAILED to start! Check tmp/worker_${worker_index}.log" >&2
  return 1
}

wait_for_gateway_ready() {
  local proto="$1"
  local curl_flags=(-fsS)
  local waited=0

  if [[ "${proto}" == "https" ]]; then
    curl_flags+=(-k)
  fi

  while (( waited < GATEWAY_READY_TIMEOUT_S )); do
    if curl "${curl_flags[@]}" "${proto}://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
      echo "[Gateway] Ready ✓"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done

  echo "[Gateway] FAILED to start! Check tmp/gateway.log" >&2
  return 1
}

cd "${DEMO_DIR}"
rm -f tmp/*.pid tmp/*.log

if [[ "${use_http}" == "1" ]]; then
  PUBLIC_URL="http://${PUBLIC_HOST}:${GATEWAY_PORT}"
else
  PUBLIC_URL="https://${PUBLIC_HOST}:${GATEWAY_PORT}"
fi

echo "=================================================="
echo "  MiniCPMO45 Service Launcher"
echo "=================================================="
echo "  GPUs: ${GPU_LIST} (${NUM_GPUS})"
echo "  Gateway bind: ${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "  Browser URL:  ${PUBLIC_URL}"
if [[ "${use_http}" == "0" ]]; then
  echo "  TLS cert:     ${CERT_FILE}"
  echo "  TLS key:      ${KEY_FILE}"
fi
echo "  Workers:      localhost:${WORKER_BASE_PORT} ~ localhost:$((WORKER_BASE_PORT + NUM_GPUS - 1)) (HTTP, internal)"
echo "=================================================="

WORKER_ADDRS=""

for idx in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$idx]}"
  worker_port="$((WORKER_BASE_PORT + idx))"
  log_path="${DEMO_DIR}/tmp/worker_${idx}.log"

  echo "[Worker ${idx}] Starting on GPU ${gpu_id}, port ${worker_port}..."

  setsid -f env \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PYTHONPATH=. \
    HF_HOME="${HF_HOME}" \
    HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
    "${VENV_PYTHON}" worker.py \
      --port "${worker_port}" \
      --gpu-id "${gpu_id}" \
      --worker-index "${idx}" \
      > "${log_path}" 2>&1

  sleep 1
  pgrep -n -f "worker.py --port ${worker_port} --gpu-id ${gpu_id} --worker-index ${idx}" > "tmp/worker_${idx}.pid"

  if [[ -z "${WORKER_ADDRS}" ]]; then
    WORKER_ADDRS="localhost:${worker_port}"
  else
    WORKER_ADDRS="${WORKER_ADDRS},localhost:${worker_port}"
  fi
done

echo
echo "Waiting for Workers to load models..."

for idx in "${!GPU_IDS[@]}"; do
  worker_port="$((WORKER_BASE_PORT + idx))"
  wait_for_worker_ready "${worker_port}" "${idx}"
done

echo
echo "[Gateway] Starting on port ${GATEWAY_PORT}..."

gateway_log_path="${DEMO_DIR}/tmp/gateway.log"
if [[ "${use_http}" == "1" ]]; then
  setsid -f env \
    PYTHONPATH=. \
    HF_HOME="${HF_HOME}" \
    HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
    "${VENV_PYTHON}" gateway.py \
      --host "${GATEWAY_HOST}" \
      --port "${GATEWAY_PORT}" \
      --workers "${WORKER_ADDRS}" \
      --http \
      > "${gateway_log_path}" 2>&1
else
  setsid -f env \
    PYTHONPATH=. \
    HF_HOME="${HF_HOME}" \
    HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
    "${VENV_PYTHON}" gateway.py \
      --host "${GATEWAY_HOST}" \
      --port "${GATEWAY_PORT}" \
      --workers "${WORKER_ADDRS}" \
      --ssl-certfile "${CERT_FILE}" \
      --ssl-keyfile "${KEY_FILE}" \
      > "${gateway_log_path}" 2>&1
fi

sleep 1
pgrep -n -f "gateway.py --host ${GATEWAY_HOST} --port ${GATEWAY_PORT} --workers ${WORKER_ADDRS}" > "tmp/gateway.pid"

if [[ "${use_http}" == "1" ]]; then
  wait_for_gateway_ready "http"
else
  wait_for_gateway_ready "https"
fi

echo
echo "=================================================="
echo "  Service is running!"
echo "  Browser URL: ${PUBLIC_URL}"
echo "  Admin:       ${PUBLIC_URL}/admin"
echo "  API Docs:    ${PUBLIC_URL}/docs"
echo "  Workers:     ${WORKER_ADDRS}"
echo
echo "  Logs:"
echo "    Gateway:  tmp/gateway.log"
echo "    Workers:  tmp/worker_*.log"
echo
echo "  To stop:"
echo "    bash scripts/stop_minicpmo45_pytorch_demo_spark.sh"
echo "=================================================="
