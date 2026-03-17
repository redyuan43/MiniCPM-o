#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${ROOT_DIR}/third_party/minicpm-o-4_5-pytorch-simple-demo"
HF_HOME_DIR="${ROOT_DIR}/third_party/models/huggingface"
CONFIG_PATH="${DEMO_DIR}/config.json"
VENV_PYTHON="${DEMO_DIR}/.venv/base/bin/python"
CERT_DIR="${DEMO_DIR}/certs"
CERT_FILE="${CERT_DIR}/cert.pem"
KEY_FILE="${CERT_DIR}/key.pem"
PROTO_MARKER="${DEMO_DIR}/tmp/gateway.proto"
WORKER_READY_TIMEOUT_S="${WORKER_READY_TIMEOUT_S:-240}"
GATEWAY_READY_TIMEOUT_S="${GATEWAY_READY_TIMEOUT_S:-60}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  bash "${ROOT_DIR}/scripts/bootstrap_minicpmo45_pytorch_demo_spark.sh"
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing demo virtualenv python: ${VENV_PYTHON}" >&2
  exit 1
fi

export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}/hub"

use_http=0
for arg in "$@"; do
  case "${arg}" in
    --http)
      use_http=1
      ;;
    *)
      echo "Unsupported argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${DEMO_DIR}/tmp"

if [[ "${use_http}" == "0" && ( ! -f "${CERT_FILE}" || ! -f "${KEY_FILE}" ) ]]; then
  if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl is required to generate self-signed certs for HTTPS" >&2
    exit 1
  fi
  mkdir -p "${CERT_DIR}"
  echo "==> Generating self-signed TLS cert for MiniCPM-o gateway"
  openssl req -x509 -newkey rsa:2048 -sha256 -days 365 -nodes \
    -keyout "${KEY_FILE}" \
    -out "${CERT_FILE}" \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" >/dev/null 2>&1
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

echo "=================================================="
echo "  MiniCPMO45 Service Launcher"
echo "=================================================="
echo "  GPUs: ${GPU_LIST} (${NUM_GPUS})"

if [[ "${use_http}" == "1" ]]; then
  echo "  Gateway: http://localhost:${GATEWAY_PORT}"
else
  echo "  Gateway: https://localhost:${GATEWAY_PORT}"
fi

echo "  Workers: localhost:${WORKER_BASE_PORT} ~ localhost:$((WORKER_BASE_PORT + NUM_GPUS - 1)) (HTTP, internal)"
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
      --port "${GATEWAY_PORT}" \
      --workers "${WORKER_ADDRS}" \
      > "${gateway_log_path}" 2>&1
fi

sleep 1
pgrep -n -f "gateway.py --port ${GATEWAY_PORT} --workers ${WORKER_ADDRS}" > "tmp/gateway.pid"

if [[ "${use_http}" == "1" ]]; then
  wait_for_gateway_ready "http"
else
  wait_for_gateway_ready "https"
fi

echo
echo "=================================================="
echo "  Service is running!"

if [[ "${use_http}" == "1" ]]; then
  echo "  Chat Demo:  http://localhost:${GATEWAY_PORT}"
  echo "  Admin:      http://localhost:${GATEWAY_PORT}/admin"
  echo "  API Docs:   http://localhost:${GATEWAY_PORT}/docs"
else
  echo "  Chat Demo:  https://localhost:${GATEWAY_PORT}"
  echo "  Admin:      https://localhost:${GATEWAY_PORT}/admin"
  echo "  API Docs:   https://localhost:${GATEWAY_PORT}/docs"
fi

echo "  Workers:    ${WORKER_ADDRS}"
echo
echo "  Logs:"
echo "    Gateway:  tmp/gateway.log"
echo "    Workers:  tmp/worker_*.log"
echo
echo "  To stop:"
echo "    bash scripts/stop_minicpmo45_pytorch_demo_spark.sh"
echo "=================================================="
