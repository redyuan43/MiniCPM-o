#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${ROOT_DIR}/third_party/minicpm-o-4_5-pytorch-simple-demo"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/third_party/models/modelscope/MiniCPM-o-4_5}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
SHARED_VENV_PATH="${SHARED_VENV_PATH:-${ROOT_DIR}/.venv/local-duplex311}"
GATEWAY_PORT="${GATEWAY_PORT:-18006}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-22400}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
CHAT_VOCODER="${CHAT_VOCODER:-token2wav}"
DATA_DIR="${DATA_DIR:-data-spark}"
DOWNLOAD_METHOD="${DOWNLOAD_METHOD:-modelscope}"
FORCE_CONFIG=0
SKIP_MODEL_DOWNLOAD=0
NO_REUSE_SHARED_VENV=0

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_minicpmo45_pytorch_demo_spark.sh [options]

Options:
  --force-config        Rewrite config.json even if it already exists.
  --skip-model-download Do not download the model if missing.
  --gateway-port PORT   Override gateway port. Default: 18006
  --worker-base-port N  Override worker base port. Default: 22400
  --model-dir PATH      Override local model directory.
  --python-bin BIN      Override Python interpreter. Default: python3.11
  --download-method M   One of: modelscope, huggingface. Default: modelscope
  --no-reuse-shared-venv
                       Build a dedicated .venv/base instead of reusing ${ROOT_DIR}/.venv/local-duplex311.
  -h, --help            Show this help.

Environment overrides:
  MODEL_DIR, PYTHON_BIN, GATEWAY_PORT, WORKER_BASE_PORT, ATTN_IMPLEMENTATION,
  CHAT_VOCODER, DATA_DIR, DOWNLOAD_METHOD, SHARED_VENV_PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-config)
      FORCE_CONFIG=1
      shift
      ;;
    --skip-model-download)
      SKIP_MODEL_DOWNLOAD=1
      shift
      ;;
    --gateway-port)
      GATEWAY_PORT="$2"
      shift 2
      ;;
    --worker-base-port)
      WORKER_BASE_PORT="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --download-method)
      DOWNLOAD_METHOD="$2"
      shift 2
      ;;
    --no-reuse-shared-venv)
      NO_REUSE_SHARED_VENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${DEMO_DIR}" ]]; then
  echo "Missing demo repo: ${DEMO_DIR}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "==> Demo dir: ${DEMO_DIR}"
echo "==> Model dir: ${MODEL_DIR}"
echo "==> Python: ${PYTHON_BIN}"

cd "${DEMO_DIR}"

if [[ "${NO_REUSE_SHARED_VENV}" == "0" && -x "${SHARED_VENV_PATH}/bin/python" ]]; then
  echo "==> Reusing shared CUDA venv: ${SHARED_VENV_PATH}"
  mkdir -p .venv
  rm -rf .venv/base
  ln -s "${SHARED_VENV_PATH}" .venv/base
else
  if [[ -d ".venv/base" ]]; then
    echo "==> Reusing existing demo environment and completing dependency install"
  else
    echo "==> Installing demo environment"
  fi
  PYTHON="${PYTHON_BIN}" bash ./install.sh
fi

VENV_PYTHON="${DEMO_DIR}/.venv/base/bin/python"
VENV_PIP="${DEMO_DIR}/.venv/base/bin/pip"
VENV_MODELSCOPE="${DEMO_DIR}/.venv/base/bin/modelscope"
VENV_HF_CLI="${DEMO_DIR}/.venv/base/bin/huggingface-cli"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Virtual environment python missing: ${VENV_PYTHON}" >&2
  exit 1
fi

ensure_python_deps() {
  local missing=()
  local mod pkg

  while read -r mod pkg; do
    if ! "${VENV_PYTHON}" -c "import ${mod}" >/dev/null 2>&1; then
      missing+=("${pkg}")
    fi
  done <<'EOF'
packaging packaging
transformers transformers==4.51.0
accelerate accelerate==1.12.0
fastapi fastapi>=0.128.0
uvicorn uvicorn>=0.40.0
httpx httpx>=0.28.0
websockets websockets>=16.0
pydantic pydantic>=2.11.0
numpy numpy>=2.2.0
PIL pillow==10.4.0
librosa librosa==0.9.0
soundfile soundfile
yaml PyYAML
onnxruntime onnxruntime
python_multipart python-multipart
safetensors safetensors>=0.7.0
stepaudio2 stepaudio2-minicpmo
tqdm tqdm>=4.67.0
pytest pytest>=9.0.0
pytest_asyncio pytest-asyncio>=1.3.0
EOF

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "==> Installing missing Python dependencies: ${missing[*]}"
    "${VENV_PIP}" install "${missing[@]}"
  fi
}

ensure_python_deps

ensure_modelscope() {
  if ! "${VENV_PYTHON}" -c "import modelscope" >/dev/null 2>&1; then
    echo "==> Installing modelscope for model download"
    "${VENV_PIP}" install modelscope
  fi
}

download_model() {
  mkdir -p "${MODEL_DIR}"
  case "${DOWNLOAD_METHOD}" in
    modelscope)
      ensure_modelscope
      echo "==> Downloading MiniCPM-o 4.5 from ModelScope"
      "${VENV_MODELSCOPE}" download --model OpenBMB/MiniCPM-o-4_5 --local_dir "${MODEL_DIR}"
      ;;
    huggingface)
      echo "==> Downloading MiniCPM-o 4.5 from Hugging Face"
      "${VENV_HF_CLI}" download openbmb/MiniCPM-o-4_5 --local-dir "${MODEL_DIR}"
      ;;
    *)
      echo "Unsupported download method: ${DOWNLOAD_METHOD}" >&2
      exit 1
      ;;
  esac
}

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  if [[ "${SKIP_MODEL_DOWNLOAD}" == "1" ]]; then
    echo "Model missing and --skip-model-download was set: ${MODEL_DIR}" >&2
    exit 1
  fi
  download_model
else
  echo "==> Reusing existing local model"
fi

if [[ "${FORCE_CONFIG}" == "1" || ! -f "${DEMO_DIR}/config.json" ]]; then
  echo "==> Writing config.json"
  "${VENV_PYTHON}" - <<PY
import json
from pathlib import Path

demo_dir = Path(${DEMO_DIR@Q})
config_path = demo_dir / "config.json"
cfg = {
    "model": {
        "model_path": ${MODEL_DIR@Q},
        "pt_path": None,
        "attn_implementation": ${ATTN_IMPLEMENTATION@Q},
    },
    "audio": {
        "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav",
        "playback_delay_ms": 200,
        "chat_vocoder": ${CHAT_VOCODER@Q},
    },
    "service": {
        "gateway_port": int(${GATEWAY_PORT@Q}),
        "worker_base_port": int(${WORKER_BASE_PORT@Q}),
        "max_queue_size": 1000,
        "request_timeout": 300.0,
        "compile": False,
        "data_dir": ${DATA_DIR@Q},
        "eta_chat_s": 15.0,
        "eta_half_duplex_s": 20.0,
        "eta_audio_duplex_s": 120.0,
        "eta_omni_duplex_s": 90.0,
        "eta_ema_alpha": 0.3,
        "eta_ema_min_samples": 3,
    },
    "duplex": {
        "pause_timeout": 60.0,
    },
}
config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=True) + "\n")
PY
else
  echo "==> Keeping existing config.json"
fi

echo
echo "Bootstrap complete."
echo "  Config: ${DEMO_DIR}/config.json"
echo "  Model : ${MODEL_DIR}"
echo "  Start : bash scripts/start_minicpmo45_pytorch_demo_spark.sh"
