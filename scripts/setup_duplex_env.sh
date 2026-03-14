#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv/local-duplex311"
PYTHON_VERSION="3.11"

cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed." >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/third_party"
mkdir -p "${ROOT_DIR}/third_party/models/huggingface/hub"
if [ ! -d "${ROOT_DIR}/third_party/MiniCPM-o-Demo/.git" ]; then
  git clone --depth 1 "https://github.com/OpenBMB/MiniCPM-o-Demo.git" "${ROOT_DIR}/third_party/MiniCPM-o-Demo"
fi

uv python install "${PYTHON_VERSION}"
uv venv --seed --python "${PYTHON_VERSION}" "${VENV_DIR}"

"${VENV_DIR}/bin/python" -m pip install --upgrade "pip<26.1" "setuptools<81" wheel
"${VENV_DIR}/bin/python" -m pip install \
  --index-url "https://download.pytorch.org/whl/cu124" \
  "torch==2.6.0" \
  "torchvision==0.21.0" \
  "torchaudio==2.6.0"
"${VENV_DIR}/bin/python" -m pip install \
  "transformers==4.51.0" \
  "accelerate==1.12.0" \
  "autoawq>=0.1.8" \
  "safetensors>=0.7.0" \
  "minicpmo-utils==1.0.6" \
  "fastapi>=0.128.0" \
  "uvicorn>=0.40.0" \
  "httpx>=0.28.0" \
  "websockets>=16.0" \
  "python-multipart" \
  "pydantic>=2.11.0" \
  "tqdm>=4.67.0" \
  "pytest>=9.0.0" \
  "pytest-asyncio>=1.3.0" \
  "opencv-python>=4.10.0" \
  "soundfile>=0.13.1" \
  "pillow==10.4.0" \
  "librosa==0.9.0" \
  "decord==0.6.0" \
  "moviepy==2.1.2" \
  "onnxruntime==1.21.0" \
  "onnx>=1.20.1" \
  "hyperpyyaml>=1.2.2" \
  "einops==0.8.1"

echo "Local duplex environment is ready at ${VENV_DIR}"
