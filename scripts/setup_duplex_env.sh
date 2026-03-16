#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv/local-duplex311"
PYTHON_VERSION="${LOCAL_DUPLEX_PYTHON_VERSION:-3.11}"
ARCH="$(uname -m)"
PYTHON_TAG="cp${PYTHON_VERSION/.}"
TORCH_WHEEL="${LOCAL_DUPLEX_TORCH_WHEEL:-}"
TORCHVISION_WHEEL="${LOCAL_DUPLEX_TORCHVISION_WHEEL:-}"
TORCHAUDIO_WHEEL="${LOCAL_DUPLEX_TORCHAUDIO_WHEEL:-}"
SKIP_TORCHVISION="${LOCAL_DUPLEX_SKIP_TORCHVISION:-0}"
SKIP_TORCHAUDIO="${LOCAL_DUPLEX_SKIP_TORCHAUDIO:-0}"
USE_SYSTEM_SITE_PACKAGES="${LOCAL_DUPLEX_USE_SYSTEM_SITE_PACKAGES:-0}"
SKIP_TORCH_INSTALL="${LOCAL_DUPLEX_SKIP_TORCH_INSTALL:-0}"
INSTALL_PYTORCH_BACKEND_DEPS="${LOCAL_DUPLEX_INSTALL_PYTORCH_BACKEND_DEPS:-1}"

cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed." >&2
  exit 1
fi

discover_wheel() {
  local package_name="$1"
  local wheel
  for wheel in \
    "${HOME}/ComfyUI/${package_name}"-*-"${PYTHON_TAG}"-"${PYTHON_TAG}"-*aarch64.whl \
    "${HOME}/${package_name}"-*-"${PYTHON_TAG}"-"${PYTHON_TAG}"-*aarch64.whl
  do
    if [ -f "${wheel}" ]; then
      printf '%s\n' "${wheel}"
      return 0
    fi
  done
  return 1
}

install_optional_wheel() {
  local wheel_path="$1"
  if [ -n "${wheel_path}" ] && [ -f "${wheel_path}" ]; then
    "${VENV_DIR}/bin/python" -m pip install "${wheel_path}"
    return 0
  fi
  return 1
}

mkdir -p "${ROOT_DIR}/third_party"
mkdir -p "${ROOT_DIR}/third_party/models/huggingface/hub"
if [ ! -d "${ROOT_DIR}/third_party/MiniCPM-o-Demo" ]; then
  git clone --depth 1 "https://github.com/OpenBMB/MiniCPM-o-Demo.git" "${ROOT_DIR}/third_party/MiniCPM-o-Demo"
elif [ ! -f "${ROOT_DIR}/third_party/MiniCPM-o-Demo/worker.py" ]; then
  echo "Existing third_party/MiniCPM-o-Demo is missing expected project files." >&2
  exit 1
fi

uv python install "${PYTHON_VERSION}"
if [ "${USE_SYSTEM_SITE_PACKAGES}" = "1" ]; then
  uv venv --seed --clear --python "${PYTHON_VERSION}" --system-site-packages "${VENV_DIR}"
else
  uv venv --seed --clear --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade "pip<26.1" "setuptools<81" wheel

if [ "${SKIP_TORCH_INSTALL}" != "1" ] && [ -z "${TORCH_WHEEL}" ] && [ "${ARCH}" = "aarch64" ]; then
  TORCH_WHEEL="$(discover_wheel "torch" || true)"
  TORCHVISION_WHEEL="${TORCHVISION_WHEEL:-$(discover_wheel "torchvision" || true)}"
  TORCHAUDIO_WHEEL="${TORCHAUDIO_WHEEL:-$(discover_wheel "torchaudio" || true)}"
  if [ -n "${TORCH_WHEEL}" ]; then
    SKIP_TORCHVISION="${LOCAL_DUPLEX_SKIP_TORCHVISION:-1}"
    SKIP_TORCHAUDIO="${LOCAL_DUPLEX_SKIP_TORCHAUDIO:-1}"
  fi
fi

if [ "${SKIP_TORCH_INSTALL}" != "1" ]; then
  if [ -n "${TORCH_WHEEL}" ]; then
    "${VENV_DIR}/bin/python" -m pip install "${TORCH_WHEEL}"
    if [ "${SKIP_TORCHVISION}" != "1" ] && ! install_optional_wheel "${TORCHVISION_WHEEL}"; then
      "${VENV_DIR}/bin/python" -m pip install "torchvision==0.21.0"
    fi
    if [ "${SKIP_TORCHAUDIO}" != "1" ] && ! install_optional_wheel "${TORCHAUDIO_WHEEL}"; then
      "${VENV_DIR}/bin/python" -m pip install "torchaudio==2.6.0"
    fi
  else
    "${VENV_DIR}/bin/python" -m pip install \
      --index-url "https://download.pytorch.org/whl/cu124" \
      "torch==2.6.0" \
      "torchvision==0.21.0" \
      "torchaudio==2.6.0"
  fi
fi

"${VENV_DIR}/bin/python" -m pip install \
  "cmake>=3.30.0" \
  "ninja>=1.11.1" \
  "transformers==4.51.0" \
  "accelerate==1.12.0" \
  "modelscope>=1.30.0" \
  "safetensors>=0.7.0" \
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
  "sounddevice>=0.5.2" \
  "soundfile>=0.13.1" \
  "edge-tts>=7.2.0" \
  "pillow==10.4.0" \
  "librosa==0.9.0" \
  "moviepy==2.1.2" \
  "onnxruntime==1.21.0" \
  "onnx>=1.20.1" \
  "hyperpyyaml>=1.2.2" \
  "einops==0.8.1"

if [ "${INSTALL_PYTORCH_BACKEND_DEPS}" = "1" ]; then
  "${VENV_DIR}/bin/python" -m pip install \
    "autoawq>=0.1.8" \
    "minicpmo-utils==1.0.6"
fi

if [ "${ARCH}" != "aarch64" ]; then
  "${VENV_DIR}/bin/python" -m pip install "decord==0.6.0"
fi

echo "Local duplex environment is ready at ${VENV_DIR}"
