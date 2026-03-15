#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
LLAMA_CPP_OMNI_DIR="${THIRD_PARTY_DIR}/llama.cpp-omni"
WORKER_SOURCE_DIR="${ROOT_DIR}/local_duplex/gguf_worker"
BUILD_DIR="${ROOT_DIR}/build/local_duplex_gguf_worker"
VENV_DIR="${ROOT_DIR}/.venv/local-duplex311"
if command -v "cmake" >/dev/null 2>&1; then
  CMAKE_BIN="$(command -v cmake)"
elif [ -x "${VENV_DIR}/bin/cmake" ]; then
  CMAKE_BIN="${VENV_DIR}/bin/cmake"
else
  echo "Missing cmake. Run scripts/setup_duplex_env.sh first." >&2
  exit 1
fi

mkdir -p "${THIRD_PARTY_DIR}"

if [ ! -d "${LLAMA_CPP_OMNI_DIR}/.git" ]; then
  git clone --depth 1 "https://github.com/tc-mb/llama.cpp-omni.git" "${LLAMA_CPP_OMNI_DIR}"
fi

"${CMAKE_BIN}" -S "${WORKER_SOURCE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_CPP_OMNI_DIR="${LLAMA_CPP_OMNI_DIR}"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target "local-duplex-gguf-worker" -j

echo "GGUF backend worker built at ${BUILD_DIR}/bin/local-duplex-gguf-worker"
