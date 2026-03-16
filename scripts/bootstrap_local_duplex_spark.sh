#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CAMERA="/dev/v4l/by-id/usb-046d_Logitech_BRIO_AE03BDC5-video-index0"

RUN_SETUP=0
RUN_BUILD=0
RUN_PREPARE=0
RUN_AUDIO_FIX=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_local_duplex_spark.sh [--setup-env] [--build-worker] [--prepare-models] [--fix-audio-loop] [--all]

Prints the Spark-specific environment overrides that make local duplex bring-up
reproducible on aarch64 hosts. With flags, it can also run the setup/build/model
preparation steps using the discovered overrides.
EOF
}

pipewire_get_volume() {
  local target="$1"
  wpctl get-volume "${target}" 2>/dev/null | awk '/Volume:/ {print $2}'
}

ensure_pipewire_min_volume() {
  local target="$1"
  local label="$2"
  local minimum="$3"
  local current

  if ! command -v wpctl >/dev/null 2>&1; then
    echo "Skipping ${label} volume check: wpctl not found" >&2
    return 0
  fi

  current="$(pipewire_get_volume "${target}")"
  if [ -z "${current}" ]; then
    echo "Skipping ${label} volume check: unable to read current volume" >&2
    return 0
  fi

  if python3 - "${current}" "${minimum}" <<'PY'
import sys
current = float(sys.argv[1])
minimum = float(sys.argv[2])
raise SystemExit(0 if current + 1e-6 >= minimum else 1)
PY
  then
    echo "${label} volume already at ${current}"
    return 0
  fi

  wpctl set-volume "${target}" "${minimum}"
  echo "${label} volume raised from ${current} to ${minimum}"
}

find_capswriter_root() {
  local candidate
  for candidate in \
    "${LOCAL_DUPLEX_CAPSWRITER_ROOT:-}" \
    "${HOME}/github/CapsWriter-Offline-Windows-64bit-main" \
    "${HOME}/github/CapsWriter-Offline-Windows-64bit"
  do
    if [ -n "${candidate}" ] && [ -f "${candidate}/http_api_server.py" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

find_camera_device() {
  local candidate
  if [ -n "${LOCAL_DUPLEX_CAMERA_DEVICE:-}" ] && [ -e "${LOCAL_DUPLEX_CAMERA_DEVICE}" ]; then
    printf '%s\n' "${LOCAL_DUPLEX_CAMERA_DEVICE}"
    return 0
  fi
  if [ -e "${DEFAULT_CAMERA}" ]; then
    printf '%s\n' "${DEFAULT_CAMERA}"
    return 0
  fi
  for candidate in /dev/v4l/by-id/*EMEET*video-index0 /dev/v4l/by-id/*PIXY*video-index0 /dev/v4l/by-id/*video-index0; do
    if [ -e "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  for candidate in /dev/video0 /dev/video1; do
    if [ -e "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

find_torch_wheel() {
  local python_version="$1"
  local tag="cp${python_version/.}"
  local candidate
  for candidate in \
    "${LOCAL_DUPLEX_TORCH_WHEEL:-}" \
    "${HOME}/ComfyUI/torch-"*-"${tag}"-"${tag}"-*aarch64.whl \
    "${HOME}/torch-"*-"${tag}"-"${tag}"-*aarch64.whl
  do
    if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

system_python_version() {
  python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

system_torch_has_cuda() {
  python3 - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --setup-env)
      RUN_SETUP=1
      ;;
    --build-worker)
      RUN_BUILD=1
      ;;
    --prepare-models)
      RUN_PREPARE=1
      ;;
    --fix-audio-loop)
      RUN_AUDIO_FIX=1
      ;;
    --all)
      RUN_SETUP=1
      RUN_BUILD=1
      RUN_PREPARE=1
      RUN_AUDIO_FIX=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

ARCH="$(uname -m)"
CAPSWRITER_ROOT="$(find_capswriter_root || true)"
CAMERA_DEVICE="$(find_camera_device || true)"
PYTHON_VERSION="${LOCAL_DUPLEX_PYTHON_VERSION:-3.11}"
TORCH_WHEEL="${LOCAL_DUPLEX_TORCH_WHEEL:-}"
SKIP_TORCHVISION="${LOCAL_DUPLEX_SKIP_TORCHVISION:-0}"
SKIP_TORCHAUDIO="${LOCAL_DUPLEX_SKIP_TORCHAUDIO:-0}"
USE_SYSTEM_SITE_PACKAGES="${LOCAL_DUPLEX_USE_SYSTEM_SITE_PACKAGES:-0}"
SKIP_TORCH_INSTALL="${LOCAL_DUPLEX_SKIP_TORCH_INSTALL:-0}"
INSTALL_PYTORCH_BACKEND_DEPS="${LOCAL_DUPLEX_INSTALL_PYTORCH_BACKEND_DEPS:-1}"

if [ "${ARCH}" = "aarch64" ]; then
  if system_torch_has_cuda; then
    PYTHON_VERSION="${LOCAL_DUPLEX_PYTHON_VERSION:-$(system_python_version)}"
    USE_SYSTEM_SITE_PACKAGES="${LOCAL_DUPLEX_USE_SYSTEM_SITE_PACKAGES:-1}"
    SKIP_TORCH_INSTALL="${LOCAL_DUPLEX_SKIP_TORCH_INSTALL:-1}"
    SKIP_TORCHVISION="${LOCAL_DUPLEX_SKIP_TORCHVISION:-1}"
    SKIP_TORCHAUDIO="${LOCAL_DUPLEX_SKIP_TORCHAUDIO:-1}"
    INSTALL_PYTORCH_BACKEND_DEPS="${LOCAL_DUPLEX_INSTALL_PYTORCH_BACKEND_DEPS:-0}"
    TORCH_WHEEL=""
  elif [ -z "${TORCH_WHEEL}" ]; then
    TORCH_WHEEL="$(find_torch_wheel 3.12 || true)"
    if [ -n "${TORCH_WHEEL}" ]; then
      PYTHON_VERSION="${LOCAL_DUPLEX_PYTHON_VERSION:-3.12}"
      SKIP_TORCHVISION="${LOCAL_DUPLEX_SKIP_TORCHVISION:-1}"
      SKIP_TORCHAUDIO="${LOCAL_DUPLEX_SKIP_TORCHAUDIO:-1}"
      INSTALL_PYTORCH_BACKEND_DEPS="${LOCAL_DUPLEX_INSTALL_PYTORCH_BACKEND_DEPS:-0}"
    fi
  fi
fi

echo "Spark local duplex environment"
echo "  arch: ${ARCH}"
echo "  camera: ${CAMERA_DEVICE:-<not found>}"
echo "  capswriter: ${CAPSWRITER_ROOT:-<not found>}"
echo "  python: ${PYTHON_VERSION}"
echo "  torch wheel: ${TORCH_WHEEL:-<none>}"
echo
echo "Suggested exports:"
[ -n "${CAPSWRITER_ROOT}" ] && echo "export LOCAL_DUPLEX_CAPSWRITER_ROOT='${CAPSWRITER_ROOT}'"
[ -n "${CAMERA_DEVICE}" ] && echo "export LOCAL_DUPLEX_CAMERA_DEVICE='${CAMERA_DEVICE}'"
echo "export LOCAL_DUPLEX_PYTHON_VERSION='${PYTHON_VERSION}'"
[ "${USE_SYSTEM_SITE_PACKAGES}" = "1" ] && echo "export LOCAL_DUPLEX_USE_SYSTEM_SITE_PACKAGES='1'"
[ "${SKIP_TORCH_INSTALL}" = "1" ] && echo "export LOCAL_DUPLEX_SKIP_TORCH_INSTALL='1'"
[ -n "${TORCH_WHEEL}" ] && echo "export LOCAL_DUPLEX_TORCH_WHEEL='${TORCH_WHEEL}'"
echo "export LOCAL_DUPLEX_SKIP_TORCHVISION='${SKIP_TORCHVISION}'"
echo "export LOCAL_DUPLEX_SKIP_TORCHAUDIO='${SKIP_TORCHAUDIO}'"
echo "export LOCAL_DUPLEX_INSTALL_PYTORCH_BACKEND_DEPS='${INSTALL_PYTORCH_BACKEND_DEPS}'"

export LOCAL_DUPLEX_CAPSWRITER_ROOT="${CAPSWRITER_ROOT:-${LOCAL_DUPLEX_CAPSWRITER_ROOT:-}}"
export LOCAL_DUPLEX_CAMERA_DEVICE="${CAMERA_DEVICE:-${LOCAL_DUPLEX_CAMERA_DEVICE:-}}"
export LOCAL_DUPLEX_PYTHON_VERSION="${PYTHON_VERSION}"
export LOCAL_DUPLEX_USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES}"
export LOCAL_DUPLEX_SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL}"
export LOCAL_DUPLEX_TORCH_WHEEL="${TORCH_WHEEL}"
export LOCAL_DUPLEX_SKIP_TORCHVISION="${SKIP_TORCHVISION}"
export LOCAL_DUPLEX_SKIP_TORCHAUDIO="${SKIP_TORCHAUDIO}"
export LOCAL_DUPLEX_INSTALL_PYTORCH_BACKEND_DEPS="${INSTALL_PYTORCH_BACKEND_DEPS}"

if [ "${RUN_AUDIO_FIX}" = "1" ]; then
  ensure_pipewire_min_volume "@DEFAULT_AUDIO_SINK@" "default sink" "${LOCAL_DUPLEX_SPARK_SINK_VOLUME:-1.0}"
  ensure_pipewire_min_volume "@DEFAULT_AUDIO_SOURCE@" "default source" "${LOCAL_DUPLEX_SPARK_SOURCE_VOLUME:-1.0}"
fi

if [ "${RUN_SETUP}" = "1" ]; then
  bash "${ROOT_DIR}/scripts/setup_duplex_env.sh"
fi

if [ "${RUN_BUILD}" = "1" ]; then
  bash "${ROOT_DIR}/scripts/build_local_duplex_gguf_backend.sh"
fi

if [ "${RUN_PREPARE}" = "1" ]; then
  bash "${ROOT_DIR}/scripts/prepare_local_duplex_gguf.sh"
fi
