#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
STAMP="$(date +%Y%m%d-%H%M%S)"
BUNDLE_NAME="MiniCPM-o-local-duplex-${STAMP}"
BUNDLE_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.tar.gz"
LOCAL_HF_HUB="${ROOT_DIR}/third_party/models/huggingface/hub"

if [ ! -x "${ROOT_DIR}/.venv/local-duplex311/bin/python" ]; then
  echo "Missing local venv: ${ROOT_DIR}/.venv/local-duplex311" >&2
  exit 1
fi

if [ ! -d "${LOCAL_HF_HUB}/models--openbmb--MiniCPM-o-4_5-awq" ]; then
  echo "Missing localized AWQ model cache under ${LOCAL_HF_HUB}" >&2
  echo "Run ./scripts/localize_models.sh first." >&2
  exit 1
fi

if [ ! -d "${LOCAL_HF_HUB}/models--openbmb--MiniCPM-o-4_5" ]; then
  echo "Missing localized base MiniCPM-o cache under ${LOCAL_HF_HUB}" >&2
  echo "Run ./scripts/localize_models.sh first." >&2
  exit 1
fi

rm -rf "${BUNDLE_DIR}"
mkdir -p "${BUNDLE_DIR}"

copy_path() {
  local rel="$1"
  local src="${ROOT_DIR}/${rel}"
  local dst="${BUNDLE_DIR}/${rel}"
  if [ -d "${src}" ]; then
    mkdir -p "${dst}"
    rsync -a "${src}/" "${dst}/"
  else
    mkdir -p "$(dirname "${dst}")"
    rsync -a "${src}" "${dst}"
  fi
}

copy_path ".gitignore"
copy_path "README.md"
copy_path "configs"
copy_path "docs/local_duplex_ubuntu24.md"
copy_path "docs/local_duplex_full_summary_zh.md"
copy_path "local_duplex"
copy_path "scripts"
copy_path "third_party/MiniCPM-o-Demo"
copy_path "third_party/models/huggingface"
copy_path ".venv/local-duplex311"

cat > "${BUNDLE_DIR}/RUN_ME_FIRST.txt" <<EOF
1. cd ${BUNDLE_NAME}
2. ./scripts/run_local_duplex.sh omni
3. ./scripts/logs_local_duplex.sh

This bundle already includes:
- Python virtual environment
- localized Hugging Face model cache
- vendored MiniCPM-o-Demo runtime
EOF

mkdir -p "${DIST_DIR}"
tar -C "${DIST_DIR}" -czf "${ARCHIVE_PATH}" "${BUNDLE_NAME}"

echo "Created bundle:"
echo "  ${ARCHIVE_PATH}"
