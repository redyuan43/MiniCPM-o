#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_HUB="${HOME}/.cache/huggingface/hub"
DST_HUB="${ROOT_DIR}/third_party/models/huggingface/hub"

copy_repo() {
  local repo_dir="$1"
  local src="${SRC_HUB}/${repo_dir}"
  local dst="${DST_HUB}/${repo_dir}"

  if [ ! -d "${src}" ]; then
    echo "Missing source model cache: ${src}" >&2
    exit 1
  fi

  mkdir -p "${DST_HUB}"
  rsync -a "${src}/" "${dst}/"
  echo "Localized ${repo_dir} -> ${dst}"
}

copy_repo "models--openbmb--MiniCPM-o-4_5-awq"
copy_repo "models--openbmb--MiniCPM-o-4_5"

echo
echo "Local Hugging Face cache is ready at:"
echo "  ${DST_HUB}"
echo
echo "Runtime will use it automatically through:"
echo "  HF_HOME=${ROOT_DIR}/third_party/models/huggingface"
