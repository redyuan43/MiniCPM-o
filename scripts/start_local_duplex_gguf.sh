#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LOCAL_DUPLEX_CONFIG="${ROOT_DIR}/configs/local_duplex_gguf.json"

"${ROOT_DIR}/scripts/start_local_duplex.sh" "${1:-omni}"
