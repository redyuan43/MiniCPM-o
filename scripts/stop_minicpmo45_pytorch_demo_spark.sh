#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${ROOT_DIR}/third_party/minicpm-o-4_5-pytorch-simple-demo"
PROTO_MARKER="${DEMO_DIR}/tmp/gateway.proto"

cd "${DEMO_DIR}"

if ls tmp/*.pid >/dev/null 2>&1; then
  kill $(cat tmp/*.pid) 2>/dev/null || true
fi

pkill -f "third_party/minicpm-o-4_5-pytorch-simple-demo/(gateway|worker)\\.py" 2>/dev/null || true
pkill -f "/home/dgx/github/MiniCPM-o/third_party/minicpm-o-4_5-pytorch-simple-demo/start_all.sh" 2>/dev/null || true
rm -f "${PROTO_MARKER}"

echo "Stop signal sent to MiniCPM-o 4.5 PyTorch demo processes."
