#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv/local-duplex311"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Missing ${VENV_DIR}. Run scripts/setup_duplex_env.sh first." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" - <<'PY'
from local_duplex.audio import list_input_devices

print("=" * 72)
print("本地 Duplex 可用输入设备")
print("=" * 72)
for index, device in list_input_devices():
    print(f"设备 ID: {index}")
    print(f"  名称: {device['name']}")
    print(f"  输入声道数: {device['max_input_channels']}")
    print(f"  默认采样率: {device['default_samplerate']}")
    print()
print("建议：")
print("  - capture_device=pipewire  优先跟随桌面当前输入设备")
print("  - capture_device=default   使用 PortAudio 默认输入设备")
print("  - capture_device=<数字ID>  固定到某个输入设备")
print("  - capture_device=<名称片段> 按名称模糊匹配输入设备")
print("=" * 72)
PY
