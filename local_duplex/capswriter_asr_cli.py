"""Run CapsWriter ASR on a WAV file and print JSON to stdout."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import os
import sys


CAPSWRITER_ROOT = Path("/home/ivan/github/CapsWriter-Offline-Windows-64bit")
HTTP_API_SERVER = CAPSWRITER_ROOT / "http_api_server.py"


def _load_capswriter_module():
    os.chdir(CAPSWRITER_ROOT)
    sys.path.insert(0, str(CAPSWRITER_ROOT))
    spec = importlib.util.spec_from_file_location("cw_http_api_server", HTTP_API_SERVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CapsWriter module from {HTTP_API_SERVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe a WAV file with CapsWriter ASR")
    parser.add_argument("wav_path", help="Path to a WAV file")
    args = parser.parse_args()

    wav_path = Path(args.wav_path).resolve()
    if not wav_path.is_file():
        raise SystemExit(f"WAV file not found: {wav_path}")

    module = _load_capswriter_module()
    module.init_recognizer_sync()
    result = module.transcribe_wav_bytes(wav_path.read_bytes())
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
