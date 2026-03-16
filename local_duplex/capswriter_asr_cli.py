"""Run CapsWriter ASR on a WAV file and print JSON to stdout."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import os
import sys

from local_duplex.host_paths import require_capswriter_root


def _load_capswriter_module():
    capswriter_root = require_capswriter_root()
    http_api_server = capswriter_root / "http_api_server.py"
    os.chdir(capswriter_root)
    sys.path.insert(0, str(capswriter_root))
    spec = importlib.util.spec_from_file_location("cw_http_api_server", http_api_server)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CapsWriter module from {http_api_server}")
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
