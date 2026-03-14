"""CLI entrypoint for the local MiniCPM-o duplex runtime."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from local_duplex.config import DEFAULT_CONFIG_PATH, load_runtime_config


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local MiniCPM-o duplex runtime")
    parser.add_argument("mode", choices=("audio", "omni"))
    parser.add_argument("--config", default=None, help="Path to a JSON config file")
    parser.add_argument("--model-path", default=None, help="Override model path or repo id")
    parser.add_argument("--capture-device", default=None, help="Override ALSA capture device")
    parser.add_argument("--playback-device", default=None, help="Override ALSA playback device")
    parser.add_argument("--camera-device", default=None, help="Override V4L2 camera device path")
    parser.add_argument("--no-preview", action="store_true", help="Disable local OpenCV preview window")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    config = load_runtime_config(args.config or str(DEFAULT_CONFIG_PATH))
    if args.model_path:
        config.model.model_path = args.model_path
    if args.capture_device:
        config.audio.capture_device = args.capture_device
    if args.playback_device:
        config.audio.playback_device = args.playback_device
    if args.camera_device:
        config.video.camera_device = args.camera_device
    if args.no_preview:
        config.video.preview = False

    runtime_dir = Path(config.runtime.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, config.runtime.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from local_duplex.runtime import LocalDuplexRunner

    runner = LocalDuplexRunner(mode=args.mode, config=config)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
