"""V4L2 video capture worker for the local duplex runtime."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time

import cv2
from PIL import Image

from local_duplex.config import VideoConfig


class VideoWorker:
    """Continuously captures the latest camera frame."""

    def __init__(self, config: VideoConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ts = 0.0
        self._preview_enabled = config.preview and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    def start(self) -> None:
        camera_path = self.config.camera_device
        if not Path(camera_path).exists():
            raise FileNotFoundError(f"Camera device not found: {camera_path}")
        capture = cv2.VideoCapture(camera_path, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open camera device: {camera_path}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.capture_width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.capture_height))
        capture.set(cv2.CAP_PROP_FPS, float(self.config.capture_fps))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        self._thread = threading.Thread(target=self._reader_loop, name="brio-video", daemon=True)
        self._thread.start()

    def latest_pil(self) -> Image.Image | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            rgb = cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._capture:
            self._capture.release()
        if self._preview_enabled:
            cv2.destroyAllWindows()

    def _reader_loop(self) -> None:
        assert self._capture is not None
        window_name = "MiniCPM-o Local Omni Preview"
        if self._preview_enabled:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.02)
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_ts = time.time()
            if self._preview_enabled:
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

