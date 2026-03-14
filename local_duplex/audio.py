"""ALSA subprocess audio pipeline for local duplex runtime."""

from __future__ import annotations

import base64
from collections import deque
import math
import queue
import subprocess
import threading
import time
from typing import Deque

import numpy as np

from local_duplex.config import AudioConfig


FLOAT32_LE = "<f4"
INT32_MAX = np.int32(2147483647)


def upsample_mono_24k_to_stereo_48k(audio: np.ndarray) -> np.ndarray:
    """Upsample 24k mono float32 audio to 48k stereo float32 audio."""
    if audio.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    upsampled = np.empty(mono.size * 2, dtype=np.float32)
    upsampled[0::2] = mono
    if mono.size > 1:
        upsampled[1:-1:2] = (mono[:-1] + mono[1:]) * 0.5
    upsampled[-1] = mono[-1]
    return np.repeat(upsampled[:, None], 2, axis=1)


def float_stereo_to_pcm_s32(audio: np.ndarray) -> np.ndarray:
    """Convert normalized stereo float audio into packed PCM S32_LE samples."""
    if audio.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 0.99999994)
    return np.rint(clipped * INT32_MAX).astype(np.int32, copy=False)


def decode_model_audio(audio_base64: str) -> np.ndarray:
    raw = base64.b64decode(audio_base64)
    return np.frombuffer(raw, dtype=np.dtype(FLOAT32_LE)).copy()


class ArecordCapture:
    """Continuously reads raw float32 audio from arecord."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.read_samples = config.input_sample_rate * config.read_window_ms // 1000
        self.read_bytes = self.read_samples * 4
        self.chunk_samples = config.input_sample_rate * config.chunk_ms // 1000
        self._buffers: Deque[np.ndarray] = deque()
        self._buffered_samples = 0
        self._hold_samples = 0
        self._interrupt_requested = False
        self._playback_active = False
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        cmd = [
            "arecord",
            "-D",
            self.config.capture_device,
            "-q",
            "-t",
            "raw",
            "-f",
            "FLOAT_LE",
            "-c",
            "1",
            "-r",
            str(self.config.input_sample_rate),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._thread = threading.Thread(target=self._reader_loop, name="arecord-capture", daemon=True)
        self._thread.start()

    def set_playback_active(self, active: bool) -> None:
        with self._lock:
            self._playback_active = active
            if not active:
                self._hold_samples = 0

    def poll_interrupt(self) -> bool:
        with self._lock:
            pending = self._interrupt_requested
            self._interrupt_requested = False
            return pending

    def read_chunk(self) -> np.ndarray:
        with self._lock:
            while not self._stop.is_set() and self._buffered_samples < self.chunk_samples:
                self._lock.wait(timeout=0.2)
                self._check_proc_state()

            if self._stop.is_set():
                raise RuntimeError("Audio capture stopped")

            out = np.empty(self.chunk_samples, dtype=np.float32)
            filled = 0
            while filled < self.chunk_samples:
                buf = self._buffers[0]
                need = self.chunk_samples - filled
                take = min(need, buf.size)
                out[filled:filled + take] = buf[:take]
                filled += take
                if take == buf.size:
                    self._buffers.popleft()
                else:
                    self._buffers[0] = buf[take:]
                self._buffered_samples -= take
            return out

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=1)

    def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        hold_target = max(1, self.config.input_sample_rate * self.config.interrupt_hold_ms // 1000)
        while not self._stop.is_set():
            raw = self._proc.stdout.read(self.read_bytes)
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=np.dtype(FLOAT32_LE))
            if chunk.size == 0:
                continue
            samples = chunk.copy()
            rms = float(math.sqrt(np.mean(samples * samples))) if samples.size else 0.0
            with self._lock:
                self._buffers.append(samples)
                self._buffered_samples += samples.size
                if self._playback_active and rms >= self.config.interrupt_rms_threshold:
                    self._hold_samples += samples.size
                    if self._hold_samples >= hold_target:
                        self._interrupt_requested = True
                else:
                    self._hold_samples = 0
                self._lock.notify_all()
        self._stop.set()
        with self._lock:
            self._lock.notify_all()

    def _check_proc_state(self) -> None:
        if self._proc and self._proc.poll() is not None:
            stderr = b""
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read()
            raise RuntimeError(f"arecord exited unexpectedly: {stderr.decode(errors='ignore').strip()}")


class AplayPlayback:
    """Writes exact-format PCM stereo audio to the configured ALSA sink."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=config.output_queue_max_chunks)
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        cmd = [
            "aplay",
            "-D",
            self.config.playback_device,
            "-q",
            "-t",
            "raw",
            "-f",
            "S32_LE",
            "-c",
            str(self.config.playback_channels),
            "-r",
            str(self.config.playback_sample_rate),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._thread = threading.Thread(target=self._writer_loop, name="aplay-playback", daemon=True)
        self._thread.start()

    @property
    def active(self) -> bool:
        return self._active.is_set()

    def enqueue_model_audio(self, mono_audio_24k: np.ndarray) -> None:
        stereo = upsample_mono_24k_to_stereo_48k(mono_audio_24k)
        pcm = float_stereo_to_pcm_s32(stereo)
        self._queue.put(pcm, timeout=5)
        self._active.set()

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                self._active.clear()
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def _writer_loop(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        while not self._stop.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._active.clear()
                self._check_proc_state()
                continue

            try:
                self._proc.stdin.write(np.asarray(chunk, dtype=np.int32).tobytes())
                self._proc.stdin.flush()
            except BrokenPipeError as exc:
                raise RuntimeError("aplay pipe closed unexpectedly") from exc
            finally:
                if self._queue.empty():
                    self._active.clear()

    def _check_proc_state(self) -> None:
        if self._proc and self._proc.poll() is not None:
            stderr = b""
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read()
            raise RuntimeError(f"aplay exited unexpectedly: {stderr.decode(errors='ignore').strip()}")
