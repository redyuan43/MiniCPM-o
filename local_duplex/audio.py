"""Local audio capture/playback pipeline for the duplex runtime."""

from __future__ import annotations

import base64
from collections import deque
import logging
import math
import queue
import subprocess
import threading
import time
from typing import Deque

import numpy as np
try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - validated in runtime preflight
    sd = None

from local_duplex.config import AudioConfig


FLOAT32_LE = "<f4"
INT32_MAX = np.int32(2147483647)
LOGGER = logging.getLogger("local_duplex.audio")


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


def ensure_sounddevice_available() -> None:
    if sd is None:
        raise RuntimeError(
            "Missing Python dependency 'sounddevice'. "
            "Run scripts/setup_duplex_env.sh to install the local duplex audio backend."
        )


def list_input_devices() -> list[tuple[int, dict]]:
    ensure_sounddevice_available()
    devices = sd.query_devices()
    return [
        (index, device)
        for index, device in enumerate(devices)
        if int(device.get("max_input_channels", 0)) > 0
    ]


def list_output_devices() -> list[tuple[int, dict]]:
    ensure_sounddevice_available()
    devices = sd.query_devices()
    return [
        (index, device)
        for index, device in enumerate(devices)
        if int(device.get("max_output_channels", 0)) > 0
    ]


def _find_pipewire_device_id() -> int | None:
    for index, device in list_input_devices():
        if "pipewire" in str(device.get("name", "")).lower():
            return index
    for index, device in list_input_devices():
        if "default" in str(device.get("name", "")).lower():
            return index
    return None


def _find_pipewire_output_device_id() -> int | None:
    for index, device in list_output_devices():
        if "pipewire" in str(device.get("name", "")).lower():
            return index
    for index, device in list_output_devices():
        if "default" in str(device.get("name", "")).lower():
            return index
    return None


def _match_input_device_by_name(selector: str) -> int | None:
    selector_lower = selector.strip().lower()
    for index, device in list_input_devices():
        if selector_lower == str(device.get("name", "")).strip().lower():
            return index
    for index, device in list_input_devices():
        if selector_lower in str(device.get("name", "")).lower():
            return index
    return None


def _match_output_device_by_name(selector: str) -> int | None:
    selector_lower = selector.strip().lower()
    for index, device in list_output_devices():
        if selector_lower == str(device.get("name", "")).strip().lower():
            return index
    for index, device in list_output_devices():
        if selector_lower in str(device.get("name", "")).lower():
            return index
    return None


def resolve_capture_device(device_selector: str | int | None) -> tuple[int | None, dict, str]:
    ensure_sounddevice_available()
    selector = "" if device_selector is None else str(device_selector).strip()

    if not selector or selector.lower() == "default":
        device_info = sd.query_devices(kind="input")
        return None, device_info, f"default:{device_info['name']}"

    if selector.lower() == "pipewire":
        pipewire_device_id = _find_pipewire_device_id()
        if pipewire_device_id is None:
            device_info = sd.query_devices(kind="input")
            return None, device_info, f"default:{device_info['name']}"
        device_info = sd.query_devices(pipewire_device_id)
        return pipewire_device_id, device_info, f"id={pipewire_device_id}:{device_info['name']}"

    if selector.isdigit():
        device_id = int(selector)
        device_info = sd.query_devices(device_id)
        if int(device_info.get("max_input_channels", 0)) < 1:
            raise RuntimeError(f"Configured capture device id={device_id} has no input channels")
        return device_id, device_info, f"id={device_id}:{device_info['name']}"

    matched_device_id = _match_input_device_by_name(selector)
    if matched_device_id is None:
        available_names = ", ".join(str(device["name"]) for _, device in list_input_devices())
        raise RuntimeError(
            f"Unable to resolve capture device '{selector}'. "
            f"Available input devices: {available_names or 'none'}"
        )
    device_info = sd.query_devices(matched_device_id)
    return matched_device_id, device_info, f"id={matched_device_id}:{device_info['name']}"


def resolve_playback_device(device_selector: str | int | None) -> tuple[int | None, dict, str]:
    ensure_sounddevice_available()
    selector = "" if device_selector is None else str(device_selector).strip()

    if not selector or selector.lower() == "default":
        device_info = sd.query_devices(kind="output")
        return None, device_info, f"default:{device_info['name']}"

    if selector.lower() == "pipewire":
        pipewire_device_id = _find_pipewire_output_device_id()
        if pipewire_device_id is None:
            device_info = sd.query_devices(kind="output")
            return None, device_info, f"default:{device_info['name']}"
        device_info = sd.query_devices(pipewire_device_id)
        return pipewire_device_id, device_info, f"id={pipewire_device_id}:{device_info['name']}"

    if selector.isdigit():
        device_id = int(selector)
        device_info = sd.query_devices(device_id)
        if int(device_info.get("max_output_channels", 0)) < 1:
            raise RuntimeError(f"Configured playback device id={device_id} has no output channels")
        return device_id, device_info, f"id={device_id}:{device_info['name']}"

    matched_device_id = _match_output_device_by_name(selector)
    if matched_device_id is None:
        available_names = ", ".join(str(device["name"]) for _, device in list_output_devices())
        raise RuntimeError(
            f"Unable to resolve playback device '{selector}'. "
            f"Available output devices: {available_names or 'none'}"
        )
    device_info = sd.query_devices(matched_device_id)
    return matched_device_id, device_info, f"id={matched_device_id}:{device_info['name']}"


def validate_capture_device(
    device_selector: str | int | None,
    sample_rate: int,
    channels: int = 1,
    dtype: str = "float32",
) -> tuple[int | None, dict, str]:
    device_id, device_info, device_label = resolve_capture_device(device_selector)
    ensure_sounddevice_available()
    sd.check_input_settings(
        device=device_id,
        samplerate=sample_rate,
        channels=channels,
        dtype=dtype,
    )
    return device_id, device_info, device_label


def validate_playback_device(
    device_selector: str | int | None,
    sample_rate: int,
    channels: int = 2,
    dtype: str = "float32",
) -> tuple[int | None, dict, str]:
    device_id, device_info, device_label = resolve_playback_device(device_selector)
    ensure_sounddevice_available()
    sd.check_output_settings(
        device=device_id,
        samplerate=sample_rate,
        channels=channels,
        dtype=dtype,
    )
    return device_id, device_info, device_label


def prefers_sounddevice_playback(device_selector: str | int | None) -> bool:
    selector = "" if device_selector is None else str(device_selector).strip().lower()
    if not selector or selector in {"default", "pipewire"}:
        return True
    if selector.startswith(("hw:", "plughw:", "sysdefault:")):
        return False
    return True


class SoundDeviceCapture:
    """Continuously reads float32 audio from a PortAudio input stream."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.read_samples = config.input_sample_rate * config.read_window_ms // 1000
        self.chunk_samples = config.input_sample_rate * config.chunk_ms // 1000
        self._buffers: Deque[np.ndarray] = deque()
        self._buffered_samples = 0
        self._hold_samples = 0
        self._interrupt_requested = False
        self._playback_active = False
        self._playback_active_ms = 0.0
        self._playback_remaining_ms = 0.0
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._reader_error: str | None = None
        self._stream = None
        self._device_label = "unresolved"

    def start(self) -> None:
        device_id, device_info, device_label = validate_capture_device(
            self.config.capture_device,
            sample_rate=self.config.input_sample_rate,
            channels=1,
            dtype="float32",
        )
        self._device_label = device_label
        try:
            self._stream = sd.InputStream(
                samplerate=self.config.input_sample_rate,
                blocksize=self.read_samples,
                device=device_id,
                dtype="float32",
                channels=1,
                callback=self._record_callback,
                finished_callback=self._finished_callback,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(f"Failed to open input stream ({device_label}): {exc}") from exc
        LOGGER.info(
            "Using audio input device: %s (channels=%s default_rate=%.1f)",
            device_label,
            int(device_info.get("max_input_channels", 0)),
            float(device_info.get("default_samplerate", 0.0)),
        )

    def set_playback_state(self, active: bool, active_ms: float, remaining_ms: float) -> None:
        with self._lock:
            self._playback_active = active
            self._playback_active_ms = active_ms
            self._playback_remaining_ms = remaining_ms
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
                self._check_stream_state()

            if self._stop.is_set():
                self._check_stream_state()
                if self._reader_error:
                    raise RuntimeError(f"Audio capture stopped: {self._reader_error}")
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
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _record_callback(self, indata, frames, _time_info, status) -> None:
        if self._stop.is_set():
            return
        if status:
            LOGGER.warning("Input stream status on %s: %s", self._device_label, status)

        hold_target = max(1, self.config.input_sample_rate * self.config.interrupt_hold_ms // 1000)
        samples = np.asarray(indata, dtype=np.float32).reshape(frames, -1)
        if samples.size == 0:
            return
        mono = samples[:, 0].copy()
        rms = float(math.sqrt(np.mean(mono * mono))) if mono.size else 0.0
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        with self._lock:
            self._buffers.append(mono)
            self._buffered_samples += mono.size
            playback_interruptible = (
                self._playback_active
                and self._playback_active_ms >= self.config.interrupt_min_playback_ms
                and self._playback_remaining_ms > self.config.interrupt_tail_protect_ms
            )
            if (
                playback_interruptible
                and rms >= self.config.interrupt_rms_threshold
                and peak >= self.config.interrupt_peak_threshold
            ):
                self._hold_samples += mono.size
                if self._hold_samples >= hold_target:
                    self._interrupt_requested = True
            else:
                self._hold_samples = 0
            self._lock.notify_all()

    def _finished_callback(self) -> None:
        if self._stop.is_set():
            return
        self._reader_error = f"input stream finished unexpectedly on {self._device_label}"
        self._stop.set()
        with self._lock:
            self._lock.notify_all()

    def _check_stream_state(self) -> None:
        if self._stream is not None and self._stream.closed and not self._stop.is_set():
            raise RuntimeError(f"input stream closed unexpectedly on {self._device_label}")
        if self._reader_error:
            raise RuntimeError(self._reader_error)


class AplayPlayback:
    """Writes exact-format PCM stereo audio to the configured ALSA sink."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=config.output_queue_max_chunks)
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._timing_lock = threading.Lock()
        self._active_since_monotonic = 0.0
        self._playback_deadline_monotonic = 0.0

    def start(self) -> None:
        self._spawn_proc()
        self._thread = threading.Thread(target=self._writer_loop, name="aplay-playback", daemon=True)
        self._thread.start()

    def _spawn_proc(self) -> None:
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

    def _restart_proc(self, reason: str) -> None:
        LOGGER.warning("Playback sink exited after %s, restarting aplay.", reason)
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                if self._proc.stderr is not None:
                    self._proc.stderr.close()
            except Exception:
                pass
        self._spawn_proc()

    @property
    def active(self) -> bool:
        if not self._active.is_set():
            return False
        with self._timing_lock:
            if time.monotonic() >= self._playback_deadline_monotonic:
                self._active.clear()
                self._active_since_monotonic = 0.0
                return False
            return True

    @property
    def active_duration_ms(self) -> float:
        if not self.active:
            return 0.0
        with self._timing_lock:
            return max(0.0, (time.monotonic() - self._active_since_monotonic) * 1000)

    @property
    def remaining_ms(self) -> float:
        if not self.active:
            return 0.0
        with self._timing_lock:
            return max(0.0, (self._playback_deadline_monotonic - time.monotonic()) * 1000)

    def enqueue_model_audio(self, mono_audio_24k: np.ndarray) -> None:
        stereo = upsample_mono_24k_to_stereo_48k(mono_audio_24k)
        pcm = float_stereo_to_pcm_s32(stereo)
        self._queue.put(pcm, timeout=5)
        duration_s = float(len(mono_audio_24k)) / float(self.config.model_output_sample_rate)
        with self._timing_lock:
            now = time.monotonic()
            if not self._active.is_set():
                self._active_since_monotonic = now
                self._playback_deadline_monotonic = now
            base = max(now, self._playback_deadline_monotonic)
            self._playback_deadline_monotonic = base + duration_s
        self._active.set()

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                self._active.clear()
                with self._timing_lock:
                    self._active_since_monotonic = 0.0
                    self._playback_deadline_monotonic = 0.0
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
                if self.remaining_ms <= 0:
                    self._active.clear()
                    with self._timing_lock:
                        self._active_since_monotonic = 0.0
                        self._playback_deadline_monotonic = 0.0
                self._check_proc_state()
                continue

            try:
                self._check_proc_state()
                self._proc.stdin.write(np.asarray(chunk, dtype=np.int32).tobytes())
                self._proc.stdin.flush()
            except BrokenPipeError as exc:
                raise RuntimeError("aplay pipe closed unexpectedly") from exc

    def _check_proc_state(self) -> None:
        if self._proc and self._proc.poll() is not None:
            stderr = b""
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read()
            message = stderr.decode(errors="ignore").strip()
            if self._stop.is_set():
                return
            if "underrun" in message.lower():
                self._restart_proc("ALSA underrun")
                return
            raise RuntimeError(f"aplay exited unexpectedly: {message}")


class SoundDevicePlayback:
    """Writes float32 stereo audio to a shared PortAudio output stream."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=config.output_queue_max_chunks)
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._device_label = "unresolved"
        self._timing_lock = threading.Lock()
        self._active_since_monotonic = 0.0
        self._playback_deadline_monotonic = 0.0

    def start(self) -> None:
        device_id, device_info, device_label = validate_playback_device(
            self.config.playback_device,
            sample_rate=self.config.playback_sample_rate,
            channels=self.config.playback_channels,
            dtype="float32",
        )
        self._device_label = device_label
        try:
            self._stream = sd.OutputStream(
                samplerate=self.config.playback_sample_rate,
                device=device_id,
                channels=self.config.playback_channels,
                dtype="float32",
                finished_callback=self._finished_callback,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(f"Failed to open output stream ({device_label}): {exc}") from exc
        LOGGER.info(
            "Using audio output device: %s (channels=%s default_rate=%.1f)",
            device_label,
            int(device_info.get("max_output_channels", 0)),
            float(device_info.get("default_samplerate", 0.0)),
        )
        self._thread = threading.Thread(target=self._writer_loop, name="sd-playback", daemon=True)
        self._thread.start()

    @property
    def active(self) -> bool:
        if not self._active.is_set():
            return False
        with self._timing_lock:
            if time.monotonic() >= self._playback_deadline_monotonic:
                self._active.clear()
                self._active_since_monotonic = 0.0
                return False
            return True

    @property
    def active_duration_ms(self) -> float:
        if not self.active:
            return 0.0
        with self._timing_lock:
            return max(0.0, (time.monotonic() - self._active_since_monotonic) * 1000.0)

    @property
    def remaining_ms(self) -> float:
        if not self.active:
            return 0.0
        with self._timing_lock:
            return max(0.0, (self._playback_deadline_monotonic - time.monotonic()) * 1000.0)

    def enqueue_model_audio(self, mono_audio_24k: np.ndarray) -> None:
        stereo = upsample_mono_24k_to_stereo_48k(mono_audio_24k)
        self._queue.put(stereo, timeout=5)
        duration_s = float(len(mono_audio_24k)) / float(self.config.model_output_sample_rate)
        with self._timing_lock:
            now = time.monotonic()
            if not self._active.is_set():
                self._active_since_monotonic = now
                self._playback_deadline_monotonic = now
            base = max(now, self._playback_deadline_monotonic)
            self._playback_deadline_monotonic = base + duration_s
        self._active.set()

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                self._active.clear()
                with self._timing_lock:
                    self._active_since_monotonic = 0.0
                    self._playback_deadline_monotonic = 0.0
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._stream is not None:
            try:
                self._stream.abort(ignore_errors=True)
            except Exception:
                pass
            try:
                self._stream.close(ignore_errors=True)
            except Exception:
                pass
            self._stream = None

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self.remaining_ms <= 0:
                    self._active.clear()
                    with self._timing_lock:
                        self._active_since_monotonic = 0.0
                        self._playback_deadline_monotonic = 0.0
                self._check_stream_state()
                continue

            try:
                self._check_stream_state()
                assert self._stream is not None
                self._stream.write(np.asarray(chunk, dtype=np.float32))
            except Exception as exc:
                raise RuntimeError(f"sounddevice playback failed on {self._device_label}: {exc}") from exc

    def _finished_callback(self) -> None:
        if self._stop.is_set():
            return
        LOGGER.warning("Output stream finished unexpectedly on %s", self._device_label)

    def _check_stream_state(self) -> None:
        if self._stream is None:
            raise RuntimeError("sounddevice output stream is not initialized")
        if self._stream.closed and not self._stop.is_set():
            raise RuntimeError(f"output stream closed unexpectedly on {self._device_label}")


def create_playback_backend(config: AudioConfig):
    if prefers_sounddevice_playback(config.playback_device):
        return SoundDevicePlayback(config)
    return AplayPlayback(config)
