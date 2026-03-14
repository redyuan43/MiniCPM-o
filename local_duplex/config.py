"""Configuration helpers for the local duplex runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "local_duplex.json"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class ModelConfig:
    model_path: str
    attn_implementation: str
    device: str
    compile: bool
    preload_both_tts: bool


@dataclass(slots=True)
class AudioConfig:
    capture_device: str
    playback_device: str
    input_sample_rate: int
    model_output_sample_rate: int
    playback_sample_rate: int
    playback_channels: int
    read_window_ms: int
    chunk_ms: int
    interrupt_rms_threshold: float
    interrupt_peak_threshold: float
    interrupt_min_playback_ms: int
    interrupt_tail_protect_ms: int
    interrupt_hold_ms: int
    output_queue_max_chunks: int
    reference_audio_path: str


@dataclass(slots=True)
class VideoConfig:
    camera_device: str
    capture_width: int
    capture_height: int
    capture_fps: int
    frame_interval_ms: int
    max_slice_nums: int
    preview: bool


@dataclass(slots=True)
class SessionConfig:
    system_prompt: str
    force_listen_count: int
    max_new_speak_tokens_per_chunk: int
    temperature: float
    top_k: int
    top_p: float
    listen_prob_scale: float
    speech_eager_chunks: int
    speech_eager_listen_prob_scale: float


@dataclass(slots=True)
class RuntimeConfig:
    runtime_dir: str
    log_level: str
    session_max_chunks: int
    enable_speech_listen_reset: bool
    session_reset_after_speech_listens: int
    session_min_chunks_before_speech_reset: int
    speech_detect_rms: float
    speech_activation_chunks: int
    speech_activation_min_rms: float
    speech_activation_grace_after_reset_chunks: int
    consistency_error_reset_threshold: int
    kv_reset_threshold: int
    stuck_listen_speech_chunks: int
    speak_latency_budget_ms: int
    listen_latency_budget_ms: int


@dataclass(slots=True)
class DuplexRuntimeConfig:
    model: ModelConfig
    audio: AudioConfig
    video: VideoConfig
    session: SessionConfig
    runtime: RuntimeConfig


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "model_path": "openbmb/MiniCPM-o-4_5-awq",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "compile": False,
        "preload_both_tts": True,
    },
    "audio": {
        "capture_device": "pipewire",
        "playback_device": "hw:CARD=NVidia,DEV=3",
        "input_sample_rate": 16000,
        "model_output_sample_rate": 24000,
        "playback_sample_rate": 48000,
        "playback_channels": 2,
        "read_window_ms": 20,
        "chunk_ms": 400,
        "interrupt_rms_threshold": 0.05,
        "interrupt_peak_threshold": 0.25,
        "interrupt_min_playback_ms": 600,
        "interrupt_tail_protect_ms": 250,
        "interrupt_hold_ms": 260,
        "output_queue_max_chunks": 32,
        "reference_audio_path": "third_party/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav",
    },
    "video": {
        "camera_device": "/dev/v4l/by-id/usb-046d_Logitech_BRIO_AE03BDC5-video-index0",
        "capture_width": 640,
        "capture_height": 360,
        "capture_fps": 15,
        "frame_interval_ms": 4000,
        "max_slice_nums": 1,
        "preview": False,
    },
    "session": {
        "system_prompt": (
            "You are a bilingual robot assistant. Respond with short spoken answers, "
            "stay interruptible, and mention important scene changes proactively."
        ),
        "force_listen_count": 4,
        "max_new_speak_tokens_per_chunk": 14,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
        "listen_prob_scale": 0.9,
        "speech_eager_chunks": 2,
        "speech_eager_listen_prob_scale": 0.8,
    },
    "runtime": {
        "runtime_dir": ".local_duplex",
        "log_level": "INFO",
        "session_max_chunks": 240,
        "enable_speech_listen_reset": False,
        "session_reset_after_speech_listens": 18,
        "session_min_chunks_before_speech_reset": 18,
        "speech_detect_rms": 0.01,
        "speech_activation_chunks": 2,
        "speech_activation_min_rms": 0.015,
        "speech_activation_grace_after_reset_chunks": 6,
        "consistency_error_reset_threshold": 1,
        "kv_reset_threshold": 2,
        "stuck_listen_speech_chunks": 10,
        "speak_latency_budget_ms": 1100,
        "listen_latency_budget_ms": 250,
    },
}


def _resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def load_runtime_config(path: str | None = None) -> DuplexRuntimeConfig:
    merged = dict(DEFAULT_CONFIG)
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        merged = _deep_merge(merged, raw)

    merged["audio"]["reference_audio_path"] = _resolve_path(merged["audio"]["reference_audio_path"])
    if merged["model"]["model_path"].startswith(("/", ".")):
        merged["model"]["model_path"] = _resolve_path(merged["model"]["model_path"])

    return DuplexRuntimeConfig(
        model=ModelConfig(**merged["model"]),
        audio=AudioConfig(**merged["audio"]),
        video=VideoConfig(**merged["video"]),
        session=SessionConfig(**merged["session"]),
        runtime=RuntimeConfig(**merged["runtime"]),
    )
