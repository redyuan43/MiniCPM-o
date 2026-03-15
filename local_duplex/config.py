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
    backend: str
    model_path: str
    gguf_repo_id: str
    gguf_variant: str
    gguf_model_path: str
    gguf_audio_path: str
    gguf_vision_path: str
    gguf_tts_path: str
    gguf_projector_path: str
    gguf_token2wav_dir: str
    gguf_worker_root: str
    gguf_worker_bin: str
    gguf_ctx_size: int
    gguf_n_gpu_layers: int
    gguf_wav_wait_ms: int
    gguf_wav_idle_stable_ms: int
    gguf_wav_empty_wait_ms: int
    gguf_trailing_wait_ms: int
    gguf_trailing_idle_stable_ms: int
    attn_implementation: str
    device: str
    device_index: int
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
    allow_unsolicited_speak: bool
    unsolicited_speak_reset_threshold: int
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
    playback_start_buffer_ms: int
    playback_start_buffer_chunks: int
    playback_immediate_start_min_ms: int
    assistant_continuation_grace_chunks: int
    idle_kv_cleanup_after_ms: int
    chunk_barge_in_rms_threshold: float
    chunk_barge_in_peak_threshold: float
    chunk_barge_in_consecutive_chunks: int


@dataclass(slots=True)
class DuplexRuntimeConfig:
    model: ModelConfig
    audio: AudioConfig
    video: VideoConfig
    session: SessionConfig
    runtime: RuntimeConfig


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "backend": "gguf",
        "model_path": "openbmb/MiniCPM-o-4_5-awq",
        "gguf_repo_id": "OpenBMB/MiniCPM-o-4_5-gguf",
        "gguf_variant": "Q4_K_M",
        "gguf_model_path": "third_party/models/modelscope/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf",
        "gguf_audio_path": "third_party/models/modelscope/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf",
        "gguf_vision_path": "third_party/models/modelscope/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf",
        "gguf_tts_path": "third_party/models/modelscope/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf",
        "gguf_projector_path": "third_party/models/modelscope/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-projector-F16.gguf",
        "gguf_token2wav_dir": "third_party/models/modelscope/MiniCPM-o-4_5-gguf/token2wav-gguf",
        "gguf_worker_root": "third_party/llama.cpp-omni",
        "gguf_worker_bin": "build/local_duplex_gguf_worker/bin/local-duplex-gguf-worker",
        "gguf_ctx_size": 4096,
        "gguf_n_gpu_layers": -1,
        "gguf_wav_wait_ms": 900,
        "gguf_wav_idle_stable_ms": 80,
        "gguf_wav_empty_wait_ms": 220,
        "gguf_trailing_wait_ms": 150,
        "gguf_trailing_idle_stable_ms": 60,
        "attn_implementation": "sdpa",
        "device": "cuda",
        "device_index": 0,
        "compile": False,
        "preload_both_tts": True,
    },
    "audio": {
        "capture_device": "pipewire",
        "playback_device": "pipewire",
        "input_sample_rate": 16000,
        "model_output_sample_rate": 24000,
        "playback_sample_rate": 48000,
        "playback_channels": 2,
        "read_window_ms": 20,
        "chunk_ms": 1000,
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
        "frame_interval_ms": 1000,
        "max_slice_nums": 1,
        "preview": False,
    },
    "session": {
        "system_prompt": (
            "You are a bilingual robot assistant. Respond with short spoken answers, "
            "stay interruptible, and only speak after clear user speech. "
            "Do not proactively start talking when the user is silent."
        ),
        "force_listen_count": 3,
        "max_new_speak_tokens_per_chunk": 20,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
        "listen_prob_scale": 1.0,
        "speech_eager_chunks": 2,
        "speech_eager_listen_prob_scale": 0.85,
    },
    "runtime": {
        "runtime_dir": ".local_duplex",
        "log_level": "INFO",
        "session_max_chunks": 360,
        "enable_speech_listen_reset": False,
        "allow_unsolicited_speak": False,
        "unsolicited_speak_reset_threshold": 2,
        "session_reset_after_speech_listens": 18,
        "session_min_chunks_before_speech_reset": 18,
        "speech_detect_rms": 0.01,
        "speech_activation_chunks": 2,
        "speech_activation_min_rms": 0.015,
        "speech_activation_grace_after_reset_chunks": 3,
        "consistency_error_reset_threshold": 1,
        "kv_reset_threshold": 2,
        "stuck_listen_speech_chunks": 10,
        "speak_latency_budget_ms": 1000,
        "listen_latency_budget_ms": 250,
        "playback_start_buffer_ms": 1600,
        "playback_start_buffer_chunks": 2,
        "playback_immediate_start_min_ms": 900,
        "assistant_continuation_grace_chunks": 4,
        "idle_kv_cleanup_after_ms": 15000,
        "chunk_barge_in_rms_threshold": 1.1,
        "chunk_barge_in_peak_threshold": 1.1,
        "chunk_barge_in_consecutive_chunks": 999,
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
    for key in (
        "gguf_model_path",
        "gguf_audio_path",
        "gguf_vision_path",
        "gguf_tts_path",
        "gguf_projector_path",
        "gguf_token2wav_dir",
        "gguf_worker_root",
        "gguf_worker_bin",
    ):
        merged["model"][key] = _resolve_path(merged["model"][key])

    return DuplexRuntimeConfig(
        model=ModelConfig(**merged["model"]),
        audio=AudioConfig(**merged["audio"]),
        video=VideoConfig(**merged["video"]),
        session=SessionConfig(**merged["session"]),
        runtime=RuntimeConfig(**merged["runtime"]),
    )
