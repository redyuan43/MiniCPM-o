"""Unattended local duplex self-test harness."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
import re
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from local_duplex.audio import (
    ensure_sounddevice_available,
    resolve_capture_device,
    resolve_playback_device,
)
from local_duplex.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_runtime_config

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None

try:
    import edge_tts
except ImportError:  # pragma: no cover
    edge_tts = None


RUNTIME_DIR = PROJECT_ROOT / ".local_duplex"
SELFTEST_DIR = RUNTIME_DIR / "selftest"
CAPSWRITER_VENV_PYTHON = Path("/home/ivan/github/CapsWriter-Offline-Windows-64bit/venv-asr/bin/python")
CAPSWRITER_ASR_CLI = PROJECT_ROOT / "local_duplex/capswriter_asr_cli.py"
START_SCRIPT = PROJECT_ROOT / "scripts/start_local_duplex.sh"
STOP_SCRIPT = PROJECT_ROOT / "scripts/stop_local_duplex.sh"
DEFAULT_EDGE_TTS_VOICE = "zh-CN-XiaoxiaoNeural"


@dataclass(slots=True)
class ScenarioStep:
    name: str
    prompt_text: str
    post_wait_s: float
    trigger: str = "immediate"
    trigger_delay_s: float = 0.0
    playback_gain: float = 1.6


@dataclass(slots=True)
class PromptScenario:
    name: str
    mode: str
    audio_only: bool
    steps: list[ScenarioStep]
    config_overrides: dict[str, Any] | None = None


@dataclass(slots=True)
class ScenarioResult:
    scenario: str
    session_dir: str
    mode: str
    audio_only: bool
    prompt_text: str
    prompt_transcript: str
    assistant_text: str
    assistant_transcript: str
    prompt_similarity: float
    assistant_similarity: float
    tail_similarity: float
    speak_latency_ms: float
    turns: int
    assistant_turn_stability: float
    barge_in_count: int
    passed: bool
    steps: list[dict[str, Any]]
    metrics: dict[str, Any]


PROMPTS: list[PromptScenario] = [
    PromptScenario(
        name="audio_short",
        mode="omni",
        audio_only=True,
        steps=[
            ScenarioStep(
                name="greeting",
                prompt_text="你好，请用一句简短的话回答我。",
                post_wait_s=8.0,
            )
        ],
    ),
    PromptScenario(
        name="audio_story",
        mode="omni",
        audio_only=True,
        config_overrides={
            "session": {
                "system_prompt": (
                    "You are a bilingual robot assistant. Only speak after clear user speech. "
                    "When the user asks for a story, tell a complete short story with multiple sentences "
                    "instead of a one-line reply. Keep the story coherent and finish the sentence before stopping."
                ),
                "max_new_speak_tokens_per_chunk": 28,
            },
            "runtime": {
                "assistant_continuation_grace_chunks": 10,
            },
        },
        steps=[
            ScenarioStep(
                name="story",
                prompt_text="请详细讲一个故事，至少讲五句话，不要只说一句。",
                post_wait_s=24.0,
            )
        ],
    ),
    PromptScenario(
        name="audio_story_interrupt",
        mode="omni",
        audio_only=True,
        config_overrides={
            "session": {
                "system_prompt": (
                    "You are a bilingual robot assistant. Only speak after clear user speech. "
                    "When asked for a story, tell a complete short story with multiple sentences. "
                    "If the user clearly interrupts while you are speaking, stop quickly and answer the interruption first."
                ),
                "max_new_speak_tokens_per_chunk": 28,
            },
            "audio": {
                "interrupt_rms_threshold": 0.025,
                "interrupt_peak_threshold": 0.12,
                "interrupt_min_playback_ms": 120,
                "interrupt_hold_ms": 70,
            },
            "runtime": {
                "assistant_continuation_grace_chunks": 10,
                "chunk_barge_in_rms_threshold": 0.02,
                "chunk_barge_in_peak_threshold": 0.12,
                "chunk_barge_in_consecutive_chunks": 1,
            },
        },
        steps=[
            ScenarioStep(
                name="story_start",
                prompt_text="请详细讲一个故事，至少讲五句话，不要只说一句。",
                post_wait_s=1.0,
            ),
            ScenarioStep(
                name="interrupt_question",
                prompt_text="停，先回答我。",
                post_wait_s=8.0,
                trigger="after_assistant_start",
                trigger_delay_s=0.15,
                playback_gain=3.0,
            ),
            ScenarioStep(
                name="follow_up",
                prompt_text="现在请继续，再补充一句话。",
                post_wait_s=8.0,
            ),
        ],
    ),
    PromptScenario(
        name="omni_short",
        mode="omni",
        audio_only=False,
        config_overrides={
            "session": {
                "system_prompt": (
                    "You are a bilingual robot assistant. Only speak after clear user speech. "
                    "When the user asks what you see, use the latest camera frame from the start of the user's question "
                    "and answer briefly but concretely."
                ),
                "max_new_speak_tokens_per_chunk": 24,
            },
        },
        steps=[
            ScenarioStep(
                name="vision_question",
                prompt_text="你现在看到了什么？请简短回答。",
                post_wait_s=12.0,
            )
        ],
    ),
    PromptScenario(
        name="omni_scene_stability",
        mode="omni",
        audio_only=False,
        config_overrides={
            "session": {
                "system_prompt": (
                    "You are a bilingual robot assistant. Only speak after clear user speech. "
                    "When the user asks what you see, answer based on the latest camera frame from the start of the user's question. "
                    "If the scene is unchanged and the same question is asked again, keep the answer consistent."
                ),
                "max_new_speak_tokens_per_chunk": 24,
            },
        },
        steps=[
            ScenarioStep(
                name="vision_question_first",
                prompt_text="你现在看到了什么？请简短回答主要物体。",
                post_wait_s=10.0,
            ),
            ScenarioStep(
                name="vision_question_repeat",
                prompt_text="现在还是同一个画面。你又看到了什么？请简短回答主要物体。",
                post_wait_s=10.0,
            ),
        ],
    ),
]

TUNING_CANDIDATES: list[dict[str, Any]] = [
    {},
    {
        "model": {
            "gguf_wav_wait_ms": 700,
            "gguf_wav_idle_stable_ms": 60,
            "gguf_trailing_wait_ms": 220,
            "gguf_trailing_idle_stable_ms": 70,
        },
        "audio": {
            "interrupt_rms_threshold": 0.035,
            "interrupt_peak_threshold": 0.16,
            "interrupt_min_playback_ms": 250,
            "interrupt_hold_ms": 120,
        },
        "runtime": {
            "playback_start_buffer_ms": 1200,
            "playback_immediate_start_min_ms": 700,
            "assistant_continuation_grace_chunks": 6,
            "chunk_barge_in_rms_threshold": 0.028,
            "chunk_barge_in_peak_threshold": 0.16,
            "chunk_barge_in_consecutive_chunks": 1,
        },
    },
    {
        "model": {
            "gguf_wav_wait_ms": 800,
            "gguf_wav_idle_stable_ms": 70,
            "gguf_trailing_wait_ms": 260,
            "gguf_trailing_idle_stable_ms": 80,
        },
        "audio": {
            "interrupt_rms_threshold": 0.03,
            "interrupt_peak_threshold": 0.14,
            "interrupt_min_playback_ms": 180,
            "interrupt_hold_ms": 90,
        },
        "runtime": {
            "playback_start_buffer_ms": 1000,
            "playback_start_buffer_chunks": 1,
            "playback_immediate_start_min_ms": 600,
            "assistant_continuation_grace_chunks": 8,
            "chunk_barge_in_rms_threshold": 0.024,
            "chunk_barge_in_peak_threshold": 0.14,
            "chunk_barge_in_consecutive_chunks": 1,
        },
    },
    {
        "model": {
            "gguf_wav_wait_ms": 650,
            "gguf_wav_idle_stable_ms": 50,
            "gguf_trailing_wait_ms": 320,
            "gguf_trailing_idle_stable_ms": 90,
        },
        "audio": {
            "interrupt_rms_threshold": 0.028,
            "interrupt_peak_threshold": 0.12,
            "interrupt_min_playback_ms": 120,
            "interrupt_hold_ms": 80,
        },
        "runtime": {
            "playback_start_buffer_ms": 900,
            "playback_start_buffer_chunks": 1,
            "playback_immediate_start_min_ms": 550,
            "assistant_continuation_grace_chunks": 10,
            "chunk_barge_in_rms_threshold": 0.02,
            "chunk_barge_in_peak_threshold": 0.12,
            "chunk_barge_in_consecutive_chunks": 1,
        },
    },
]

SELFTEST_BASE_OVERRIDES: dict[str, Any] = {
    "session": {
        "force_listen_count": 1,
    },
    "runtime": {
        "allow_unsolicited_speak": True,
        "unsolicited_speak_reset_threshold": 999,
        "idle_kv_cleanup_after_ms": 30000,
        "speech_activation_chunks": 1,
        "speech_activation_min_rms": 0.01,
    }
}


class MicRecorder:
    """Capture microphone audio into float32 mono samples."""

    def __init__(self, device_selector: str, sample_rate: int) -> None:
        ensure_sounddevice_available()
        self.device_id, _device_info, self.device_label = resolve_capture_device(device_selector)
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            pass
        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        with self._lock:
            self._chunks.append(mono)

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self.device_id,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=0,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks)


def _run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _flatten_overrides(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _flatten_overrides(child_key, child_value, out)
        return
    out[prefix] = value


def _sync_python_defaults(overrides: dict[str, Any]) -> None:
    config_py = (PROJECT_ROOT / "local_duplex/config.py").read_text(encoding="utf-8")
    flat: dict[str, Any] = {}
    for key, value in overrides.items():
        _flatten_overrides(key, value, flat)
    updated = config_py
    for key, value in flat.items():
        pattern = re.compile(rf'("{re.escape(key)}":\s*)([^,\n]+)')
        replacement = _json_value(value)
        updated, count = pattern.subn(lambda match: f"{match.group(1)}{replacement}", updated, count=1)
        if count == 0:
            raise RuntimeError(f"Unable to sync local_duplex/config.py for key: {key}")
    (PROJECT_ROOT / "local_duplex/config.py").write_text(updated, encoding="utf-8")


def _write_overlay_config(run_dir: Path, overrides: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    merged = _deep_merge(config, SELFTEST_BASE_OVERRIDES)
    merged = _deep_merge(merged, overrides)
    out_path = run_dir / "config.override.json"
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


async def _save_edge_tts_wav(prompt_text: str, wav_path: Path) -> None:
    mp3_path = wav_path.with_suffix(".mp3")
    communicate = edge_tts.Communicate(prompt_text, DEFAULT_EDGE_TTS_VOICE)
    await communicate.save(str(mp3_path))
    try:
        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp3_path),
                "-ar",
                "22050",
                "-ac",
                "1",
                str(wav_path),
            ]
        )
    finally:
        if mp3_path.exists():
            mp3_path.unlink()


def _generate_prompt_wav(prompt_text: str, wav_path: Path) -> float:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if edge_tts is not None and _contains_chinese(prompt_text):
        try:
            asyncio.run(_save_edge_tts_wav(prompt_text, wav_path))
            return _wav_duration_s(wav_path)
        except Exception:
            pass
    _run(
        [
            "espeak-ng",
            "-s",
            "165",
            "-w",
            str(wav_path),
            prompt_text,
        ]
    )
    return _wav_duration_s(wav_path)


def _wav_duration_s(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as fh:
        return fh.getnframes() / float(fh.getframerate())


def _read_wav_float32(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as fh:
        n_channels = fh.getnchannels()
        sample_width = fh.getsampwidth()
        sample_rate = fh.getframerate()
        frames = fh.readframes(fh.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.float32)
    return audio, sample_rate


def _write_wav_float32(wav_path: Path, audio: np.ndarray, sample_rate: int) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 0.9999695)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def _play_wav(wav_path: Path, device_selector: str, gain: float = 1.6) -> None:
    ensure_sounddevice_available()
    device_id, _device_info, _label = resolve_playback_device(device_selector)
    audio, sample_rate = _read_wav_float32(wav_path)
    audio = np.clip(audio * gain, -1.0, 0.9999695)
    sd.play(audio, samplerate=sample_rate, device=device_id, blocking=True)


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\s,.;:!?，。！？；：]+", "", text)
    return text


def _similarity(expected: str, actual: str) -> float:
    lhs = _normalize_text(expected)
    rhs = _normalize_text(actual)
    if not lhs and not rhs:
        return 1.0
    if not lhs or not rhs:
        return 0.0
    return SequenceMatcher(None, lhs, rhs).ratio()


def _tail_similarity(expected: str, actual: str, tail_chars: int = 8) -> float:
    lhs = _normalize_text(expected)[-tail_chars:]
    rhs = _normalize_text(actual)[-tail_chars:]
    if not lhs and not rhs:
        return 1.0
    if not lhs or not rhs:
        return 0.0
    return SequenceMatcher(None, lhs, rhs).ratio()


def _transcribe_wav_with_capswriter(wav_path: Path) -> dict[str, Any]:
    result = _run(
        [
            str(CAPSWRITER_VENV_PYTHON),
            str(CAPSWRITER_ASR_CLI),
            str(wav_path),
        ]
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(
        f"Unable to parse CapsWriter ASR JSON for {wav_path}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _latest_session_dir() -> Path:
    latest_path = (RUNTIME_DIR / "latest_session").read_text(encoding="utf-8").strip()
    return Path(latest_path)


def _stop_runtime() -> None:
    _run([str(STOP_SCRIPT)], check=False)


def _start_runtime(mode: str, *, audio_only: bool, config_path: Path) -> None:
    env = dict(os.environ)
    env["LOCAL_DUPLEX_CONFIG"] = str(config_path)
    cmd = [str(START_SCRIPT), mode]
    if audio_only:
        cmd.append("--audio-only")
    result = _run(cmd, env=env)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)


def _wait_for_runtime_ready(timeout_s: float = 40.0) -> None:
    log_path = RUNTIME_DIR / "runtime.log"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            if any(
                marker in text
                for marker in (
                    "双工会话准备完成",
                    "Starting local",
                    "Interaction session log directory:",
                    "input chunk=",
                )
            ):
                return
        time.sleep(0.5)
    raise RuntimeError("Local duplex runtime did not reach ready state in time")


def _collect_assistant_text(summary_path: Path) -> tuple[str, int, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assistant_turns = [item for item in summary.get("conversation_flow", []) if item.get("kind") == "assistant"]
    assistant_text = " ".join(item.get("text", "").strip() for item in assistant_turns if item.get("text"))
    return assistant_text.strip(), len(assistant_turns), float(summary.get("avg_speak_latency_ms") or 0.0)


def _assistant_turn_stability(summary: dict[str, Any]) -> float:
    assistant_turns = [item.get("text", "").strip() for item in summary.get("conversation_flow", []) if item.get("kind") == "assistant" and item.get("text")]
    if len(assistant_turns) < 2:
        return 0.0
    return _similarity(assistant_turns[0], assistant_turns[1])


def _load_summary(summary_path: Path) -> dict[str, Any]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _scenario_report_path(run_dir: Path, scenario: str) -> Path:
    return run_dir / "reports" / f"{scenario}.json"


def _score_result(
    scenario: PromptScenario,
    session_dir: Path,
    step_artifacts: list[dict[str, Any]],
) -> ScenarioResult:
    summary_path = session_dir / "summary.json"
    merged_replay = session_dir / "merged_replay.wav"
    summary = _load_summary(summary_path)
    assistant_text, turns, speak_latency_ms = _collect_assistant_text(summary_path)
    assistant_turn_stability = _assistant_turn_stability(summary)

    step_results: list[dict[str, Any]] = []
    prompt_scores: list[float] = []
    prompt_transcripts: list[str] = []
    for artifact in step_artifacts:
        mic_asr = _transcribe_wav_with_capswriter(Path(artifact["mic_wav"]))
        prompt_similarity = _similarity(artifact["prompt_text"], mic_asr.get("text", ""))
        prompt_scores.append(prompt_similarity)
        prompt_transcripts.append(mic_asr.get("text", ""))
        step_results.append(
            {
                "name": artifact["name"],
                "prompt_text": artifact["prompt_text"],
                "prompt_transcript": mic_asr.get("text", ""),
                "prompt_similarity": round(prompt_similarity, 4),
                "trigger": artifact["trigger"],
                "assistant_started_before_trigger": artifact["assistant_started_before_trigger"],
            }
        )

    playback_asr = {"text": ""}
    if merged_replay.exists():
        playback_asr = _transcribe_wav_with_capswriter(merged_replay)

    prompt_similarity = sum(prompt_scores) / len(prompt_scores) if prompt_scores else 0.0
    assistant_similarity = (
        _similarity(assistant_text, playback_asr.get("text", ""))
        if assistant_text
        else 0.0
    )
    tail_similarity = (
        _tail_similarity(assistant_text, playback_asr.get("text", ""))
        if assistant_text
        else 0.0
    )
    passed = bool(
        assistant_text
        and assistant_similarity >= 0.55
        and tail_similarity >= 0.55
        and ("interrupt" not in scenario.name or int(summary.get("barge_in_count", 0)) >= 1)
        and ("stability" not in scenario.name or (turns >= 2 and assistant_turn_stability >= 0.35))
    )
    return ScenarioResult(
        scenario=scenario.name,
        session_dir=str(session_dir),
        mode=scenario.mode,
        audio_only=scenario.audio_only,
        prompt_text=" | ".join(step.prompt_text for step in scenario.steps),
        prompt_transcript=" | ".join(prompt_transcripts),
        assistant_text=assistant_text,
        assistant_transcript=playback_asr.get("text", ""),
        prompt_similarity=round(prompt_similarity, 4),
        assistant_similarity=round(assistant_similarity, 4),
        tail_similarity=round(tail_similarity, 4),
        speak_latency_ms=round(speak_latency_ms, 1),
        turns=turns,
        assistant_turn_stability=round(assistant_turn_stability, 4),
        barge_in_count=int(summary.get("barge_in_count", 0)),
        passed=passed,
        steps=step_results,
        metrics={
            "summary": summary,
            "assistant_asr": playback_asr,
        },
    )


def _record_and_play_prompt(
    *,
    prompt_wav: Path,
    mic_wav: Path,
    capture_device: str,
    playback_device: str,
    record_padding_s: float,
    playback_gain: float,
) -> None:
    recorder = MicRecorder(capture_device, sample_rate=16000)
    recorder.start()
    try:
        time.sleep(0.25)
        _play_wav(prompt_wav, playback_device, gain=playback_gain)
        time.sleep(record_padding_s)
    finally:
        audio = recorder.stop()
    _write_wav_float32(mic_wav, audio, 16000)


def _wait_for_session_dir(timeout_s: float = 15.0) -> Path:
    latest_file = RUNTIME_DIR / "latest_session"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if latest_file.exists():
            target = Path(latest_file.read_text(encoding="utf-8").strip())
            if target.exists():
                return target
        time.sleep(0.2)
    raise RuntimeError("Timed out waiting for latest session directory")


def _count_assistant_chunks(session_dir: Path) -> int:
    jsonl_path = session_dir / "interaction.jsonl"
    if not jsonl_path.exists():
        return 0
    count = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "chunk" and event.get("assistant", {}).get("state") == "speak":
            count += 1
    return count


def _wait_for_assistant_start(session_dir: Path, baseline_count: int, timeout_s: float = 25.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _count_assistant_chunks(session_dir) > baseline_count:
            return True
        time.sleep(0.2)
    return False


def _run_single_scenario(
    *,
    run_dir: Path,
    scenario: PromptScenario,
    overrides: dict[str, Any],
    capture_device: str,
    playback_device: str,
) -> ScenarioResult:
    prompt_dir = run_dir / "prompts"
    mic_dir = run_dir / "mic_capture"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    mic_dir.mkdir(parents=True, exist_ok=True)
    step_artifacts: list[dict[str, Any]] = []
    scenario_overrides = _deep_merge(overrides, scenario.config_overrides or {})
    config_path = _write_overlay_config(run_dir, scenario_overrides)

    _stop_runtime()
    try:
        _start_runtime(scenario.mode, audio_only=scenario.audio_only, config_path=config_path)
        _wait_for_runtime_ready()
        session_dir = _wait_for_session_dir()
        assistant_count = _count_assistant_chunks(session_dir)
        for index, step in enumerate(scenario.steps):
            prompt_wav = prompt_dir / f"{scenario.name}_{index:02d}_{step.name}.wav"
            mic_wav = mic_dir / f"{scenario.name}_{index:02d}_{step.name}.wav"
            _generate_prompt_wav(step.prompt_text, prompt_wav)
            assistant_started = False
            if step.trigger == "after_assistant_start":
                assistant_started = _wait_for_assistant_start(session_dir, assistant_count)
                if step.trigger_delay_s > 0:
                    time.sleep(step.trigger_delay_s)
            _record_and_play_prompt(
                prompt_wav=prompt_wav,
                mic_wav=mic_wav,
                capture_device=capture_device,
                playback_device=playback_device,
                record_padding_s=1.2,
                playback_gain=step.playback_gain,
            )
            step_artifacts.append(
                {
                    "name": step.name,
                    "prompt_text": step.prompt_text,
                    "mic_wav": str(mic_wav),
                    "trigger": step.trigger,
                    "assistant_started_before_trigger": assistant_started,
                }
            )
            time.sleep(step.post_wait_s)
            assistant_count = _count_assistant_chunks(session_dir)
    finally:
        _stop_runtime()
        time.sleep(1.0)

    session_dir = _latest_session_dir()
    result = _score_result(scenario=scenario, session_dir=session_dir, step_artifacts=step_artifacts)
    report_path = _scenario_report_path(run_dir, scenario.name)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _candidate_score(results: list[ScenarioResult]) -> tuple[float, float, float, float, float]:
    tail = sum(item.tail_similarity for item in results) / len(results)
    assist = sum(item.assistant_similarity for item in results) / len(results)
    prompt = sum(item.prompt_similarity for item in results) / len(results)
    latency = sum(item.speak_latency_ms for item in results) / len(results)
    interrupt_items = [item for item in results if "interrupt" in item.scenario]
    interrupt_hit = (
        sum(1.0 if item.barge_in_count >= 1 else 0.0 for item in interrupt_items) / len(interrupt_items)
        if interrupt_items
        else 0.0
    )
    interrupt_excess_penalty = (
        -sum(max(0, item.barge_in_count - 4) for item in interrupt_items) / len(interrupt_items)
        if interrupt_items
        else 0.0
    )
    return (interrupt_hit, interrupt_excess_penalty, tail, assist, prompt - (latency / 10000.0))


def _render_final_summary(
    *,
    run_dir: Path,
    selected_overrides: dict[str, Any],
    results: list[ScenarioResult],
) -> Path:
    lines = [
        "# Local Duplex Self-Test Summary",
        "",
        "## Selected Parameters",
        "",
        "```json",
        json.dumps(selected_overrides, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Scenario Results",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"### {result.scenario}",
                f"- mode: `{result.mode}`",
                f"- audio_only: `{result.audio_only}`",
                f"- session: `{result.session_dir}`",
                f"- prompt_similarity: `{result.prompt_similarity}`",
                f"- assistant_similarity: `{result.assistant_similarity}`",
                f"- tail_similarity: `{result.tail_similarity}`",
                f"- speak_latency_ms: `{result.speak_latency_ms}`",
                f"- assistant_turn_stability: `{result.assistant_turn_stability}`",
                f"- barge_in_count: `{result.barge_in_count}`",
                f"- passed: `{result.passed}`",
                f"- assistant_text: `{result.assistant_text}`",
                f"- assistant_transcript: `{result.assistant_transcript}`",
                "",
            ]
        )
        if result.steps:
            lines.extend(["#### Steps", ""])
            for step in result.steps:
                lines.extend(
                    [
                        f"- `{step['name']}` trigger=`{step['trigger']}` prompt_similarity=`{step['prompt_similarity']}` assistant_started_before_trigger=`{step['assistant_started_before_trigger']}`",
                        f"  prompt_text: `{step['prompt_text']}`",
                        f"  prompt_transcript: `{step['prompt_transcript']}`",
                    ]
                )
            lines.append("")
    path = run_dir / "reports" / "final_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _apply_best_overrides(overrides: dict[str, Any]) -> None:
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    merged = _deep_merge(config, overrides)
    DEFAULT_CONFIG_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_python_defaults(overrides)


def _run_search(
    *,
    run_dir: Path,
    capture_device: str,
    playback_device: str,
    scenarios: list[PromptScenario],
    tune: bool,
) -> tuple[dict[str, Any], list[ScenarioResult]]:
    candidates = TUNING_CANDIDATES if tune else [TUNING_CANDIDATES[0]]
    best_overrides = candidates[0]
    best_results: list[ScenarioResult] = []
    best_score: tuple[float, float, float, float] | None = None

    for index, overrides in enumerate(candidates):
        candidate_dir = run_dir / f"candidate_{index:02d}"
        candidate_results: list[ScenarioResult] = []
        selected_scenarios = scenarios if tune else scenarios[:1]
        for scenario in selected_scenarios:
            candidate_results.append(
                _run_single_scenario(
                    run_dir=candidate_dir,
                    scenario=scenario,
                    overrides=overrides,
                    capture_device=capture_device,
                    playback_device=playback_device,
                )
            )
        score = _candidate_score(candidate_results)
        if best_score is None or score > best_score:
            best_score = score
            best_overrides = overrides
            best_results = candidate_results
    return best_overrides, best_results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unattended local duplex self-test")
    parser.add_argument("--tune", action="store_true", help="Run the small parameter search instead of a single baseline test")
    parser.add_argument(
        "--scenario",
        choices=("audio_short", "audio_story", "audio_story_interrupt", "omni_short", "omni_scene_stability", "all"),
        default="all",
        help="Limit self-test to a specific scenario",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    ensure_sounddevice_available()
    base_config = load_runtime_config(str(DEFAULT_CONFIG_PATH))
    selected_scenarios = PROMPTS
    if args.scenario != "all":
        selected_scenarios = [item for item in PROMPTS if item.name == args.scenario]
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = SELFTEST_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_overrides, best_results = _run_search(
        run_dir=run_dir,
        capture_device=base_config.audio.capture_device,
        playback_device=base_config.audio.playback_device,
        scenarios=selected_scenarios,
        tune=args.tune,
    )

    if args.tune and best_overrides:
        _apply_best_overrides(best_overrides)

    final_summary = _render_final_summary(
        run_dir=run_dir,
        selected_overrides=best_overrides,
        results=best_results,
    )
    print(f"Self-test run directory: {run_dir}")
    print(f"Final summary: {final_summary}")
    print(f"Selected overrides: {json.dumps(best_overrides, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
