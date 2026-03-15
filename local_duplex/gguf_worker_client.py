"""Local stdin/stdout client for the GGUF duplex worker."""

from __future__ import annotations

import base64
from collections import deque
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel
import soundfile as sf

from local_duplex.config import DuplexRuntimeConfig


class GgufDuplexGenerateResult(BaseModel):
    """GGUF backend result compatible with the current session logger."""

    is_listen: bool
    text: str = ""
    audio_data: str | None = None
    end_of_turn: bool = False
    backend_end_of_turn: bool = False
    ended_with_listen: bool = False
    stop_reason: str = "empty"
    current_time: int = 0
    cost_llm_ms: float | None = None
    cost_tts_prep_ms: float | None = None
    cost_tts_ms: float | None = None
    cost_token2wav_ms: float | None = None
    cost_all_ms: float | None = None
    worker_decode_ms: float | None = None
    worker_wav_wait_ms: float | None = None
    worker_trailing_wait_ms: float | None = None
    n_tokens: int | None = None
    n_tts_tokens: int | None = None
    server_send_ts: float | None = None


class GgufWorkerClient:
    """Manages the local GGUF worker subprocess and command protocol."""

    def __init__(self, config: DuplexRuntimeConfig, mode: str, logger_name: str) -> None:
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(logger_name)
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._seq = 0
        self._current_chunk_index = 0
        self._io_counter = 0
        session_tag = f"{mode}_{int(time.time() * 1000)}"
        self._worker_runtime_dir = Path(config.runtime.runtime_dir) / "gguf_worker" / session_tag
        self._input_dir = self._worker_runtime_dir / "inputs"
        self._output_dir = self._worker_runtime_dir / "worker_output"
        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pending_audio_path: Path | None = None
        self._pending_frame_path: Path | None = None

    def start(self) -> None:
        cmd = [self.config.model.gguf_worker_bin]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", str(self.config.model.device_index))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._stderr_thread = threading.Thread(target=self._pump_stderr, name="gguf-worker-stderr", daemon=True)
        self._stderr_thread.start()
        payload = {
            "type": "init",
            "mode": self.mode,
            "llm_model_path": self.config.model.gguf_model_path,
            "audio_model_path": self.config.model.gguf_audio_path,
            "vision_model_path": self.config.model.gguf_vision_path,
            "tts_model_path": self.config.model.gguf_tts_path,
            "projector_model_path": self.config.model.gguf_projector_path,
            "token2wav_dir": self.config.model.gguf_token2wav_dir,
            "base_output_dir": str(self._output_dir),
            "ctx_size": self.config.model.gguf_ctx_size,
            "n_gpu_layers": self.config.model.gguf_n_gpu_layers,
            "wav_wait_ms": self.config.model.gguf_wav_wait_ms,
            "wav_idle_stable_ms": self.config.model.gguf_wav_idle_stable_ms,
            "wav_empty_wait_ms": self.config.model.gguf_wav_empty_wait_ms,
            "trailing_wait_ms": self.config.model.gguf_trailing_wait_ms,
            "trailing_idle_stable_ms": self.config.model.gguf_trailing_idle_stable_ms,
            "final_speek_wait_ms": self.config.model.gguf_final_speek_wait_ms,
            "device_index": self.config.model.device_index,
            "max_new_speak_tokens_per_chunk": self.config.session.max_new_speak_tokens_per_chunk,
            "listen_prob_scale": self.config.session.listen_prob_scale,
            "temperature": self.config.session.temperature,
            "top_k": self.config.session.top_k,
            "top_p": self.config.session.top_p,
            "n_predict": max(128, self.config.session.max_new_speak_tokens_per_chunk * 8),
            "language": "zh",
            "use_tts": True,
        }
        self._request(payload)

    def prepare(self, system_prompt_text: str, prompt_wav_path: str) -> None:
        self._current_chunk_index = 0
        self._request(
            {
                "type": "prepare",
                "system_prompt_text": system_prompt_text,
                "prompt_wav_path": prompt_wav_path,
            }
        )

    def prefill(self, audio_waveform, frame_list, max_slice_nums: int) -> None:
        audio_path = self._write_audio_input(np.asarray(audio_waveform, dtype=np.float32))
        frame_path = self._write_frame_input(frame_list[0]) if frame_list else None
        self._pending_audio_path = audio_path
        self._pending_frame_path = frame_path
        self._request(
            {
                "type": "prefill",
                "audio_path": str(audio_path),
                "image_path": str(frame_path) if frame_path is not None else "",
                "max_slice_nums": max_slice_nums,
                "chunk_index": self._current_chunk_index,
            }
        )

    def generate(self, listen_prob_scale_override: float | None = None) -> GgufDuplexGenerateResult:
        response = self._request(
            {
                "type": "generate",
                "chunk_index": self._current_chunk_index,
                "listen_prob_scale": (
                    listen_prob_scale_override
                    if listen_prob_scale_override is not None
                    else self.config.session.listen_prob_scale
                ),
            }
        )
        self._current_chunk_index += 1
        audio_data = None
        audio_paths = response.get("audio_wav_paths") or []
        if audio_paths:
            audio = self._load_worker_audio(audio_paths)
            if audio.size:
                audio_data = base64.b64encode(audio.astype(np.float32, copy=False).tobytes()).decode("utf-8")
        clean_text = self._strip_think_blocks(response.get("text", "") or "")
        return GgufDuplexGenerateResult(
            is_listen=bool(response.get("is_listen", True)),
            text=clean_text,
            audio_data=audio_data,
            end_of_turn=bool(response.get("end_of_turn", False)),
            backend_end_of_turn=bool(response.get("backend_end_of_turn", response.get("end_of_turn", False))),
            ended_with_listen=bool(response.get("ended_with_listen", False)),
            stop_reason=str(response.get("stop_reason", "empty") or "empty"),
            current_time=int(response.get("chunk_index", self._current_chunk_index - 1)),
            cost_all_ms=float(response["cost_all_ms"]) if response.get("cost_all_ms") is not None else None,
            cost_llm_ms=float(response["decode_ms"]) if response.get("decode_ms") is not None else None,
            cost_tts_prep_ms=float(response["wav_wait_ms"]) if response.get("wav_wait_ms") is not None else None,
            cost_token2wav_ms=(
                float(response["trailing_wait_ms"]) if response.get("trailing_wait_ms") is not None else None
            ),
            worker_decode_ms=float(response["decode_ms"]) if response.get("decode_ms") is not None else None,
            worker_wav_wait_ms=(
                float(response["wav_wait_ms"]) if response.get("wav_wait_ms") is not None else None
            ),
            worker_trailing_wait_ms=(
                float(response["trailing_wait_ms"]) if response.get("trailing_wait_ms") is not None else None
            ),
            n_tokens=response.get("n_tokens"),
            n_tts_tokens=response.get("n_tts_tokens"),
            server_send_ts=time.time(),
        )

    def set_break(self) -> None:
        self._request({"type": "break"})

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._request({"type": "shutdown"}, allow_failure=True)
        except RuntimeError:
            pass
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

    def cleanup(self) -> None:
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)

    def _request(self, payload: dict[str, Any], allow_failure: bool = False) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("GGUF worker process is not running")
        self._seq += 1
        payload = dict(payload)
        payload["seq"] = self._seq
        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            if allow_failure:
                return {"ok": False, "seq": self._seq, "error": "worker stdin closed"}
            raise RuntimeError("GGUF worker stdin closed") from exc

        while True:
            line = self._proc.stdout.readline()
            if line == "":
                if allow_failure:
                    return {"ok": False, "seq": self._seq, "error": "worker exited"}
                self._raise_worker_exit()
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"GGUF worker returned invalid JSON: {line}") from exc
            if response.get("seq") != self._seq:
                continue
            if not response.get("ok", False) and not allow_failure:
                error = response.get("error", "unknown worker error")
                detail = response.get("detail")
                if detail:
                    error = f"{error}: {detail}"
                raise RuntimeError(error)
            return response

    def _write_audio_input(self, audio_waveform: np.ndarray) -> Path:
        audio_path = self._input_dir / f"audio_{self._io_counter:08d}.wav"
        self._io_counter += 1
        sf.write(str(audio_path), audio_waveform, self.config.audio.input_sample_rate, subtype="PCM_16")
        return audio_path

    def _write_frame_input(self, frame: Image.Image) -> Path:
        frame_path = self._input_dir / f"frame_{self._io_counter:08d}.jpg"
        self._io_counter += 1
        frame.save(frame_path, format="JPEG", quality=90)
        return frame_path

    def _load_worker_audio(self, wav_paths: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for wav_path in wav_paths:
            audio_np, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
            if sample_rate != self.config.audio.model_output_sample_rate:
                raise RuntimeError(f"Unexpected GGUF worker audio sample rate: {sample_rate}")
            audio_np = np.asarray(audio_np, dtype=np.float32)
            if audio_np.ndim == 2:
                audio_np = audio_np[:, 0]
            chunks.append(audio_np.reshape(-1))
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks)

    def _pump_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            message = line.rstrip()
            if not message:
                continue
            self._stderr_tail.append(message)
            self.logger.info(message)

    def _raise_worker_exit(self) -> None:
        assert self._proc is not None
        return_code = self._proc.poll()
        stderr_tail = "\n".join(self._stderr_tail)
        raise RuntimeError(
            "GGUF worker exited unexpectedly"
            f" (returncode={return_code}). Recent worker logs:\n{stderr_tail}"
        )

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
