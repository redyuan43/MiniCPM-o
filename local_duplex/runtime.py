"""Main runtime for local MiniCPM-o duplex sessions."""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
import signal
import subprocess
import threading
import time

import numpy as np
import torch

from local_duplex.audio import AplayPlayback, ArecordCapture, decode_model_audio
from local_duplex.config import DuplexRuntimeConfig
from local_duplex.vendor import import_duplex_core
from local_duplex.video import VideoWorker


LOGGER = logging.getLogger("local_duplex")


class LocalDuplexRunner:
    """Local audio/omni duplex runtime."""

    def __init__(self, mode: str, config: DuplexRuntimeConfig) -> None:
        self.mode = mode
        self.config = config
        self.stop_event = threading.Event()
        self.runtime_dir = Path(config.runtime.runtime_dir)
        self._video_every_n_chunks = max(1, config.video.frame_interval_ms // config.audio.chunk_ms)
        self.capture = ArecordCapture(config.audio)
        self.playback = AplayPlayback(config.audio)
        self.video = VideoWorker(config.video) if mode == "omni" else None
        self._vision_enabled = self.video is not None
        self._saved_debug_frame = False
        self._processor = None
        self._duplex = None
        self._session_chunk_count = 0
        self._speech_listen_streak = 0

    def run(self) -> None:
        self._preflight()
        self._install_signal_handlers()
        self._load_model()
        LOGGER.info("Starting local %s duplex session", self.mode)
        self.capture.start()
        self.playback.start()
        if self.video:
            self.video.start()

        self._reset_session(reason="startup", clear_playback=False)

        chunk_index = 0
        try:
            while not self.stop_event.is_set():
                self.capture.set_playback_active(self.playback.active)
                audio_chunk = self.capture.read_chunk()
                audio_rms = float(math.sqrt(np.mean(audio_chunk * audio_chunk))) if audio_chunk.size else 0.0
                audio_peak = float(np.max(np.abs(audio_chunk))) if audio_chunk.size else 0.0
                speech_detected = audio_rms >= self.config.runtime.speech_detect_rms
                if self.playback.active and self.capture.poll_interrupt():
                    LOGGER.info("Barge-in detected, issuing duplex break")
                    self._duplex.set_break()

                frame_list = None
                if self._vision_enabled and self.video and chunk_index % self._video_every_n_chunks == 0:
                    frame = self.video.latest_pil()
                    if frame is not None:
                        frame_list = [frame]
                        self._save_debug_frame(frame, chunk_index)

                if chunk_index == self.config.session.force_listen_count:
                    LOGGER.info(
                        "Forced-listen protection window finished at chunk=%s; model may start speaking",
                        chunk_index,
                    )

                LOGGER.info(
                    "input chunk=%s audio_rms=%.4f audio_peak=%.4f speech_detected=%s playback_active=%s vision_enabled=%s vision_attempt=%s",
                    chunk_index,
                    audio_rms,
                    audio_peak,
                    speech_detected,
                    self.playback.active,
                    self._vision_enabled,
                    bool(frame_list),
                )

                prefill_start = time.time()
                self._prefill_with_fallback(audio_chunk, frame_list)
                result = self._duplex.generate()
                self._duplex.finalize()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                latency_ms = (time.time() - prefill_start) * 1000

                if result.is_listen:
                    LOGGER.info("listen chunk=%s latency_ms=%.1f", chunk_index, latency_ms)
                else:
                    LOGGER.info(
                        "speak chunk=%s latency_ms=%.1f text=%s",
                        chunk_index,
                        latency_ms,
                        (result.text or "").strip().replace("\n", " "),
                    )
                    if result.audio_data:
                        self.playback.enqueue_model_audio(decode_model_audio(result.audio_data))

                self._after_chunk(result_is_listen=result.is_listen, speech_detected=speech_detected)

                chunk_index += 1
        finally:
            self.stop()

    def stop(self) -> None:
        if self.stop_event.is_set():
            pass
        else:
            self.stop_event.set()

        if self.video:
            self.video.stop()
        self.capture.stop()
        self.playback.stop()

        if self._duplex is not None:
            try:
                self._duplex.stop()
            except Exception:
                LOGGER.exception("Failed to stop duplex session cleanly")
            try:
                self._duplex.cleanup()
            except Exception:
                LOGGER.exception("Failed to cleanup duplex session cleanly")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prefill_with_fallback(self, audio_chunk, frame_list) -> None:
        try:
            self._duplex.prefill(
                audio_waveform=audio_chunk,
                frame_list=frame_list,
                max_slice_nums=self.config.video.max_slice_nums,
            )
        except torch.OutOfMemoryError:
            if not frame_list:
                raise
            self._vision_enabled = False
            LOGGER.warning("Vision prefill OOM, disabling vision and retrying current chunk in audio-only mode")
            torch.cuda.empty_cache()
            self._duplex.prefill(
                audio_waveform=audio_chunk,
                frame_list=None,
                max_slice_nums=self.config.video.max_slice_nums,
            )

    def _load_model(self) -> None:
        UnifiedProcessor, DuplexConfig = import_duplex_core()
        from MiniCPMO45.utils import DuplexWindowConfig

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        processor = UnifiedProcessor(
            model_path=self.config.model.model_path,
            device=self.config.model.device,
            ref_audio_path=self.config.audio.reference_audio_path,
            duplex_config=DuplexConfig(
                chunk_ms=self.config.audio.chunk_ms,
                force_listen_count=self.config.session.force_listen_count,
                max_new_speak_tokens_per_chunk=self.config.session.max_new_speak_tokens_per_chunk,
                temperature=self.config.session.temperature,
                top_k=self.config.session.top_k,
                top_p=self.config.session.top_p,
                listen_prob_scale=self.config.session.listen_prob_scale,
            ),
            preload_both_tts=self.config.model.preload_both_tts,
            compile=self.config.model.compile,
            attn_implementation=self.config.model.attn_implementation,
        )
        self._processor = processor
        processor.ref_audio_path = None
        self._duplex = processor.set_duplex_mode()
        self._duplex.ref_audio_path = None
        self._duplex._model.duplex.decoder.set_window_config(
            DuplexWindowConfig(
                sliding_window_mode="basic",
                basic_window_high_tokens=1800,
                basic_window_low_tokens=1400,
                context_previous_max_tokens=256,
                context_max_units=12,
            )
        )
        self._duplex._model.duplex.decoder.set_window_enabled(True)

    def _after_chunk(self, result_is_listen: bool, speech_detected: bool) -> None:
        self._session_chunk_count += 1
        if result_is_listen and speech_detected:
            self._speech_listen_streak += 1
        elif result_is_listen:
            self._speech_listen_streak = 0
        else:
            self._speech_listen_streak = 0

        if self._speech_listen_streak >= self.config.runtime.session_reset_after_speech_listens:
            self._reset_session(
                reason=(
                    "speech_detected_but_model_kept_listening"
                    f"_streak={self._speech_listen_streak}"
                ),
                clear_playback=True,
            )
            return

        if self._session_chunk_count >= self.config.runtime.session_max_chunks:
            self._reset_session(
                reason=f"session_chunk_budget_reached chunks={self._session_chunk_count}",
                clear_playback=True,
            )

    def _reset_session(self, reason: str, clear_playback: bool) -> None:
        assert self._duplex is not None
        if clear_playback:
            self.playback.clear()
        LOGGER.info("Resetting duplex session: %s", reason)
        self._duplex.prepare(
            system_prompt_text=self.config.session.system_prompt,
            prompt_wav_path=self.config.audio.reference_audio_path,
        )
        self._session_chunk_count = 0
        self._speech_listen_streak = 0

    def _save_debug_frame(self, frame, chunk_index: int) -> None:
        if self._saved_debug_frame:
            return
        debug_frame_path = self.runtime_dir / "debug_last_vision_frame.jpg"
        frame.save(debug_frame_path, format="JPEG", quality=90)
        self._saved_debug_frame = True
        LOGGER.info("Saved debug vision frame at chunk=%s path=%s", chunk_index, debug_frame_path)

    def _preflight(self) -> None:
        for command in ("arecord", "aplay", "ffmpeg"):
            if subprocess.run(["bash", "-lc", f"command -v {command}"], check=False, capture_output=True).returncode != 0:
                raise RuntimeError(f"Missing required command: {command}")

        if self.mode == "omni" and not Path(self.config.video.camera_device).exists():
            raise FileNotFoundError(f"Camera device not found: {self.config.video.camera_device}")

        reference_audio = Path(self.config.audio.reference_audio_path)
        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        if self.config.model.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the current Python environment")

        if self.config.audio.playback_device.startswith("hw:CARD=NVidia,DEV=3"):
            eld_path = Path("/proc/asound/card2/eld#0.4")
            if eld_path.exists():
                eld_text = eld_path.read_text(encoding="utf-8", errors="ignore")
                if "monitor_present\t\t1" not in eld_text:
                    raise RuntimeError("NVIDIA HDMI sink is not active; check the monitor connection")

    def _install_signal_handlers(self) -> None:
        def _handle_signal(signum, _frame):
            LOGGER.info("Received signal %s, stopping session", signum)
            self.stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
