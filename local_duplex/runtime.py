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

from local_duplex.audio import (
    AplayPlayback,
    SoundDeviceCapture,
    decode_model_audio,
    ensure_sounddevice_available,
    validate_capture_device,
)
from local_duplex.config import DuplexRuntimeConfig
from local_duplex.session_logging import InteractionSessionLogger
from local_duplex.vendor import import_duplex_core
from local_duplex.video import VideoWorker


LOGGER = logging.getLogger("local_duplex")


class _RuntimeSignalHandler(logging.Handler):
    """Captures model-side health signals from upstream loggers."""

    def __init__(self, runner: "LocalDuplexRunner") -> None:
        super().__init__(level=logging.INFO)
        self.runner = runner

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "CONSISTENCY ERROR!" in message:
            self.runner._note_consistency_error(message)
        elif "audio_past_key_values length" in message and "exceed" in message:
            self.runner._note_kv_reset(message)


class LocalDuplexRunner:
    """Local audio/omni duplex runtime."""

    def __init__(self, mode: str, config: DuplexRuntimeConfig) -> None:
        self.mode = mode
        self.config = config
        self.stop_event = threading.Event()
        self.runtime_dir = Path(config.runtime.runtime_dir)
        self._video_every_n_chunks = max(1, config.video.frame_interval_ms // config.audio.chunk_ms)
        self.capture = SoundDeviceCapture(config.audio)
        self.playback = AplayPlayback(config.audio)
        self.video = VideoWorker(config.video) if mode == "omni" else None
        self._vision_enabled = self.video is not None
        self._saved_debug_frame = False
        self._processor = None
        self._duplex = None
        self._session_chunk_count = 0
        self._speech_listen_streak = 0
        self._speech_recent_chunks_remaining = 0
        self._speech_activation_streak = 0
        self._stuck_listen_streak = 0
        self._pending_reset_reason: str | None = None
        self._session_logger: InteractionSessionLogger | None = None
        self._session_health = "healthy"
        self._consistency_error_count = 0
        self._kv_reset_count = 0
        self._session_consistency_error_count = 0
        self._session_kv_reset_count = 0
        self._stuck_listen_count = 0
        self._last_model_decision_bias = "default"
        self._active_chunk_index = -1
        self._signal_handler: _RuntimeSignalHandler | None = None
        self._observed_loggers: list[logging.Logger] = []

    def run(self) -> None:
        self._preflight()
        self._install_signal_handlers()
        self._attach_runtime_signal_watchers()
        self._load_model()
        self._session_logger = InteractionSessionLogger(self.runtime_dir, self.mode, self.config)
        LOGGER.info("Starting local %s duplex session", self.mode)
        LOGGER.info("Interaction session log directory: %s", self._session_logger.session_path)
        self.capture.start()
        self.playback.start()
        if self.video:
            self.video.start()

        self._reset_session(reason="startup", clear_playback=False, chunk_index=-1)

        chunk_index = 0
        try:
            while not self.stop_event.is_set():
                playback_active = self.playback.active
                playback_active_ms = self.playback.active_duration_ms
                playback_remaining_ms = self.playback.remaining_ms
                self.capture.set_playback_state(
                    playback_active,
                    playback_active_ms,
                    playback_remaining_ms,
                )
                audio_chunk = self.capture.read_chunk()
                self._active_chunk_index = chunk_index
                audio_rms = float(math.sqrt(np.mean(audio_chunk * audio_chunk))) if audio_chunk.size else 0.0
                audio_peak = float(np.max(np.abs(audio_chunk))) if audio_chunk.size else 0.0
                speech_detected = audio_rms >= self.config.runtime.speech_detect_rms
                speech_activation_window_open = self._update_speech_activation(audio_rms, speech_detected)
                barge_in_detected = playback_active and self.capture.poll_interrupt()
                if barge_in_detected:
                    LOGGER.info("Barge-in detected, issuing duplex break")
                    self._duplex.set_break()
                    if self._session_logger is not None:
                        self._session_logger.record_barge_in(
                            chunk_index=chunk_index,
                            audio_rms=audio_rms,
                            audio_peak=audio_peak,
                            playback_active_ms=playback_active_ms,
                            playback_remaining_ms=playback_remaining_ms,
                        )

                frame = None
                frame_list = None
                if self._vision_enabled and self.video and chunk_index % self._video_every_n_chunks == 0:
                    frame = self.video.latest_pil()
                    if frame is not None:
                        frame_list = [frame]
                        self._save_debug_frame(frame, chunk_index)

                if self._session_chunk_count == self.config.session.force_listen_count:
                    LOGGER.info(
                        "Forced-listen protection window finished at session_chunk=%s global_chunk=%s; model may start speaking",
                        self._session_chunk_count,
                        chunk_index,
                    )

                LOGGER.info(
                    "input chunk=%s audio_rms=%.4f audio_peak=%.4f speech_detected=%s speech_recent=%s speech_activation_window_open=%s playback_active=%s playback_active_ms=%.1f playback_remaining_ms=%.1f vision_enabled=%s vision_attempt=%s session_health=%s",
                    chunk_index,
                    audio_rms,
                    audio_peak,
                    speech_detected,
                    self._speech_recent_chunks_remaining > 0,
                    speech_activation_window_open,
                    playback_active,
                    playback_active_ms,
                    playback_remaining_ms,
                    self._vision_enabled,
                    bool(frame_list),
                    self._session_health,
                )

                chunk_start = time.perf_counter()
                effective_frame_list, prefill_ms = self._prefill_with_fallback(audio_chunk, frame_list)
                result, model_decision_bias = self._generate_with_speech_bias(
                    speech_activation_window_open=speech_activation_window_open,
                    playback_active=playback_active,
                )
                self._duplex.finalize()
                latency_ms = (time.perf_counter() - chunk_start) * 1000

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

                reset_reason = self._after_chunk(
                    chunk_index=chunk_index,
                    result_is_listen=result.is_listen,
                    result_end_of_turn=bool(result.end_of_turn),
                    speech_detected=speech_detected,
                    speech_activation_window_open=speech_activation_window_open,
                )
                if self._session_logger is not None:
                    self._session_logger.record_chunk(
                        chunk_index=chunk_index,
                        audio_rms=audio_rms,
                        audio_peak=audio_peak,
                        speech_detected=speech_detected,
                        speech_recent=self._speech_recent_chunks_remaining > 0,
                        playback_active=playback_active,
                        playback_active_ms=playback_active_ms,
                        playback_remaining_ms=playback_remaining_ms,
                        vision_enabled=self._vision_enabled,
                        vision_used=bool(effective_frame_list),
                        frame=frame if effective_frame_list else None,
                        result=result,
                        prefill_ms=prefill_ms,
                        latency_ms=latency_ms,
                        barge_in_detected=barge_in_detected,
                        reset_reason=reset_reason,
                        session_health=self._session_health,
                        consistency_error_count=self._consistency_error_count,
                        kv_reset_count=self._kv_reset_count,
                        speech_activation_window_open=speech_activation_window_open,
                        model_decision_bias=model_decision_bias,
                    )

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
        self._detach_runtime_signal_watchers()
        if self._session_logger is not None:
            try:
                self._session_logger.finalize()
                LOGGER.info("Interaction session summary written to %s", self._session_logger.markdown_path)
            except Exception:
                LOGGER.exception("Failed to finalize interaction session log")

    def _prefill_with_fallback(self, audio_chunk, frame_list):
        prefill_start = time.perf_counter()
        try:
            self._duplex.prefill(
                audio_waveform=audio_chunk,
                frame_list=frame_list,
                max_slice_nums=self.config.video.max_slice_nums,
            )
            return frame_list, (time.perf_counter() - prefill_start) * 1000
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
            return None, (time.perf_counter() - prefill_start) * 1000

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

    def _after_chunk(
        self,
        chunk_index: int,
        result_is_listen: bool,
        result_end_of_turn: bool,
        speech_detected: bool,
        speech_activation_window_open: bool,
    ) -> str | None:
        self._session_chunk_count += 1

        if self._pending_reset_reason is not None and (result_is_listen or result_end_of_turn):
            reason = self._pending_reset_reason
            self._pending_reset_reason = None
            self._reset_session(
                reason=reason,
                clear_playback=False,
                chunk_index=chunk_index,
            )
            return reason

        speech_reset_enabled = (
            self.config.runtime.enable_speech_listen_reset
            and self.config.runtime.session_reset_after_speech_listens > 0
        )
        speech_reset_window_open = (
            self._session_chunk_count >= self.config.runtime.session_min_chunks_before_speech_reset
        )

        if result_is_listen and speech_detected and speech_reset_window_open:
            self._speech_listen_streak += 1
        elif result_is_listen:
            self._speech_listen_streak = 0
        else:
            self._speech_listen_streak = 0

        if speech_reset_enabled and self._speech_listen_streak >= self.config.runtime.session_reset_after_speech_listens:
            reason = (
                "speech_detected_but_model_kept_listening"
                f"_streak={self._speech_listen_streak}"
            )
            self._reset_session(
                reason=reason,
                clear_playback=True,
                chunk_index=chunk_index,
            )
            return reason

        if result_is_listen and speech_activation_window_open and speech_reset_window_open:
            self._stuck_listen_streak += 1
        elif result_is_listen:
            self._stuck_listen_streak = 0
        else:
            self._stuck_listen_streak = 0

        if (
            self._session_health == "degraded"
            and self._stuck_listen_streak >= self.config.runtime.stuck_listen_speech_chunks
        ):
            self._stuck_listen_count += 1
            reason = (
                "stuck_listen_after_degraded"
                f"_streak={self._stuck_listen_streak}"
                f"_consistency={self._session_consistency_error_count}"
                f"_kv={self._session_kv_reset_count}"
            )
            self._set_session_health("stuck_listen", reason, chunk_index)
            self._reset_session(
                reason=reason,
                clear_playback=False,
                chunk_index=chunk_index,
            )
            return reason

        if self._session_chunk_count >= self.config.runtime.session_max_chunks:
            reason = f"session_chunk_budget_reached chunks={self._session_chunk_count}"
            if result_is_listen or result_end_of_turn:
                self._reset_session(
                    reason=reason,
                    clear_playback=False,
                    chunk_index=chunk_index,
                )
                return reason
            if self._pending_reset_reason != reason:
                self._pending_reset_reason = reason
                LOGGER.info("Deferring session reset until current speaking turn completes: %s", reason)
        return None

    def _reset_session(self, reason: str, clear_playback: bool, chunk_index: int) -> None:
        assert self._duplex is not None
        if clear_playback:
            self.playback.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.info("Resetting duplex session: %s", reason)
        if self._session_logger is not None:
            self._session_logger.record_reset(reason=reason, chunk_index=chunk_index)
        self._duplex.prepare(
            system_prompt_text=self.config.session.system_prompt,
            prompt_wav_path=self.config.audio.reference_audio_path,
        )
        self._session_chunk_count = 0
        self._speech_listen_streak = 0
        self._speech_recent_chunks_remaining = 0
        self._speech_activation_streak = 0
        self._stuck_listen_streak = 0
        self._pending_reset_reason = None
        self._session_consistency_error_count = 0
        self._session_kv_reset_count = 0
        self._set_session_health("healthy", f"reset_complete:{reason}", chunk_index)

    def _generate_with_speech_bias(self, speech_activation_window_open: bool, playback_active: bool):
        assert self._duplex is not None
        original_listen_prob_scale = self._duplex.config.listen_prob_scale
        use_speech_bias = not playback_active and speech_activation_window_open
        model_decision_bias = "default"
        if use_speech_bias:
            self._duplex.config.listen_prob_scale = self.config.session.speech_eager_listen_prob_scale
            model_decision_bias = "speech_activation"
        elif playback_active and speech_activation_window_open:
            model_decision_bias = "suppressed_during_playback"
        self._last_model_decision_bias = model_decision_bias
        try:
            return self._duplex.generate(), model_decision_bias
        finally:
            self._duplex.config.listen_prob_scale = original_listen_prob_scale

    def _save_debug_frame(self, frame, chunk_index: int) -> None:
        if self._saved_debug_frame:
            return
        debug_frame_path = self.runtime_dir / "debug_last_vision_frame.jpg"
        frame.save(debug_frame_path, format="JPEG", quality=90)
        self._saved_debug_frame = True
        LOGGER.info("Saved debug vision frame at chunk=%s path=%s", chunk_index, debug_frame_path)

    def _preflight(self) -> None:
        ensure_sounddevice_available()

        for command in ("aplay", "ffmpeg"):
            if subprocess.run(["bash", "-lc", f"command -v {command}"], check=False, capture_output=True).returncode != 0:
                raise RuntimeError(f"Missing required command: {command}")

        validate_capture_device(
            self.config.audio.capture_device,
            sample_rate=self.config.audio.input_sample_rate,
            channels=1,
            dtype="float32",
        )

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

    def _attach_runtime_signal_watchers(self) -> None:
        if self._signal_handler is not None:
            return
        self._signal_handler = _RuntimeSignalHandler(self)
        self._observed_loggers = [
            logging.getLogger("MiniCPMO45.utils"),
            logging.getLogger("MiniCPMO45.modeling_minicpmo_unified"),
        ]
        for target_logger in self._observed_loggers:
            target_logger.addHandler(self._signal_handler)

    def _detach_runtime_signal_watchers(self) -> None:
        if self._signal_handler is None:
            return
        for target_logger in self._observed_loggers:
            target_logger.removeHandler(self._signal_handler)
        self._observed_loggers = []
        self._signal_handler = None

    def _update_speech_activation(self, audio_rms: float, speech_detected: bool) -> bool:
        speech_activation_signal = (
            speech_detected and audio_rms >= self.config.runtime.speech_activation_min_rms
        )
        if speech_activation_signal:
            self._speech_activation_streak += 1
            self._speech_recent_chunks_remaining = self.config.session.speech_eager_chunks
        else:
            self._speech_activation_streak = 0
            if self._speech_recent_chunks_remaining > 0:
                self._speech_recent_chunks_remaining -= 1

        return (
            self._session_chunk_count >= self.config.runtime.speech_activation_grace_after_reset_chunks
            and (
                self._speech_activation_streak >= self.config.runtime.speech_activation_chunks
                or self._speech_recent_chunks_remaining > 0
            )
        )

    def _set_session_health(self, new_state: str, reason: str, chunk_index: int) -> None:
        if self._session_health == new_state:
            return
        self._session_health = new_state
        log_fn = LOGGER.warning if new_state != "healthy" else LOGGER.info
        log_fn("Session health -> %s: %s", new_state, reason)
        if self._session_logger is not None:
            self._session_logger.record_health_change(
                state=new_state,
                reason=reason,
                chunk_index=chunk_index,
            )

    def _note_consistency_error(self, message: str) -> None:
        self._consistency_error_count += 1
        self._session_consistency_error_count += 1
        LOGGER.warning(
            "Observed consistency error signal total=%s session=%s chunk=%s",
            self._consistency_error_count,
            self._session_consistency_error_count,
            self._active_chunk_index,
        )
        if self._session_consistency_error_count >= self.config.runtime.consistency_error_reset_threshold:
            self._set_session_health(
                "degraded",
                f"consistency_error_count={self._session_consistency_error_count}",
                self._active_chunk_index,
            )

    def _note_kv_reset(self, message: str) -> None:
        self._kv_reset_count += 1
        self._session_kv_reset_count += 1
        LOGGER.warning(
            "Observed KV reset signal total=%s session=%s chunk=%s",
            self._kv_reset_count,
            self._session_kv_reset_count,
            self._active_chunk_index,
        )
        if self._session_kv_reset_count >= self.config.runtime.kv_reset_threshold:
            self._set_session_health(
                "degraded",
                f"kv_reset_count={self._session_kv_reset_count}",
                self._active_chunk_index,
            )
