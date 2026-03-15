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
    SoundDeviceCapture,
    create_playback_backend,
    decode_model_audio,
    ensure_sounddevice_available,
    prefers_sounddevice_playback,
    validate_capture_device,
    validate_playback_device,
)
from local_duplex.backends import create_duplex_backend
from local_duplex.config import DuplexRuntimeConfig
from local_duplex.session_logging import InteractionSessionLogger
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
        self.playback = create_playback_backend(config.audio)
        self.video = VideoWorker(config.video) if mode == "omni" else None
        self._vision_enabled = self.video is not None
        self._saved_debug_frame = False
        self._backend = create_duplex_backend(config, mode)
        self._session_chunk_count = 0
        self._speech_listen_streak = 0
        self._speech_recent_chunks_remaining = 0
        self._speech_activation_streak = 0
        self._stuck_listen_streak = 0
        self._assistant_turn_open = False
        self._unsolicited_speak_suppressed_count = 0
        self._session_unsolicited_speak_suppressed_count = 0
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
        self._pending_assistant_audio: list[np.ndarray] = []
        self._pending_assistant_audio_duration_ms = 0.0
        self._pending_assistant_chunks = 0
        self._assistant_playback_started = False
        self._assistant_continuation_grace_remaining = 0
        self._last_interaction_monotonic = time.monotonic()
        self._chunk_barge_in_streak = 0
        self._signal_handler: _RuntimeSignalHandler | None = None
        self._observed_loggers: list[logging.Logger] = []

    def run(self) -> None:
        self._preflight()
        self._install_signal_handlers()
        self._attach_runtime_signal_watchers()
        self._backend.load()
        self._session_logger = InteractionSessionLogger(self.runtime_dir, self.mode, self.config)
        LOGGER.info("Starting local %s duplex session", self.mode)
        LOGGER.info("Using duplex backend: %s", self._backend.backend_name)
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
                barge_in_detected = playback_active and self.capture.poll_interrupt()
                chunk_barge_in_detected = self._update_chunk_barge_in(
                    playback_active=playback_active,
                    playback_active_ms=playback_active_ms,
                    playback_remaining_ms=playback_remaining_ms,
                    audio_rms=audio_rms,
                    audio_peak=audio_peak,
                )
                barge_in_detected = barge_in_detected or chunk_barge_in_detected
                speech_activation_window_open = self._update_speech_activation(
                    audio_rms=audio_rms,
                    speech_detected=speech_detected,
                    playback_active=playback_active,
                    barge_in_detected=barge_in_detected,
                )
                if barge_in_detected:
                    LOGGER.info("Barge-in detected, issuing duplex break")
                    self._backend.set_break()
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
                if self._should_send_vision(
                    chunk_index=chunk_index,
                    playback_active=playback_active,
                    speech_detected=speech_detected,
                    speech_activation_window_open=speech_activation_window_open,
                ):
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
                try:
                    effective_frame_list, prefill_ms = self._prefill_with_fallback(audio_chunk, frame_list)
                    result, model_decision_bias = self._generate_with_speech_bias(
                        speech_activation_window_open=speech_activation_window_open,
                        playback_active=playback_active,
                    )
                    result = self._apply_force_listen_guard(result)
                    result, unsolicited_speak_suppressed = self._apply_unsolicited_speak_guard(
                        result=result,
                        speech_detected=speech_detected,
                        speech_activation_window_open=speech_activation_window_open,
                        playback_active=playback_active,
                    )
                    if unsolicited_speak_suppressed:
                        model_decision_bias = self._last_model_decision_bias
                    self._backend.finalize()
                except Exception as exc:
                    if self.stop_event.is_set() and self._is_shutdown_interruption(exc):
                        LOGGER.info("Backend interrupted during shutdown; ending session loop cleanly")
                        break
                    raise
                latency_ms = (time.perf_counter() - chunk_start) * 1000

                if result.is_listen:
                    if result.audio_data:
                        self._handle_assistant_audio(
                            decode_model_audio(result.audio_data),
                            end_of_turn=bool(result.end_of_turn),
                        )
                    self._flush_pending_assistant_audio(force=True)
                    LOGGER.info("listen chunk=%s latency_ms=%.1f", chunk_index, latency_ms)
                else:
                    self._last_interaction_monotonic = time.monotonic()
                    LOGGER.info(
                        "speak chunk=%s latency_ms=%.1f text=%s",
                        chunk_index,
                        latency_ms,
                        (result.text or "").strip().replace("\n", " "),
                    )
                    if result.audio_data:
                        self._handle_assistant_audio(
                            decode_model_audio(result.audio_data),
                            end_of_turn=bool(result.end_of_turn),
                        )
                if speech_detected or playback_active:
                    self._last_interaction_monotonic = time.monotonic()
                self._update_assistant_turn_state(result)

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
                        unsolicited_speak_suppressed=unsolicited_speak_suppressed,
                        unsolicited_speak_suppressed_count=self._session_unsolicited_speak_suppressed_count,
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
        self._flush_pending_assistant_audio(force=True)
        self.playback.stop()

        if self._backend is not None:
            try:
                self._backend.stop()
            except Exception:
                LOGGER.exception("Failed to stop duplex backend cleanly")
            try:
                self._backend.cleanup()
            except Exception:
                LOGGER.exception("Failed to cleanup duplex backend cleanly")
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
            self._backend.prefill(
                audio_waveform=audio_chunk,
                frame_list=frame_list,
                max_slice_nums=self.config.video.max_slice_nums,
            )
            return frame_list, (time.perf_counter() - prefill_start) * 1000
        except Exception as exc:
            if not self._backend.is_vision_oom(exc):
                raise
            if not frame_list:
                raise
            self._vision_enabled = False
            LOGGER.warning("Vision prefill OOM, disabling vision and retrying current chunk in audio-only mode")
            torch.cuda.empty_cache()
            self._backend.prefill(
                audio_waveform=audio_chunk,
                frame_list=None,
                max_slice_nums=self.config.video.max_slice_nums,
            )
            return None, (time.perf_counter() - prefill_start) * 1000

    def _after_chunk(
        self,
        chunk_index: int,
        result_is_listen: bool,
        result_end_of_turn: bool,
        speech_detected: bool,
        speech_activation_window_open: bool,
    ) -> str | None:
        self._session_chunk_count += 1

        idle_ms = (time.monotonic() - self._last_interaction_monotonic) * 1000.0
        if (
            self.config.runtime.idle_kv_cleanup_after_ms > 0
            and self._session_chunk_count > 0
            and result_is_listen
            and not speech_detected
            and not self.playback.active
            and not self._pending_assistant_audio
            and idle_ms >= self.config.runtime.idle_kv_cleanup_after_ms
        ):
            reason = f"idle_kv_cleanup idle_ms={int(idle_ms)}"
            self._reset_session(
                reason=reason,
                clear_playback=False,
                chunk_index=chunk_index,
            )
            return reason

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
        if clear_playback:
            self.playback.clear()
        self._clear_pending_assistant_audio()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.info("Resetting duplex session: %s", reason)
        if self._session_logger is not None:
            self._session_logger.record_reset(reason=reason, chunk_index=chunk_index)
        self._backend.prepare(
            system_prompt_text=self.config.session.system_prompt,
            prompt_wav_path=self.config.audio.reference_audio_path,
        )
        self._session_chunk_count = 0
        self._speech_listen_streak = 0
        self._speech_recent_chunks_remaining = 0
        self._speech_activation_streak = 0
        self._stuck_listen_streak = 0
        self._assistant_turn_open = False
        self._unsolicited_speak_suppressed_count = 0
        self._pending_reset_reason = None
        self._session_consistency_error_count = 0
        self._session_kv_reset_count = 0
        self._assistant_playback_started = False
        self._assistant_continuation_grace_remaining = 0
        self._last_interaction_monotonic = time.monotonic()
        self._chunk_barge_in_streak = 0
        self._set_session_health("healthy", f"reset_complete:{reason}", chunk_index)

    def _generate_with_speech_bias(self, speech_activation_window_open: bool, playback_active: bool):
        force_listen_active = self._session_chunk_count < self.config.session.force_listen_count
        use_speech_bias = not force_listen_active and not playback_active and speech_activation_window_open
        model_decision_bias = "default"
        listen_prob_scale_override: float | None = None
        if force_listen_active:
            listen_prob_scale_override = max(self.config.session.listen_prob_scale, 2.0)
            model_decision_bias = "forced_listen"
        elif use_speech_bias:
            listen_prob_scale_override = self.config.session.speech_eager_listen_prob_scale
            model_decision_bias = "speech_activation"
        elif playback_active and speech_activation_window_open:
            model_decision_bias = "suppressed_during_playback"
        self._last_model_decision_bias = model_decision_bias
        return self._backend.generate(listen_prob_scale_override), model_decision_bias

    def _apply_force_listen_guard(self, result):
        if self._session_chunk_count >= self.config.session.force_listen_count or result.is_listen:
            return result
        LOGGER.info(
            "Suppressing assistant output during forced-listen window at session_chunk=%s",
            self._session_chunk_count,
        )
        result.is_listen = True
        result.text = ""
        result.audio_data = None
        return result

    def _apply_unsolicited_speak_guard(
        self,
        *,
        result,
        speech_detected: bool,
        speech_activation_window_open: bool,
        playback_active: bool,
    ):
        if result.is_listen or self.config.runtime.allow_unsolicited_speak:
            if speech_detected or speech_activation_window_open or playback_active:
                self._unsolicited_speak_suppressed_count = 0
            return result, False

        has_recent_user_input = (
            speech_detected
            or speech_activation_window_open
            or self._speech_recent_chunks_remaining > 0
        )
        if (
            has_recent_user_input
            or playback_active
            or self._assistant_turn_open
            or self._assistant_continuation_grace_remaining > 0
        ):
            self._unsolicited_speak_suppressed_count = 0
            return result, False

        self._unsolicited_speak_suppressed_count += 1
        self._session_unsolicited_speak_suppressed_count += 1
        preview = (result.text or "").strip().replace("\n", " ")
        LOGGER.warning(
            "Suppressing unsolicited assistant output without recent user speech at session_chunk=%s count=%s text=%s",
            self._session_chunk_count,
            self._unsolicited_speak_suppressed_count,
            preview,
        )
        if (
            self._unsolicited_speak_suppressed_count
            >= self.config.runtime.unsolicited_speak_reset_threshold
            and self._pending_reset_reason is None
        ):
            self._pending_reset_reason = (
                "unsolicited_speak_without_user_input"
                f"_count={self._unsolicited_speak_suppressed_count}"
            )
            LOGGER.warning("Scheduling safe session reset after repeated unsolicited speech")
        result.is_listen = True
        result.text = ""
        result.audio_data = None
        result.end_of_turn = True
        self._last_model_decision_bias = "unsolicited_speak_suppressed"
        return result, True

    def _update_assistant_turn_state(self, result) -> None:
        if result.is_listen:
            if not self.playback.active:
                self._assistant_turn_open = False
            if self._assistant_continuation_grace_remaining > 0:
                self._assistant_continuation_grace_remaining -= 1
            return
        self._unsolicited_speak_suppressed_count = 0
        self._assistant_continuation_grace_remaining = self.config.runtime.assistant_continuation_grace_chunks
        self._assistant_turn_open = not bool(result.end_of_turn)

    @staticmethod
    def _is_shutdown_interruption(exc: Exception) -> bool:
        message = str(exc).lower()
        return "broken pipe" in message or "worker stdin closed" in message or "worker exited unexpectedly" in message

    def _save_debug_frame(self, frame, chunk_index: int) -> None:
        if self._saved_debug_frame:
            return
        debug_frame_path = self.runtime_dir / "debug_last_vision_frame.jpg"
        frame.save(debug_frame_path, format="JPEG", quality=90)
        self._saved_debug_frame = True
        LOGGER.info("Saved debug vision frame at chunk=%s path=%s", chunk_index, debug_frame_path)

    def _handle_assistant_audio(self, audio_chunk: np.ndarray, *, end_of_turn: bool) -> None:
        duration_ms = len(audio_chunk) / self.config.audio.model_output_sample_rate * 1000.0
        if self._assistant_playback_started:
            self.playback.enqueue_model_audio(audio_chunk)
            return

        self._pending_assistant_audio.append(audio_chunk)
        self._pending_assistant_audio_duration_ms += duration_ms
        self._pending_assistant_chunks += 1
        should_start_immediately = (
            end_of_turn
            or duration_ms >= self.config.runtime.playback_immediate_start_min_ms
        )
        if not self._assistant_playback_started:
            if (
                should_start_immediately
                or
                self._pending_assistant_chunks >= self.config.runtime.playback_start_buffer_chunks
                or self._pending_assistant_audio_duration_ms >= self.config.runtime.playback_start_buffer_ms
            ):
                merged = (
                    np.concatenate(self._pending_assistant_audio)
                    if self._pending_assistant_audio
                    else np.zeros((0,), dtype=np.float32)
                )
                if merged.size:
                    self.playback.enqueue_model_audio(merged)
                self._assistant_playback_started = True
                self._clear_pending_assistant_audio()
            return

    def _flush_pending_assistant_audio(self, force: bool) -> None:
        if not self._pending_assistant_audio:
            if force:
                self._assistant_playback_started = False
            return
        merged = np.concatenate(self._pending_assistant_audio)
        if merged.size:
            self.playback.enqueue_model_audio(merged)
        self._assistant_playback_started = bool(not force or self.playback.active)
        self._clear_pending_assistant_audio()
        if force:
            self._assistant_playback_started = False

    def _clear_pending_assistant_audio(self) -> None:
        self._pending_assistant_audio.clear()
        self._pending_assistant_audio_duration_ms = 0.0
        self._pending_assistant_chunks = 0

    def _should_send_vision(
        self,
        *,
        chunk_index: int,
        playback_active: bool,
        speech_detected: bool,
        speech_activation_window_open: bool,
    ) -> bool:
        if not self._vision_enabled or self.video is None:
            return False
        if chunk_index % self._video_every_n_chunks != 0:
            return False
        if playback_active:
            return False
        return (
            speech_detected
            or speech_activation_window_open
            or self._speech_recent_chunks_remaining > 0
        )

    def _preflight(self) -> None:
        ensure_sounddevice_available()

        required_commands = ["ffmpeg"]
        if not prefers_sounddevice_playback(self.config.audio.playback_device):
            required_commands.append("aplay")

        for command in required_commands:
            if subprocess.run(["bash", "-lc", f"command -v {command}"], check=False, capture_output=True).returncode != 0:
                raise RuntimeError(f"Missing required command: {command}")

        validate_capture_device(
            self.config.audio.capture_device,
            sample_rate=self.config.audio.input_sample_rate,
            channels=1,
            dtype="float32",
        )
        if prefers_sounddevice_playback(self.config.audio.playback_device):
            validate_playback_device(
                self.config.audio.playback_device,
                sample_rate=self.config.audio.playback_sample_rate,
                channels=self.config.audio.playback_channels,
                dtype="float32",
            )

        if self.mode == "omni" and not Path(self.config.video.camera_device).exists():
            raise FileNotFoundError(f"Camera device not found: {self.config.video.camera_device}")

        reference_audio = Path(self.config.audio.reference_audio_path)
        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        if self.config.model.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the current Python environment")
        self._backend.preflight()

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
        self._observed_loggers = [logging.getLogger(name) for name in self._backend.observed_logger_names]
        for target_logger in self._observed_loggers:
            target_logger.addHandler(self._signal_handler)

    def _detach_runtime_signal_watchers(self) -> None:
        if self._signal_handler is None:
            return
        for target_logger in self._observed_loggers:
            target_logger.removeHandler(self._signal_handler)
        self._observed_loggers = []
        self._signal_handler = None

    def _update_speech_activation(
        self,
        *,
        audio_rms: float,
        speech_detected: bool,
        playback_active: bool,
        barge_in_detected: bool,
    ) -> bool:
        if playback_active and not barge_in_detected:
            self._speech_activation_streak = 0
            self._speech_recent_chunks_remaining = 0
            return False

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

    def _update_chunk_barge_in(
        self,
        *,
        playback_active: bool,
        playback_active_ms: float,
        playback_remaining_ms: float,
        audio_rms: float,
        audio_peak: float,
    ) -> bool:
        if (
            not playback_active
            or playback_active_ms < self.config.audio.interrupt_min_playback_ms
            or playback_remaining_ms <= self.config.audio.interrupt_tail_protect_ms
        ):
            self._chunk_barge_in_streak = 0
            return False

        if (
            audio_rms >= self.config.runtime.chunk_barge_in_rms_threshold
            and audio_peak >= self.config.runtime.chunk_barge_in_peak_threshold
        ):
            self._chunk_barge_in_streak += 1
        else:
            self._chunk_barge_in_streak = 0

        if self._chunk_barge_in_streak >= self.config.runtime.chunk_barge_in_consecutive_chunks:
            self._chunk_barge_in_streak = 0
            LOGGER.info(
                "Chunk-level barge-in fallback triggered rms=%.4f peak=%.4f playback_active_ms=%.1f remaining_ms=%.1f",
                audio_rms,
                audio_peak,
                playback_active_ms,
                playback_remaining_ms,
            )
            return True
        return False

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
