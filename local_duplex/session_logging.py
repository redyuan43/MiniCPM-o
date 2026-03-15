"""Session-level interaction logging for the local duplex runtime."""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from local_duplex.audio import decode_model_audio
from local_duplex.config import DuplexRuntimeConfig


LOGGER = logging.getLogger("local_duplex")


class InteractionSessionLogger:
    """Writes a user-facing per-session interaction timeline and summary."""

    def __init__(self, runtime_dir: Path, mode: str, config: DuplexRuntimeConfig) -> None:
        self.runtime_dir = runtime_dir
        self.mode = mode
        self.config = config
        self.session_id = f"{mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid4().hex[:6]}"
        self.session_dir = runtime_dir / "sessions" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.session_dir / "interaction.jsonl"
        self.markdown_path = self.session_dir / "interaction.md"
        self.summary_path = self.session_dir / "summary.json"
        self.latest_session_path = runtime_dir / "latest_session"
        self._started_at = time.time()
        self._events: list[dict[str, Any]] = []
        self._recorder = self._create_vendor_recorder()
        self._finalized = False

        self.latest_session_path.write_text(str(self.session_dir), encoding="utf-8")
        self._append_event(
            {
                "type": "session_started",
                "ts": self._now_iso(),
                "mode": self.mode,
                "session_id": self.session_id,
                "runtime_dir": str(self.runtime_dir),
                "config": asdict(self.config),
            }
        )
        self._rewrite_outputs()

    @property
    def session_path(self) -> Path:
        return self.session_dir

    def record_barge_in(
        self,
        chunk_index: int,
        audio_rms: float,
        audio_peak: float,
        playback_active_ms: float,
        playback_remaining_ms: float,
        barge_in_kind: str,
        barge_in_group_id: int,
        assistant_turn_role: str,
    ) -> None:
        self._append_event(
            {
                "type": "system",
                "ts": self._now_iso(),
                "event": "barge_in",
                "chunk_index": chunk_index,
                "audio_rms": round(audio_rms, 4),
                "audio_peak": round(audio_peak, 4),
                "playback_active_ms": round(playback_active_ms, 1),
                "playback_remaining_ms": round(playback_remaining_ms, 1),
                "barge_in_kind": barge_in_kind,
                "barge_in_group_id": barge_in_group_id,
                "assistant_turn_role": assistant_turn_role,
            }
        )
        self._rewrite_outputs()

    def record_reset(self, reason: str, chunk_index: int) -> None:
        self._append_event(
            {
                "type": "system",
                "ts": self._now_iso(),
                "event": "session_reset",
                "chunk_index": chunk_index,
                "reason": reason,
            }
        )
        self._rewrite_outputs()

    def record_health_change(self, state: str, reason: str, chunk_index: int) -> None:
        self._append_event(
            {
                "type": "system",
                "ts": self._now_iso(),
                "event": "health_change",
                "chunk_index": chunk_index,
                "state": state,
                "reason": reason,
            }
        )
        self._rewrite_outputs()

    def record_chunk(
        self,
        *,
        chunk_index: int,
        audio_rms: float,
        audio_peak: float,
        speech_detected: bool,
        speech_recent: bool,
        playback_active: bool,
        playback_active_ms: float,
        playback_remaining_ms: float,
        vision_enabled: bool,
        vision_used: bool,
        frame: Image.Image | None,
        result,
        prefill_ms: float,
        latency_ms: float,
        barge_in_detected: bool,
        reset_reason: str | None,
        session_health: str,
        consistency_error_count: int,
        kv_reset_count: int,
        speech_activation_window_open: bool,
        model_decision_bias: str,
        unsolicited_speak_suppressed: bool,
        unsolicited_speak_suppressed_count: int,
        assistant_turn_role: str,
        interrupt_group_id: int | None,
    ) -> None:
        user_frame_rel = self._save_frame(chunk_index, frame) if vision_used and frame is not None else None
        ai_audio_rel, ai_audio_duration_ms = self._save_ai_audio(chunk_index, result)

        result_text = (result.text or "").strip().replace("\n", " ")
        result_dict = result.model_dump()
        result_dict["wall_clock_ms"] = round(latency_ms, 1)

        if self._recorder is not None:
            self._recorder.record_chunk(
                index=chunk_index,
                receive_ts_ms=(time.time() - self._started_at) * 1000,
                result_dict=result_dict,
                prefill_ms=prefill_ms,
                user_audio_rel=None,
                user_frame_rel=user_frame_rel,
                ai_audio_rel=ai_audio_rel,
                ai_audio_samples=int(ai_audio_duration_ms * self.config.audio.model_output_sample_rate / 1000)
                if ai_audio_duration_ms
                else 0,
            )

        entry = {
            "type": "chunk",
            "ts": self._now_iso(),
            "since_session_start_ms": round((time.time() - self._started_at) * 1000, 1),
            "chunk_index": chunk_index,
            "input": {
                "audio_rms": round(audio_rms, 4),
                "audio_peak": round(audio_peak, 4),
                "speech_detected": speech_detected,
                "speech_recent": speech_recent,
                "playback_active": playback_active,
                "playback_active_ms": round(playback_active_ms, 1),
                "playback_remaining_ms": round(playback_remaining_ms, 1),
            },
            "vision": {
                "enabled": vision_enabled,
                "used": vision_used,
                "frame_path": user_frame_rel,
            },
            "assistant": {
                "state": "listen" if result.is_listen else "speak",
                "text": result_text,
                "end_of_turn": bool(result.end_of_turn),
                "backend_end_of_turn": bool(
                    getattr(result, "backend_end_of_turn", getattr(result, "end_of_turn", False))
                ),
                "ended_with_listen": bool(getattr(result, "ended_with_listen", False)),
                "stop_reason": str(getattr(result, "stop_reason", "") or "empty"),
                "turn_role": assistant_turn_role,
                "interrupt_group_id": interrupt_group_id,
                "audio_path": ai_audio_rel,
                "audio_duration_ms": ai_audio_duration_ms,
            },
            "performance": {
                "chunk_budget_ms": self.config.audio.chunk_ms,
                "prefill_ms": round(prefill_ms, 1),
                "latency_ms": round(latency_ms, 1),
                "cost_all_ms": self._round_or_none(result.cost_all_ms),
                "cost_llm_ms": self._round_or_none(result.cost_llm_ms),
                "cost_tts_prep_ms": self._round_or_none(result.cost_tts_prep_ms),
                "cost_tts_ms": self._round_or_none(result.cost_tts_ms),
                "cost_token2wav_ms": self._round_or_none(result.cost_token2wav_ms),
                "n_tokens": result.n_tokens,
                "n_tts_tokens": result.n_tts_tokens,
                "latency_budget_ms": (
                    self.config.runtime.listen_latency_budget_ms
                    if result.is_listen
                    else self.config.runtime.speak_latency_budget_ms
                ),
                "over_budget": (
                    latency_ms > self.config.runtime.listen_latency_budget_ms
                    if result.is_listen
                    else latency_ms > self.config.runtime.speak_latency_budget_ms
                ),
            },
            "analysis": {
                "fragmented_speech": bool(
                    not result.is_listen
                    and not getattr(result, "backend_end_of_turn", getattr(result, "end_of_turn", False))
                ),
                "barge_in_detected": barge_in_detected,
                "reset_after_chunk": reset_reason,
                "session_health": session_health,
                "consistency_error_count": consistency_error_count,
                "kv_reset_count": kv_reset_count,
                "speech_activation_window_open": speech_activation_window_open,
                "model_decision_bias": model_decision_bias,
                "unsolicited_speak_suppressed": unsolicited_speak_suppressed,
                "unsolicited_speak_suppressed_count": unsolicited_speak_suppressed_count,
                "likely_causes": self._build_likely_causes(
                    result=result,
                    latency_ms=latency_ms,
                    barge_in_detected=barge_in_detected,
                ),
            },
        }
        self._append_event(entry)
        self._rewrite_outputs()

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._append_event(
            {
                "type": "session_finished",
                "ts": self._now_iso(),
                "duration_s": round(time.time() - self._started_at, 1),
            }
        )
        self._rewrite_outputs()
        if self._recorder is not None:
            try:
                self._recorder.finalize()
            except Exception:
                LOGGER.exception("Failed to finalize vendor session recorder")

    def _create_vendor_recorder(self):
        try:
            from session_recorder import DuplexSessionRecorder
        except Exception:
            LOGGER.warning("Vendor session recorder unavailable; continuing with local interaction logs only")
            return None

        data_dir = str(Path("../../.local_duplex"))
        app_type = "omni_duplex" if self.mode == "omni" else "audio_duplex"
        return DuplexSessionRecorder(
            session_id=self.session_id,
            app_type=app_type,
            worker_id=0,
            config_snapshot=asdict(self.config),
            data_dir=data_dir,
        )

    def _save_frame(self, chunk_index: int, frame: Image.Image) -> str | None:
        if self._recorder is None:
            out_path = self.session_dir / "user_frames" / f"{chunk_index:03d}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            frame.save(out_path, format="JPEG", quality=90)
            return str(out_path.relative_to(self.session_dir))

        with io.BytesIO() as buf:
            frame.save(buf, format="JPEG", quality=90)
            return self._recorder.save_user_frame(chunk_index, buf.getvalue())

    def _save_ai_audio(self, chunk_index: int, result) -> tuple[str | None, int]:
        if result.is_listen or not result.audio_data:
            return None, 0

        audio_array = decode_model_audio(result.audio_data)
        if self._recorder is not None:
            rel = self._recorder.save_ai_audio(self._recorder.turn_index, chunk_index, audio_array)
            duration_ms = round(len(audio_array) / self.config.audio.model_output_sample_rate * 1000)
            return rel, duration_ms

        out_path = self.session_dir / "ai_audio" / f"chunk_{chunk_index:03d}.raw.f32"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_array.astype("float32", copy=False).tobytes())
        duration_ms = round(len(audio_array) / self.config.audio.model_output_sample_rate * 1000)
        return str(out_path.relative_to(self.session_dir)), duration_ms

    def _append_event(self, event: dict[str, Any]) -> None:
        self._events.append(event)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _rewrite_outputs(self) -> None:
        summary = self._build_summary()
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.markdown_path.write_text(self._render_markdown(summary), encoding="utf-8")

    def _build_summary(self) -> dict[str, Any]:
        chunk_events = [event for event in self._events if event["type"] == "chunk"]
        system_events = [event for event in self._events if event["type"] == "system"]

        speak_events = [event for event in chunk_events if event["assistant"]["state"] == "speak"]
        listen_events = [event for event in chunk_events if event["assistant"]["state"] == "listen"]
        speak_latencies = [event["performance"]["latency_ms"] for event in speak_events]
        listen_latencies = [event["performance"]["latency_ms"] for event in listen_events]
        omni_latencies = [event["performance"]["latency_ms"] for event in chunk_events if event["vision"]["used"]]
        audio_latencies = [event["performance"]["latency_ms"] for event in chunk_events if not event["vision"]["used"]]
        worker_decode_latencies = [
            event["performance"]["cost_llm_ms"]
            for event in speak_events
            if event["performance"]["cost_llm_ms"] is not None
        ]
        worker_wav_wait_latencies = [
            event["performance"]["cost_tts_prep_ms"]
            for event in speak_events
            if event["performance"]["cost_tts_prep_ms"] is not None
        ]
        worker_trailing_wait_latencies = [
            event["performance"]["cost_token2wav_ms"]
            for event in speak_events
            if event["performance"]["cost_token2wav_ms"] is not None
        ]
        fragmented = [event for event in speak_events if event["analysis"]["fragmented_speech"]]
        over_budget = [event for event in speak_events if event["performance"]["over_budget"]]
        vision_used_count = sum(1 for event in chunk_events if event["vision"]["used"])
        barge_ins = [event for event in system_events if event["event"] == "barge_in"]
        effective_barge_ins = [event for event in barge_ins if event.get("barge_in_kind") == "effective_interrupt"]
        confirmation_overlaps = [event for event in barge_ins if event.get("barge_in_kind") == "confirmation_overlap"]
        resets = [event for event in system_events if event["event"] == "session_reset"]
        health_changes = [event for event in system_events if event["event"] == "health_change"]
        first_speak_event = speak_events[0] if speak_events else None
        consistency_error_count = max(
            [event["analysis"].get("consistency_error_count", 0) for event in chunk_events] or [0]
        )
        kv_reset_count = max(
            [event["analysis"].get("kv_reset_count", 0) for event in chunk_events] or [0]
        )
        stuck_listen_count = sum(1 for event in health_changes if event.get("state") == "stuck_listen")
        unsolicited_speak_suppressed_count = max(
            [event["analysis"].get("unsolicited_speak_suppressed_count", 0) for event in chunk_events] or [0]
        )

        conversation_flow = self._build_conversation_flow(chunk_events, system_events)
        assistant_turns = [event for event in conversation_flow if event["kind"] == "assistant"]

        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "backend": self.config.model.backend,
            "model_variant": (
                self.config.model.gguf_variant if self.config.model.backend == "gguf" else "awq"
            ),
            "gguf_variant": self.config.model.gguf_variant if self.config.model.backend == "gguf" else None,
            "session_dir": str(self.session_dir),
            "started_at": self._events[0]["ts"] if self._events else self._now_iso(),
            "total_chunks": len(chunk_events),
            "listen_chunks": len(listen_events),
            "speak_chunks": len(speak_events),
            "vision_chunks": vision_used_count,
            "barge_in_count": len(effective_barge_ins),
            "barge_in_raw_count": len(barge_ins),
            "barge_in_confirmation_count": len(confirmation_overlaps),
            "reset_count": len(resets),
            "consistency_error_count": consistency_error_count,
            "kv_reset_count": kv_reset_count,
            "stuck_listen_count": stuck_listen_count,
            "unsolicited_speak_suppressed_count": unsolicited_speak_suppressed_count,
            "health_change_count": len(health_changes),
            "last_session_health": chunk_events[-1]["analysis"]["session_health"] if chunk_events else "healthy",
            "fragmented_speak_chunks": len(fragmented),
            "speak_over_budget_chunks": len(over_budget),
            "first_speak_delay_ms": first_speak_event["since_session_start_ms"] if first_speak_event else None,
            "avg_speak_latency_ms": self._avg(speak_latencies),
            "avg_listen_latency_ms": self._avg(listen_latencies),
            "avg_omni_latency_ms": self._avg(omni_latencies),
            "avg_audio_latency_ms": self._avg(audio_latencies),
            "avg_worker_decode_ms": self._avg(worker_decode_latencies),
            "avg_worker_wav_wait_ms": self._avg(worker_wav_wait_latencies),
            "avg_worker_trailing_wait_ms": self._avg(worker_trailing_wait_latencies),
            "speak_latency_budget_ms": self.config.runtime.speak_latency_budget_ms,
            "listen_latency_budget_ms": self.config.runtime.listen_latency_budget_ms,
            "assistant_turn_count": len(assistant_turns),
            "avg_chunks_per_assistant_turn": self._avg(
                [turn["chunk_end"] - turn["chunk_start"] + 1 for turn in assistant_turns]
            ),
            "avg_turn_audio_duration_ms": self._avg(
                [turn["audio_duration_ms"] for turn in assistant_turns]
            ),
            "conversation_flow": conversation_flow,
            "likely_stutter_causes": self._build_stutter_causes(
                over_budget_count=len(over_budget),
                fragmented_count=len(fragmented),
                barge_in_count=len(effective_barge_ins),
                barge_in_confirmation_count=len(confirmation_overlaps),
                consistency_error_count=consistency_error_count,
                kv_reset_count=kv_reset_count,
                stuck_listen_count=stuck_listen_count,
            ),
        }

    def _build_conversation_flow(
        self,
        chunk_events: list[dict[str, Any]],
        system_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        flow: list[dict[str, Any]] = []

        user_segment: list[dict[str, Any]] = []
        assistant_segment: list[dict[str, Any]] = []

        def flush_user_segment() -> None:
            nonlocal user_segment
            if not user_segment:
                return
            flow.append(
                {
                    "kind": "user",
                    "ts": user_segment[0]["ts"],
                    "chunk_start": user_segment[0]["chunk_index"],
                    "chunk_end": user_segment[-1]["chunk_index"],
                    "speech_chunks": len(user_segment),
                    "avg_rms": sum(event["input"]["audio_rms"] for event in user_segment) / len(user_segment),
                    "vision_used": any(event["vision"]["used"] for event in user_segment),
                }
            )
            user_segment = []

        def flush_assistant_segment() -> None:
            nonlocal assistant_segment
            if not assistant_segment:
                return
            flow.append(
                {
                    "kind": "assistant",
                    "ts": assistant_segment[0]["ts"],
                    "chunk_start": assistant_segment[0]["chunk_index"],
                    "chunk_end": assistant_segment[-1]["chunk_index"],
                    "text": "".join(event["assistant"]["text"] for event in assistant_segment),
                    "avg_latency_ms": self._avg([event["performance"]["latency_ms"] for event in assistant_segment]),
                    "audio_duration_ms": sum(event["assistant"]["audio_duration_ms"] for event in assistant_segment),
                    "vision_used": any(event["vision"]["used"] for event in assistant_segment),
                    "fragmented": any(event["analysis"]["fragmented_speech"] for event in assistant_segment),
                    "turn_role": assistant_segment[0]["assistant"].get("turn_role", "normal_reply"),
                    "interrupt_group_id": assistant_segment[0]["assistant"].get("interrupt_group_id"),
                    "stop_reasons": [event["assistant"]["stop_reason"] for event in assistant_segment],
                }
            )
            assistant_segment = []

        for event in chunk_events:
            if event["input"]["speech_detected"]:
                user_segment.append(event)
            else:
                flush_user_segment()

            if event["assistant"]["state"] == "speak" and event["assistant"]["text"]:
                assistant_segment.append(event)
            else:
                flush_assistant_segment()

        flush_user_segment()
        flush_assistant_segment()

        for event in system_events:
            if event["event"] == "barge_in":
                if event.get("barge_in_kind") == "confirmation_overlap":
                    message = "Confirmation overlap during interrupt acknowledgement"
                else:
                    message = "Barge-in detected during playback"
                flow.append(
                    {
                        "kind": "system",
                        "ts": event["ts"],
                        "message": message,
                    }
                )
            elif event["event"] == "session_reset":
                flow.append(
                    {
                        "kind": "system",
                        "ts": event["ts"],
                        "message": f"Session reset: {event['reason']}",
                    }
                )
            elif event["event"] == "health_change":
                flow.append(
                    {
                        "kind": "system",
                        "ts": event["ts"],
                        "message": f"Health -> {event['state']}: {event['reason']}",
                    }
                )

        flow.sort(key=lambda item: item["ts"])
        return flow

    def _build_stutter_causes(
        self,
        *,
        over_budget_count: int,
        fragmented_count: int,
        barge_in_count: int,
        barge_in_confirmation_count: int,
        consistency_error_count: int,
        kv_reset_count: int,
        stuck_listen_count: int,
    ) -> list[str]:
        causes: list[str] = []
        if barge_in_count:
            causes.append(f"{barge_in_count} barge-in events interrupted playback")
        if barge_in_confirmation_count:
            causes.append(f"{barge_in_confirmation_count} confirmation overlaps were grouped into interrupt acknowledgements")
        if over_budget_count:
            causes.append(
                f"{over_budget_count} speak chunks exceeded the {self.config.runtime.speak_latency_budget_ms}ms speak budget"
            )
        if fragmented_count:
            causes.append(f"{fragmented_count} speak chunks were mid-turn fragments rather than end-of-turn chunks")
        if consistency_error_count:
            causes.append(f"{consistency_error_count} consistency-error signals degraded the long-running session state")
        if kv_reset_count:
            causes.append(f"{kv_reset_count} KV-reset signals were observed during the session")
        if stuck_listen_count:
            causes.append(f"{stuck_listen_count} stuck-listen recoveries were triggered")
        if not causes:
            causes.append("No obvious stutter cause was detected in this session")
        return causes

    def _render_markdown(self, summary: dict[str, Any]) -> str:
        lines = [
            "# Local Duplex Interaction Session",
            "",
            f"- Session ID: `{summary['session_id']}`",
            f"- Mode: `{summary['mode']}`",
            f"- Backend: `{summary['backend']}`",
            f"- Model Variant: `{summary['model_variant']}`",
            f"- GGUF Variant: `{summary['gguf_variant'] or 'n/a'}`",
            f"- Session Dir: `{summary['session_dir']}`",
            "",
            "## Overview",
            "",
            f"- Total chunks: `{summary['total_chunks']}`",
            f"- Listen chunks: `{summary['listen_chunks']}`",
            f"- Speak chunks: `{summary['speak_chunks']}`",
            f"- Vision chunks: `{summary['vision_chunks']}`",
            f"- Session resets: `{summary['reset_count']}`",
            f"- Session health: `{summary['last_session_health']}`",
            f"- Barge-ins: `{summary['barge_in_count']}`",
            f"- Raw barge-ins: `{summary['barge_in_raw_count']}`",
            f"- Confirmation overlaps: `{summary['barge_in_confirmation_count']}`",
            f"- Consistency errors: `{summary['consistency_error_count']}`",
            f"- KV resets: `{summary['kv_reset_count']}`",
            f"- Stuck-listen recoveries: `{summary['stuck_listen_count']}`",
            f"- Suppressed unsolicited speaks: `{summary['unsolicited_speak_suppressed_count']}`",
            "",
            "## Performance",
            "",
            f"- Avg audio-only latency: `{self._format_number(summary['avg_audio_latency_ms'])} ms`",
            f"- Avg omni latency: `{self._format_number(summary['avg_omni_latency_ms'])} ms`",
            f"- Avg listen latency: `{self._format_number(summary['avg_listen_latency_ms'])} ms`",
            f"- Avg speak latency: `{self._format_number(summary['avg_speak_latency_ms'])} ms`",
            f"- Avg worker decode: `{self._format_number(summary['avg_worker_decode_ms'])} ms`",
            f"- Avg worker wav wait: `{self._format_number(summary['avg_worker_wav_wait_ms'])} ms`",
            f"- Avg worker trailing wait: `{self._format_number(summary['avg_worker_trailing_wait_ms'])} ms`",
            f"- First speak delay: `{self._format_number(summary['first_speak_delay_ms'])} ms`",
            f"- Assistant turns: `{summary['assistant_turn_count']}`",
            f"- Avg chunks per assistant turn: `{self._format_number(summary['avg_chunks_per_assistant_turn'])}`",
            f"- Avg assistant turn audio: `{self._format_number(summary['avg_turn_audio_duration_ms'])} ms`",
            f"- Fragmented speak chunks: `{summary['fragmented_speak_chunks']}`",
            f"- Speak chunks over budget: `{summary['speak_over_budget_chunks']}`",
            "",
            "## Conversation Flow",
            "",
        ]

        if not summary["conversation_flow"]:
            lines.append("- No chunk data recorded yet.")
        else:
            for event in summary["conversation_flow"]:
                stamp = event["ts"].split("T")[-1][:12].replace("Z", "")
                if event["kind"] == "user":
                    lines.append(
                        "- "
                        f"`{stamp}` user speaking "
                        f"(chunks {event['chunk_start']}-{event['chunk_end']}, "
                        f"avg_rms={event['avg_rms']:.4f}, "
                        f"vision={'used' if event['vision_used'] else 'not used'})"
                    )
                elif event["kind"] == "assistant":
                    text = event["text"] or "<empty>"
                    lines.append(
                        "- "
                        f"`{stamp}` assistant: {text} "
                        f"(chunks {event['chunk_start']}-{event['chunk_end']}, "
                        f"avg_latency={self._format_number(event['avg_latency_ms'])} ms, "
                        f"audio={self._format_number(event['audio_duration_ms'])} ms, "
                        f"turn_role={event.get('turn_role', 'normal_reply')}, "
                        f"vision={'used' if event['vision_used'] else 'not used'}, "
                        f"fragmented={'yes' if event['fragmented'] else 'no'}, "
                        f"stop_reason={','.join(event['stop_reasons'])})"
                    )
                else:
                    lines.append(f"- `{stamp}` {event['message']}")

        lines.extend(
            [
                "",
                "## Stutter Analysis",
                "",
            ]
        )
        for cause in summary["likely_stutter_causes"]:
            lines.append(f"- {cause}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _avg(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 1)

    @staticmethod
    def _round_or_none(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 1)

    @staticmethod
    def _format_number(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1f}"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _build_likely_causes(self, *, result, latency_ms: float, barge_in_detected: bool) -> list[str]:
        causes: list[str] = []
        if barge_in_detected:
            causes.append("barge_in")
        latency_budget_ms = (
            self.config.runtime.listen_latency_budget_ms
            if result.is_listen
            else self.config.runtime.speak_latency_budget_ms
        )
        if latency_ms > latency_budget_ms:
            causes.append("latency_over_budget")
        backend_end_of_turn = bool(getattr(result, "backend_end_of_turn", getattr(result, "end_of_turn", False)))
        if not result.is_listen and not backend_end_of_turn:
            causes.append("continuing_turn")
        stop_reason = str(getattr(result, "stop_reason", "") or "")
        if stop_reason == "chunk_limit":
            causes.append("chunk_limit")
        if result.n_tokens and result.n_tokens >= self.config.session.max_new_speak_tokens_per_chunk:
            causes.append("token_budget")
        return causes
