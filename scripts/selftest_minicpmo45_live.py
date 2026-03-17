#!/usr/bin/env python3
import argparse
import asyncio
import base64
import io
import json
import ssl
import subprocess
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import soundfile as sf
import websockets
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "third_party" / "minicpm-o-4_5-pytorch-simple-demo"
COMMON_DIR = DEMO_DIR / "tests" / "cases" / "common"
USER_AUDIO_LONG = COMMON_DIR / "user_audio" / "000_user_audio0.wav"
USER_AUDIO_OMNI = COMMON_DIR / "user_audio" / "当出现植物大战僵尸的时候提醒我.wav"
REF_AUDIO = COMMON_DIR / "ref_audio" / "BH-Ref-HT-F224-Ref06_82_U001_话题_3_348s-355s.wav"
IMAGE_PATH = COMMON_DIR / "images" / "image.png"
START_SCRIPT = ROOT / "scripts" / "start_minicpmo45_pytorch_demo_spark.sh"
STOP_SCRIPT = ROOT / "scripts" / "stop_minicpmo45_pytorch_demo_spark.sh"
REPORT_ROOT = ROOT / ".minicpmo45_live_selftest"


def load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sr} for {path}")
    return np.asarray(audio, dtype=np.float32)


def pcm_f32_b64(audio: np.ndarray) -> str:
    return base64.b64encode(np.asarray(audio, dtype=np.float32).tobytes()).decode("ascii")


def chunk_audio(audio: np.ndarray, chunk_ms: int) -> List[np.ndarray]:
    step = int(16000 * (chunk_ms / 1000.0))
    return [audio[i:i + step] for i in range(0, len(audio), step) if len(audio[i:i + step]) > 0]


def save_float32_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def image_to_jpeg_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def keyword_hit(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def ws_url_from_gateway(gateway_url: str, path: str) -> str:
    if gateway_url.startswith("https://"):
        return "wss://" + gateway_url[len("https://"):] + path
    if gateway_url.startswith("http://"):
        return "ws://" + gateway_url[len("http://"):] + path
    raise ValueError(f"Unsupported gateway URL: {gateway_url}")


async def wait_health(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return
            except Exception:
                await asyncio.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for health: {url}")


async def ensure_service(gateway_url: str) -> None:
    try:
        await wait_health(f"{gateway_url}/health", timeout_s=3.0)
        return
    except Exception:
        pass

    subprocess.run(["bash", str(START_SCRIPT)], cwd=ROOT, check=True)
    await wait_health(f"{gateway_url}/health", timeout_s=300.0)
    await wait_health("http://127.0.0.1:22400/health", timeout_s=300.0)


def log_progress(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


@dataclass
class StageResult:
    name: str
    passed: bool
    summary: str
    report_path: Path
    audio_path: Optional[Path]
    details: Dict[str, Any]


async def run_half_duplex(gateway_url: str, stage_dir: Path) -> StageResult:
    log_progress("Plan 1/3: running Half-Duplex Audio live self test")
    session_id = f"hdx_live_{int(time.time())}"
    ws_url = ws_url_from_gateway(gateway_url, f"/ws/half_duplex/{session_id}")
    ssl_ctx = ssl._create_unverified_context()

    user_audio = load_mono_16k(USER_AUDIO_LONG)
    ref_audio = load_mono_16k(REF_AUDIO)
    send_audio = np.concatenate([user_audio, np.zeros(16000, dtype=np.float32)])
    chunks = chunk_audio(send_audio, chunk_ms=200)

    combined_ai_audio: List[np.ndarray] = []
    events: List[Dict[str, Any]] = []
    chunk_text_parts: List[str] = []
    final_text = ""
    prepared = False
    generating_seen = False
    turn_done = False
    error_text = None

    async with websockets.connect(ws_url, ssl=ssl_ctx, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "prepare",
            "system_content": [
                {"type": "text", "text": "请用中文复述用户说的话，只输出复述内容。"},
                {"type": "audio", "data": pcm_f32_b64(ref_audio)},
            ],
            "config": {
                "session": {"timeout_s": 120},
                "tts": {"enabled": True},
                "generation": {"max_new_tokens": 96, "length_penalty": 1.0, "temperature": 0.2},
                "vad": {"threshold": 0.6, "min_silence_duration_ms": 400},
            },
        }))

        async def receiver() -> None:
            nonlocal prepared, generating_seen, turn_done, final_text, error_text
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                msg["_recv_ts"] = time.time()
                events.append(msg)
                msg_type = msg.get("type")
                if msg_type == "prepared":
                    prepared = True
                elif msg_type == "generating":
                    generating_seen = True
                elif msg_type == "chunk":
                    if msg.get("text_delta"):
                        chunk_text_parts.append(msg["text_delta"])
                    if msg.get("audio_data"):
                        ai_audio = np.frombuffer(base64.b64decode(msg["audio_data"]), dtype=np.float32)
                        combined_ai_audio.append(ai_audio)
                elif msg_type == "turn_done":
                    turn_done = True
                    final_text = msg.get("text", "")
                    return
                elif msg_type == "error":
                    error_text = msg.get("error")
                    return
                elif msg_type == "timeout":
                    error_text = msg.get("reason", "timeout")
                    return

        recv_task = asyncio.create_task(receiver())

        while not prepared:
            await asyncio.sleep(0.05)

        send_start = time.monotonic()
        for chunk in chunks:
            await ws.send(json.dumps({"type": "audio_chunk", "audio_base64": pcm_f32_b64(chunk)}))
            await asyncio.sleep(len(chunk) / 16000.0)
            if recv_task.done():
                break
        send_end = time.monotonic()

        try:
            await asyncio.wait_for(recv_task, timeout=60.0)
        except asyncio.TimeoutError:
            error_text = "half_duplex_timeout_waiting_turn_done"

        try:
            await ws.send(json.dumps({"type": "stop"}))
        except Exception:
            pass

    if not final_text:
        final_text = "".join(chunk_text_parts)

    ai_audio_path = None
    if combined_ai_audio:
        ai_audio_path = stage_dir / "assistant_audio.wav"
        save_float32_wav(ai_audio_path, np.concatenate(combined_ai_audio), 24000)

    details = {
        "session_id": session_id,
        "prepared": prepared,
        "generating_seen": generating_seen,
        "turn_done": turn_done,
        "final_text": final_text,
        "error": error_text,
        "events_count": len(events),
        "send_duration_s": round(send_end - send_start, 3),
        "keyword_hit": keyword_hit(final_text, ["我错了", "下次", "不敢"]),
    }
    save_json(stage_dir / "report.json", {"details": details, "events": events})

    passed = prepared and generating_seen and turn_done and bool(ai_audio_path) and details["keyword_hit"]
    summary = (
        f"half_duplex prepared={prepared} generating={generating_seen} "
        f"turn_done={turn_done} keyword_hit={details['keyword_hit']} text={final_text[:120]!r}"
    )
    return StageResult(
        name="half_duplex",
        passed=passed,
        summary=summary,
        report_path=stage_dir / "report.json",
        audio_path=ai_audio_path,
        details=details,
    )


async def run_duplex_audio(gateway_url: str, stage_dir: Path) -> StageResult:
    log_progress("Plan 2/3: running Audio Full-Duplex live self test")
    session_id = f"adx_live_{int(time.time())}"
    ws_url = ws_url_from_gateway(gateway_url, f"/ws/duplex/{session_id}")
    ssl_ctx = ssl._create_unverified_context()

    user_audio = load_mono_16k(USER_AUDIO_LONG)
    send_audio = np.concatenate([user_audio, np.zeros(16000, dtype=np.float32)])
    chunks = chunk_audio(send_audio, chunk_ms=200)

    events: List[Dict[str, Any]] = []
    combined_ai_audio: List[np.ndarray] = []
    final_text_parts: List[str] = []
    prepared = False
    speak_results = 0
    error_text = None
    first_speak_ms = None
    end_of_turn_seen = False

    async with websockets.connect(ws_url, ssl=ssl_ctx, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "prepare",
            "system_prompt": (
                "Streaming Duplex Conversation. "
                "请用中文交流。当你听懂用户主要意思时，可以先简短说“收到”，"
                "然后继续听，最后简短复述用户的话。尽量不要等用户完全说完才开始回应。"
            ),
            "ref_audio_path": str(REF_AUDIO),
            "deferred_finalize": True,
        }))

        prepared_event = asyncio.Event()
        stop_receive = asyncio.Event()
        send_start = time.monotonic()
        send_end = None

        async def receiver() -> None:
            nonlocal prepared, speak_results, error_text, first_speak_ms, end_of_turn_seen
            while not stop_receive.is_set():
                raw = await ws.recv()
                msg = json.loads(raw)
                msg["_recv_ts"] = time.time()
                events.append(msg)
                msg_type = msg.get("type")
                if msg_type == "prepared":
                    prepared = True
                    prepared_event.set()
                elif msg_type == "result":
                    if not msg.get("is_listen", True):
                        speak_results += 1
                        if first_speak_ms is None:
                            first_speak_ms = round((time.monotonic() - send_start) * 1000, 1)
                        if msg.get("text"):
                            final_text_parts.append(msg["text"])
                        if msg.get("audio_data"):
                            ai_audio = np.frombuffer(base64.b64decode(msg["audio_data"]), dtype=np.float32)
                            combined_ai_audio.append(ai_audio)
                        if msg.get("end_of_turn"):
                            end_of_turn_seen = True
                            stop_receive.set()
                            return
                elif msg_type in ("error", "timeout"):
                    error_text = msg.get("error") or msg.get("reason")
                    stop_receive.set()
                    return
                elif msg_type == "stopped":
                    stop_receive.set()
                    return

        recv_task = asyncio.create_task(receiver())
        await asyncio.wait_for(prepared_event.wait(), timeout=30.0)

        for chunk in chunks:
            await ws.send(json.dumps({"type": "audio_chunk", "audio_base64": pcm_f32_b64(chunk)}))
            await asyncio.sleep(len(chunk) / 16000.0)
            if stop_receive.is_set():
                break
        send_end = time.monotonic()

        if not stop_receive.is_set():
            try:
                await asyncio.wait_for(recv_task, timeout=20.0)
            except asyncio.TimeoutError:
                error_text = "audio_duplex_timeout_waiting_result"
        try:
            await ws.send(json.dumps({"type": "stop"}))
        except Exception:
            pass
        stop_receive.set()
        if not recv_task.done():
            await asyncio.wait_for(recv_task, timeout=5.0)

    final_text = "".join(final_text_parts)
    ai_audio_path = None
    if combined_ai_audio:
        ai_audio_path = stage_dir / "assistant_audio.wav"
        save_float32_wav(ai_audio_path, np.concatenate(combined_ai_audio), 24000)

    speak_before_input_end = bool(first_speak_ms is not None and send_end is not None and first_speak_ms < (send_end - send_start) * 1000)
    details = {
        "session_id": session_id,
        "prepared": prepared,
        "speak_results": speak_results,
        "end_of_turn_seen": end_of_turn_seen,
        "final_text": final_text,
        "error": error_text,
        "events_count": len(events),
        "first_speak_ms": first_speak_ms,
        "send_duration_ms": round((send_end - send_start) * 1000, 1) if send_end else None,
        "speak_before_input_end": speak_before_input_end,
        "keyword_hit": keyword_hit(final_text, ["收到", "我错了", "下次", "下单"]),
    }
    save_json(stage_dir / "report.json", {"details": details, "events": events})

    passed = prepared and speak_results > 0 and bool(ai_audio_path) and details["keyword_hit"] and speak_before_input_end
    summary = (
        f"audio_duplex prepared={prepared} speak_results={speak_results} "
        f"overlap={speak_before_input_end} keyword_hit={details['keyword_hit']} "
        f"text={final_text[:120]!r}"
    )
    return StageResult(
        name="audio_duplex",
        passed=passed,
        summary=summary,
        report_path=stage_dir / "report.json",
        audio_path=ai_audio_path,
        details=details,
    )


async def run_duplex_omni(gateway_url: str, stage_dir: Path) -> StageResult:
    log_progress("Plan 3/3: running Omnimodal Full-Duplex live self test")
    session_id = f"omni_live_{int(time.time())}"
    ws_url = ws_url_from_gateway(gateway_url, f"/ws/duplex/{session_id}")
    ssl_ctx = ssl._create_unverified_context()

    user_audio = load_mono_16k(USER_AUDIO_OMNI)
    total_duration_s = 8.0
    total_samples = int(total_duration_s * 16000)
    send_audio = np.zeros(total_samples, dtype=np.float32)
    send_audio[:len(user_audio)] = user_audio
    chunks = chunk_audio(send_audio, chunk_ms=200)

    src_image = Image.open(IMAGE_PATH).convert("RGB")
    black_image = Image.new("RGB", src_image.size, (0, 0, 0))
    src_b64 = image_to_jpeg_b64(src_image)
    black_b64 = image_to_jpeg_b64(black_image)
    switch_chunk_idx = int(3.0 / 0.2)

    events: List[Dict[str, Any]] = []
    combined_ai_audio: List[np.ndarray] = []
    final_text_parts: List[str] = []
    prepared = False
    speak_results = 0
    error_text = None
    first_speak_ms = None
    first_speak_after_switch = False
    end_of_turn_seen = False
    max_vision_slices = 0

    async with websockets.connect(ws_url, ssl=ssl_ctx, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "prepare",
            "system_prompt": (
                "你是一个视频监控助手。只有当你确认画面中出现植物大战僵尸时，"
                "才用中文简短提醒“发现植物大战僵尸”。在看到目标画面之前请保持倾听。"
            ),
            "ref_audio_path": str(REF_AUDIO),
            "deferred_finalize": True,
            "max_slice_nums": 1,
        }))

        prepared_event = asyncio.Event()
        stop_receive = asyncio.Event()
        send_start = time.monotonic()
        send_end = None

        async def receiver() -> None:
            nonlocal prepared, speak_results, error_text, first_speak_ms, first_speak_after_switch, end_of_turn_seen, max_vision_slices
            while not stop_receive.is_set():
                raw = await ws.recv()
                msg = json.loads(raw)
                msg["_recv_ts"] = time.time()
                events.append(msg)
                msg_type = msg.get("type")
                if msg_type == "prepared":
                    prepared = True
                    prepared_event.set()
                elif msg_type == "result":
                    max_vision_slices = max(max_vision_slices, int(msg.get("vision_slices") or 0))
                    if not msg.get("is_listen", True):
                        speak_results += 1
                        if first_speak_ms is None:
                            first_speak_ms = round((time.monotonic() - send_start) * 1000, 1)
                            if send_end is None and first_speak_ms >= switch_chunk_idx * 200:
                                first_speak_after_switch = True
                        if msg.get("text"):
                            final_text_parts.append(msg["text"])
                        if msg.get("audio_data"):
                            ai_audio = np.frombuffer(base64.b64decode(msg["audio_data"]), dtype=np.float32)
                            combined_ai_audio.append(ai_audio)
                        if msg.get("end_of_turn"):
                            end_of_turn_seen = True
                            stop_receive.set()
                            return
                elif msg_type in ("error", "timeout"):
                    error_text = msg.get("error") or msg.get("reason")
                    stop_receive.set()
                    return
                elif msg_type == "stopped":
                    stop_receive.set()
                    return

        recv_task = asyncio.create_task(receiver())
        await asyncio.wait_for(prepared_event.wait(), timeout=30.0)

        for idx, chunk in enumerate(chunks):
            frame_b64 = black_b64 if idx < switch_chunk_idx else src_b64
            await ws.send(json.dumps({
                "type": "audio_chunk",
                "audio_base64": pcm_f32_b64(chunk),
                "frame_base64_list": [frame_b64],
                "max_slice_nums": 1,
            }))
            await asyncio.sleep(len(chunk) / 16000.0)
            if stop_receive.is_set():
                break
        send_end = time.monotonic()

        if not stop_receive.is_set():
            try:
                await asyncio.wait_for(recv_task, timeout=20.0)
            except asyncio.TimeoutError:
                error_text = "omni_duplex_timeout_waiting_result"
        try:
            await ws.send(json.dumps({"type": "stop"}))
        except Exception:
            pass
        stop_receive.set()
        if not recv_task.done():
            await asyncio.wait_for(recv_task, timeout=5.0)

    final_text = "".join(final_text_parts)
    ai_audio_path = None
    if combined_ai_audio:
        ai_audio_path = stage_dir / "assistant_audio.wav"
        save_float32_wav(ai_audio_path, np.concatenate(combined_ai_audio), 24000)

    details = {
        "session_id": session_id,
        "prepared": prepared,
        "speak_results": speak_results,
        "end_of_turn_seen": end_of_turn_seen,
        "final_text": final_text,
        "error": error_text,
        "events_count": len(events),
        "first_speak_ms": first_speak_ms,
        "send_duration_ms": round((send_end - send_start) * 1000, 1) if send_end else None,
        "first_speak_after_switch": first_speak_after_switch,
        "max_vision_slices": max_vision_slices,
        "keyword_hit": keyword_hit(final_text, ["植物", "僵尸", "提醒", "发现"]),
    }
    save_json(stage_dir / "report.json", {"details": details, "events": events})

    passed = (
        prepared and speak_results > 0 and bool(ai_audio_path) and details["keyword_hit"]
        and first_speak_after_switch and max_vision_slices > 0
    )
    summary = (
        f"omni_duplex prepared={prepared} speak_results={speak_results} "
        f"after_switch={first_speak_after_switch} vision_slices={max_vision_slices} "
        f"keyword_hit={details['keyword_hit']} text={final_text[:120]!r}"
    )
    return StageResult(
        name="omni_duplex",
        passed=passed,
        summary=summary,
        report_path=stage_dir / "report.json",
        audio_path=ai_audio_path,
        details=details,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Live selftest for MiniCPM-o 4.5 realtime demo")
    parser.add_argument("--gateway-url", default="https://127.0.0.1:18006")
    parser.add_argument("--stage", choices=["all", "half_duplex", "audio_duplex", "omni_duplex"], default="all")
    args = parser.parse_args()

    log_progress("Checking MiniCPM-o 4.5 PyTorch demo service health")
    await ensure_service(args.gateway_url)
    log_progress("Service is ready, starting staged live self tests")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = REPORT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stages: List[Tuple[str, Any]] = [
        ("half_duplex", run_half_duplex),
        ("audio_duplex", run_duplex_audio),
        ("omni_duplex", run_duplex_omni),
    ]
    if args.stage != "all":
        stages = [stage for stage in stages if stage[0] == args.stage]

    results: List[StageResult] = []
    for stage_name, runner in stages:
        stage_dir = run_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        result = await runner(args.gateway_url, stage_dir)
        results.append(result)
        log_progress(f"{stage_name} {'PASS' if result.passed else 'FAIL'}: {result.summary}")

    summary_lines = [
        "# MiniCPM-o 4.5 Live Selftest",
        "",
        f"- run_id: `{run_id}`",
        f"- gateway: `{args.gateway_url}`",
        "",
        "| Stage | Passed | Summary | Report |",
        "|---|---|---|---|",
    ]
    for result in results:
        summary_lines.append(
            f"| {result.name} | {'PASS' if result.passed else 'FAIL'} | {result.summary.replace('|', '/')} | {result.report_path} |"
        )
    save_markdown(run_dir / "summary.md", "\n".join(summary_lines) + "\n")

    print(json.dumps({
        "run_id": run_id,
        "run_dir": str(run_dir),
        "results": [
            {
                "stage": result.name,
                "passed": result.passed,
                "summary": result.summary,
                "report_path": str(result.report_path),
                "audio_path": str(result.audio_path) if result.audio_path else None,
                "details": result.details,
            }
            for result in results
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
