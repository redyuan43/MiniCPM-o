#!/usr/bin/env bash
# Parse the local duplex runtime log and produce a human-readable
# interaction summary: what was said, vision usage, and performance.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${1:-${ROOT_DIR}/.local_duplex/runtime.log}"

if [ ! -f "${LOG_FILE}" ]; then
  echo "Log file not found: ${LOG_FILE}" >&2
  exit 1
fi

python3 - "${ROOT_DIR}" "${LOG_FILE}" <<'PYEOF'
import json
from pathlib import Path
import re
import sys
from collections import defaultdict

root_dir = Path(sys.argv[1])
log_path = sys.argv[2]
config_path = root_dir / "configs" / "local_duplex.json"

chunk_budget_ms = 1000
if config_path.exists():
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        chunk_budget_ms = int(config.get("audio", {}).get("chunk_ms", chunk_budget_ms))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

# Patterns
RE_INPUT = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*"
    r"input chunk=(?P<chunk>\d+) "
    r"audio_rms=(?P<rms>[\d.]+) .*"
    r"speech_detected=(?P<speech>\w+) .*"
    r"playback_active=(?P<playback>\w+) .*"
    r"vision_enabled=(?P<vision_en>\w+) "
    r"vision_attempt=(?P<vision_att>\w+)"
)
RE_LISTEN = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*"
    r"listen chunk=(?P<chunk>\d+) latency_ms=(?P<lat>[\d.]+)"
)
RE_SPEAK = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*"
    r"speak chunk=(?P<chunk>\d+) latency_ms=(?P<lat>[\d.]+) text=(?P<text>.*)"
)
RE_RESET = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*"
    r"Resetting duplex session: (?P<reason>.*)"
)
RE_INTERRUPT = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*Barge-in detected"
)

# Accumulators
user_speech_chunks = []      # chunks where speech_detected=True
model_speak_segments = []    # list of {chunks, texts, latencies, start_ts}
current_speak_seg = None
vision_attempts = 0
vision_omni_chunks = 0
total_chunks = 0
listen_count = 0
speak_count = 0
speak_latencies = []
listen_latencies = []
session_resets = []
interrupts = []

with open(log_path, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = RE_INPUT.match(line)
        if m:
            total_chunks += 1
            if m.group("speech") == "True":
                user_speech_chunks.append((m.group("ts"), int(m.group("chunk")), float(m.group("rms"))))
            if m.group("vision_att") == "True":
                vision_attempts += 1
            if m.group("vision_en") == "True":
                vision_omni_chunks += 1
            continue

        m = RE_SPEAK.match(line)
        if m:
            speak_count += 1
            chunk = int(m.group("chunk"))
            lat = float(m.group("lat"))
            text = m.group("text").strip()
            speak_latencies.append(lat)
            if current_speak_seg is None or chunk - current_speak_seg["last_chunk"] > 1:
                if current_speak_seg:
                    model_speak_segments.append(current_speak_seg)
                current_speak_seg = {
                    "start_ts": m.group("ts"),
                    "chunks": [chunk],
                    "texts": [text],
                    "latencies": [lat],
                    "last_chunk": chunk,
                }
            else:
                current_speak_seg["chunks"].append(chunk)
                current_speak_seg["texts"].append(text)
                current_speak_seg["latencies"].append(lat)
                current_speak_seg["last_chunk"] = chunk
            continue

        m = RE_LISTEN.match(line)
        if m:
            listen_count += 1
            listen_latencies.append(float(m.group("lat")))
            continue

        m = RE_RESET.match(line)
        if m:
            session_resets.append((m.group("ts"), m.group("reason")))
            continue

        m = RE_INTERRUPT.match(line)
        if m:
            interrupts.append(m.group("ts"))

if current_speak_seg:
    model_speak_segments.append(current_speak_seg)

# Group user speech into segments (consecutive or near-consecutive chunks)
user_speech_segs = []
if user_speech_chunks:
    seg = [user_speech_chunks[0]]
    for i in range(1, len(user_speech_chunks)):
        if user_speech_chunks[i][1] - user_speech_chunks[i - 1][1] <= 3:
            seg.append(user_speech_chunks[i])
        else:
            user_speech_segs.append(seg)
            seg = [user_speech_chunks[i]]
    user_speech_segs.append(seg)

# Output
sep = "=" * 72
thin = "-" * 72

print(sep)
print("  LOCAL DUPLEX INTERACTION LOG")
print(sep)
print()

# Session overview
print("SESSION OVERVIEW")
print(thin)
print(f"  Total chunks processed:   {total_chunks}")
print(f"  Listen chunks:            {listen_count}")
print(f"  Speak chunks:             {speak_count}")
print(f"  Session resets:           {len(session_resets)}")
print(f"  Interrupts:               {len(interrupts)}")
print(f"  Vision frames sent:       {vision_attempts}")
print()

# Performance
print("PERFORMANCE")
print(thin)
if speak_latencies:
    avg_s = sum(speak_latencies) / len(speak_latencies)
    print(f"  Speak latency:  avg={avg_s:.0f}ms  min={min(speak_latencies):.0f}ms  max={max(speak_latencies):.0f}ms")
    over_budget = sum(1 for l in speak_latencies if l > chunk_budget_ms)
    print(
        f"  Speak chunks exceeding {chunk_budget_ms}ms budget: "
        f"{over_budget}/{len(speak_latencies)} ({100*over_budget/len(speak_latencies):.0f}%)"
    )
if listen_latencies:
    avg_l = sum(listen_latencies) / len(listen_latencies)
    print(f"  Listen latency: avg={avg_l:.0f}ms  min={min(listen_latencies):.0f}ms  max={max(listen_latencies):.0f}ms")
print()

# Conversation flow
print("CONVERSATION FLOW")
print(thin)

# Merge user speech segments and model speak segments into a timeline
events = []
for seg in user_speech_segs:
    duration_chunks = seg[-1][1] - seg[0][1] + 1
    avg_rms = sum(s[2] for s in seg) / len(seg)
    events.append(("user", seg[0][0], f"[User speaking] {len(seg)} chunks (chunks {seg[0][1]}-{seg[-1][1]}), avg_rms={avg_rms:.4f}"))

for seg in model_speak_segments:
    full_text = "".join(seg["texts"])
    avg_lat = sum(seg["latencies"]) / len(seg["latencies"])
    events.append(("model", seg["start_ts"], f"[Model speaking] \"{full_text}\"  ({len(seg['chunks'])} chunks, avg_lat={avg_lat:.0f}ms)"))

for ts in interrupts:
    events.append(("system", ts, "[Interrupt] Playback was stopped"))

for ts, reason in session_resets:
    short = reason.split()[0] if reason else reason
    events.append(("system", ts, f"[Session reset] {reason}"))

events.sort(key=lambda e: e[1])

for kind, ts, msg in events:
    time_part = ts.split(" ")[1].split(",")[0] if " " in ts else ts
    if kind == "user":
        print(f"  {time_part}  >> {msg}")
    elif kind == "model":
        print(f"  {time_part}  << {msg}")
    else:
        print(f"  {time_part}  -- {msg}")

print()

# Vision summary
print("VISION USAGE")
print(thin)
if vision_attempts > 0:
    print(f"  Vision frames fed to model: {vision_attempts} (every ~{total_chunks // max(vision_attempts,1)} chunks)")
    print(f"  Chunks with vision enabled: {vision_omni_chunks}/{total_chunks}")
else:
    print("  Vision was not active during this session.")
print()

# Stuttering analysis
print("STUTTERING ANALYSIS")
print(thin)
if speak_latencies:
    over = [l for l in speak_latencies if l > chunk_budget_ms]
    if over:
        print(f"  {len(over)}/{len(speak_latencies)} speak chunks exceeded the {chunk_budget_ms}ms chunk budget")
        print(f"  Average overshoot: {sum(o - chunk_budget_ms for o in over) / len(over):.0f}ms")
        print(f"  This causes the playback queue to drain, creating audible gaps.")
    else:
        if interrupts:
            print("  No chunk-level latency overrun was detected, but explicit interrupts can still cut playback.")
        else:
            print("  All speak chunks completed within the chunk budget. No chunk-level stutter detected.")
print()
print(sep)
PYEOF
