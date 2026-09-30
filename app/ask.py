"""Ask scene: one question streamed to every variant at once, with timing measured per variant.

Live mode streams from the vLLM endpoints. If a live request fails, or its first token takes longer than
FIRST_TOKEN_TIMEOUT_S, or it stalls for STALL_TIMEOUT_S, a preset question falls back to that setup's
recorded answer, labeled as recorded and never passed off as live. Replay mode (no models connected)
streams text at the speed measured for that variant in the benchmark, using that variant's captured
output for the preset questions. A setup with no capture shows a note instead, so replay never shows
an answer no model gave.
"""

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import AsyncIterator
from datetime import datetime

from app.config import settings
from app.openshift import openshift_client
from app.quality import PROMPTS, recorded_answer

logger = logging.getLogger(__name__)

# preset key -> (scenario, button label). The order is the order of the buttons.
PRESETS = {
    "reasoning": ("complex_reasoning", "Sheep riddle"),
    "code": ("code_generation", "Python function"),
    "summary": ("summarization", "Three bullet summary"),
    "decline": ("polite_decline", "Decline a meeting"),
    "fact": ("quick_fact", "Quick fact"),
    "puzzle": ("logic_puzzle", "Logic puzzle"),
    "explain": ("long_explanation", "Explain KV cache"),
    "json": ("json_extraction", "Extract to JSON"),
}
REPLAY_NOTE = (
    "Replay mode: the live models aren't connected right now, so this column plays back the speed "
    "this setup measured on the H200s without calling a model. Connect the vLLM endpoints to see "
    "real answers to your own questions."
)
NOT_CAPTURED_NOTE = (
    "Replay mode: no answer to this question was captured for this setup, so this column plays back "
    "the speed it measured on the H200s without calling a model."
)
NO_RECORDING_TYPED = "Live request failed. No recording exists for a typed question."
NO_RECORDING_PRESET = "Live request failed, and no answer to this question was recorded for this setup."
NOT_LIVE_TODAY = "Not live today. Recorded answers cover the preset questions."
NOT_RECORDED_YET = "Not live today, and no answer to this question was recorded for this setup yet."
MAX_TOKENS = 1024  # high enough that answer length differences between setups show up
# A healthy first token takes about 0.3 s, so 8 s only trips on a real failure.
FIRST_TOKEN_TIMEOUT_S = 8.0
STALL_TIMEOUT_S = 10.0
_sleep = asyncio.sleep  # indirection so tests can replay instantly


def event(kind: str, **data) -> bytes:
    return (json.dumps({"t": kind, **data}) + "\n").encode()


def scenario_for(preset: str | None) -> str | None:
    found = PRESETS.get(preset or "")
    return found[0] if found else None


def preset_list() -> list[dict]:
    return [{"key": key, "label": label, "prompt": PROMPTS[sc]} for key, (sc, label) in PRESETS.items()]


def replay_text(variant: str, preset: str | None) -> tuple[str, str, dict | None]:
    """(text, where it came from, the recording): "captured" model output or a replay "note"."""
    scenario = scenario_for(preset)
    if not scenario:
        return REPLAY_NOTE, "note", None
    found = recorded_answer(settings.captures_for(variant), scenario)
    if not found:
        return NOT_CAPTURED_NOTE, "note", None
    return found["text"], "captured", found


def preset_prompt(preset: str | None) -> str | None:
    scenario = scenario_for(preset)
    return PROMPTS[scenario] if scenario else None


async def paced(text: str, tps: float) -> AsyncIterator[bytes]:
    """Stream text as delta events at roughly tps tokens per second."""
    pieces = re.findall(r"\S+\s*|\s+", text)
    # about 1.3 tokens per word piece for English text
    per_piece = 1.3 / tps
    buf, due = "", 0.0
    for piece in pieces:
        buf += piece
        due += per_piece * random.uniform(0.85, 1.15)
        if due >= 0.05:
            await _sleep(due)
            yield event("delta", text=buf)
            buf, due = "", 0.0
    if buf:
        await _sleep(due)
        yield event("delta", text=buf)


def timing(found: dict | None, variant: str, benchmark) -> dict:
    """What a replayed or recorded answer reports: the recording's own timings when it has them (the
    round-2 captures do), otherwise the benchmark's speed and first-token time with nothing invented."""
    bm = benchmark.variant(variant)
    if found and found.get("tokens_per_second"):
        return {
            "tps": float(found["tokens_per_second"]), "tps_basis": "recorded",
            "ttft_ms": found.get("ttft_ms"), "ttft_basis": "recorded" if found.get("ttft_ms") else None,
            "total_ms": found.get("total_ms"), "completion_tokens": found.get("completion_tokens"),
        }
    return {
        "tps": float(bm.get("throughput_tps") or 40.0), "tps_basis": "benchmark",
        "ttft_ms": bm.get("ttft_ms_avg"), "ttft_basis": "benchmark" if bm.get("ttft_ms_avg") else None,
        "total_ms": None, "completion_tokens": (found or {}).get("completion_tokens"),
    }


def done_event(source: str, t: dict) -> bytes:
    return event("done", source=source, ttft_ms=t["ttft_ms"], ttft_basis=t["ttft_basis"],
                 total_ms=t["total_ms"],
                 completion_tokens=t["completion_tokens"], tokens_per_second=round(t["tps"], 1),
                 tps_basis=t["tps_basis"])


async def replay_stream(variant: str, preset: str | None, benchmark) -> AsyncIterator[bytes]:
    text, text_source, found = replay_text(variant, preset)
    t = timing(found, variant, benchmark)
    yield event("start", source="replay", text_source=text_source)
    async for chunk in paced(text, t["tps"]):
        yield chunk
    yield done_event("replay", t)


def recorded_label(captured_at: str | None, live_tokens: int, failed: bool = True) -> str:
    try:
        when = datetime.fromisoformat(captured_at).strftime("%b %-d") if captured_at else None
    except ValueError:
        when = None
    label = f"Recorded {when}" if when else "Recorded earlier"
    if failed:
        label += " · live request failed"
        if live_tokens:
            label += f" after {live_tokens} tokens"
    return label


async def recorded_stream(
    variant: str, preset: str | None, live_tokens: int, benchmark
) -> AsyncIterator[bytes]:
    """After a live failure: that setup's recorded answer for the preset, labeled as recorded."""
    scenario = scenario_for(preset)
    found = recorded_answer(settings.captures_for(variant), scenario) if scenario else None
    if not found:
        yield event("error", detail=NO_RECORDING_PRESET if scenario else NO_RECORDING_TYPED)
        return
    yield event("fallback", label=recorded_label(found["captured_at"], live_tokens), live_tokens=live_tokens)
    t = timing(found, variant, benchmark)
    async for chunk in paced(found["text"], t["tps"]):
        yield chunk
    yield done_event("recorded", t)


async def planned_recorded_stream(variant: str, preset: str | None, benchmark) -> AsyncIterator[bytes]:
    """A setup switched to recorded for the day: never calls its endpoint, plays the preset's recording
    at the timing it was recorded with, and says it's recorded (not that anything failed)."""
    yield event("start", source="recorded")
    scenario = scenario_for(preset)
    found = recorded_answer(settings.captures_for(variant), scenario) if scenario else None
    if not found:
        yield event("delta", text=NOT_LIVE_TODAY if not scenario else NOT_RECORDED_YET)
        yield event("done", source="recorded", ttft_ms=None, total_ms=None, completion_tokens=None,
                    tokens_per_second=None, note=True)
        return
    yield event("recorded", label=recorded_label(found["captured_at"], 0, failed=False))
    t = timing(found, variant, benchmark)
    if t["ttft_basis"] == "recorded":
        await _sleep(t["ttft_ms"] / 1000)
    async for chunk in paced(found["text"], t["tps"]):
        yield chunk
    yield done_event("recorded", t)


async def _next_line(lines, timeout: float) -> str:
    return await asyncio.wait_for(lines.__anext__(), timeout=max(timeout, 0.01))


class LiveFailed(Exception):
    """A live request failed after streaming some tokens."""

    def __init__(self, tokens: int, cause: BaseException):
        super().__init__(f"after {tokens} tokens: {cause!r}")
        self.tokens = tokens


async def live_stream(variant: str, prompt: str) -> AsyncIterator[bytes]:
    endpoint = openshift_client.endpoint(variant)
    await openshift_client.start()
    client = openshift_client._client
    payload = {
        "model": settings.served_name_for(variant),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first = None
    chunks = 0
    usage_tokens = None
    yield event("start", source="live")
    try:
        request = client.build_request("POST", endpoint, json=payload)
        response = await asyncio.wait_for(client.send(request, stream=True), FIRST_TOKEN_TIMEOUT_S)
        try:
            response.raise_for_status()
            lines = response.aiter_lines()
            while True:
                # the first token must arrive within FIRST_TOKEN_TIMEOUT_S of sending, and after that
                # the stream may not go quiet for longer than STALL_TIMEOUT_S
                if first is None:
                    limit = FIRST_TOKEN_TIMEOUT_S - (time.perf_counter() - start)
                else:
                    limit = STALL_TIMEOUT_S
                try:
                    line = await _next_line(lines, limit)
                except StopAsyncIteration:
                    break
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    usage_tokens = chunk["usage"].get("completion_tokens")
                for choice in chunk.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        if first is None:
                            first = time.perf_counter()
                        chunks += 1
                        yield event("delta", text=text)
        finally:
            await response.aclose()
    except Exception as e:  # noqa: BLE001 - timeouts, HTTP errors and dropped connections alike
        raise LiveFailed(chunks, e) from e
    end = time.perf_counter()
    tokens = int(usage_tokens or chunks)
    yield event(
        "done",
        source="live",
        ttft_ms=round((first - start) * 1000) if first else None,
        total_ms=round((end - start) * 1000),
        completion_tokens=tokens,
        # same definition as benchmark_results.json: output tokens over the whole request time
        tokens_per_second=round(tokens / (end - start), 1) if end > start else None,
    )


async def ask_stream(
    variant: str, prompt: str, preset: str | None, live: bool, benchmark
) -> AsyncIterator[bytes]:
    try:
        if live and settings.mode_for(variant) == "recorded":
            async for chunk in planned_recorded_stream(variant, preset, benchmark):
                yield chunk
        elif live:
            try:
                async for chunk in live_stream(variant, prompt):
                    yield chunk
            except Exception as e:  # noqa: BLE001 - any live failure falls back for this column only
                logger.warning(f"Live ask for {variant} failed: {e}")
                tokens = e.tokens if isinstance(e, LiveFailed) else 0
                async for chunk in recorded_stream(variant, preset, tokens, benchmark):
                    yield chunk
        else:
            async for chunk in replay_stream(variant, preset, benchmark):
                yield chunk
    except Exception as e:  # noqa: BLE001 - any failure becomes an error event for this column only
        logger.warning(f"Ask stream for {variant} failed: {e}")
        yield event("error", detail="this model didn't answer")
