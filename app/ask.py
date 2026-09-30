"""Ask scene: one question streamed to every variant at once, with timing measured per variant.

Live mode streams from the vLLM endpoints. Replay mode (no models connected) streams text at the
speed measured for that variant in the benchmark, using captured outputs for the preset questions
when they exist and the baseline's text otherwise, so replay never invents a quality difference.
"""

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import AsyncIterator

from app.config import settings
from app.openshift import openshift_client
from app.quality import ILLUSTRATIVE, get_comparison

logger = logging.getLogger(__name__)

PRESETS = {
    "reasoning": "complex_reasoning",
    "code": "code_generation",
    "summary": "summarization",
}
REPLAY_NOTE = (
    "Replay mode: the live models aren't connected right now, so this column plays back the speed "
    "this setup measured on the H200s without calling a model. Connect the vLLM endpoints to see "
    "real answers to your own questions."
)
MAX_TOKENS = 1024  # high enough that answer length differences between setups show up
_sleep = asyncio.sleep  # indirection so tests can replay instantly


def event(kind: str, **data) -> bytes:
    return (json.dumps({"t": kind, **data}) + "\n").encode()


def replay_text(variant: str, preset: str | None) -> tuple[str, str, int | None]:
    """(text, where it came from, completion tokens if known): "captured" model output, a "scripted"
    example, or the replay "note"."""
    scenario = PRESETS.get(preset or "")
    if not scenario:
        return REPLAY_NOTE, "note", None
    comparison = get_comparison(scenario, [variant, "FP16"])
    if comparison["source"] == "captured" and variant in comparison["responses"]:
        found = comparison["responses"][variant]
        return found["text"], "captured", (found.get("usage") or {}).get("completion_tokens")
    return ILLUSTRATIVE[scenario]["FP16"], "scripted", None


def preset_prompt(preset: str | None) -> str | None:
    scenario = PRESETS.get(preset or "")
    return ILLUSTRATIVE[scenario]["prompt"] if scenario else None


async def replay_stream(variant: str, preset: str | None, benchmark) -> AsyncIterator[bytes]:
    text, text_source, captured_tokens = replay_text(variant, preset)
    pieces = re.findall(r"\S+\s*|\s+", text)
    tps = float(benchmark.variant(variant).get("throughput_tps") or 40.0)
    # about 1.3 tokens per word piece for English text
    per_piece = 1.3 / tps
    yield event("start", source="replay", text_source=text_source)
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
    # Replay only paces the text at the measured speed. It reports that measured speed and nothing
    # else, since first-token time, token count, and total time weren't measured for this text.
    # A captured answer's token count is real; everything else stays "live only".
    yield event("done", source="replay", ttft_ms=None, total_ms=None,
                completion_tokens=captured_tokens, tokens_per_second=round(tps, 1))


async def live_stream(variant: str, prompt: str) -> AsyncIterator[bytes]:
    endpoint = openshift_client.endpoint(variant)
    await openshift_client.start()
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
    async with openshift_client._client.stream("POST", endpoint, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
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
        if live:
            async for chunk in live_stream(variant, prompt):
                yield chunk
        else:
            async for chunk in replay_stream(variant, preset, benchmark):
                yield chunk
    except Exception as e:  # noqa: BLE001 - any failure becomes an error event for this column only
        logger.warning(f"Ask stream for {variant} failed: {e}")
        yield event("error", detail="this model didn't answer")
