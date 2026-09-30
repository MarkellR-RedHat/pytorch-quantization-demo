"""Side-by-side quality comparison from captured model output only"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SCENARIO_ALIASES = {
    "complex_reasoning": ("complex_reasoning", "reasoning", "sheep"),
    "code_generation": ("code_generation", "code"),
    "summarization": ("summarization", "summary"),
}

VARIANT_DIRS = {
    "BF16": ("BF16", "BF16", "bf16", "bf16"),
    "FP8": ("FP8", "fp8"),
    "INT4": ("INT4", "INT4_AWQ", "int4", "int4_awq"),
    "SPEC_DECODE": ("SPEC_DECODE", "SPEC", "spec_decode", "spec"),
    "INT4_RH": ("INT4_RH", "int4_rh"),
}

# The exact prompts behind the captures in quality/<VARIANT>/<scenario>.json, and the same ones
# scripts/capture_presets.py sends. There is no made-up fallback text: a setup with no capture
# shows no answer. The Sep 29 spec decode captures match the BF16 ones on the sheep riddle only;
# the code and summary answers diverge partway through.
PROMPTS = {
    "complex_reasoning": "A farmer has 17 sheep. All but 9 run away. "
    "How many sheep does the farmer have left? Explain your reasoning step by step.",
    "code_generation": "Write a Python function that returns the second largest number in a list. "
    "Handle edge cases.",
    "summarization": "Summarize the key trade-offs of model quantization for production LLM deployments "
    "in 3 bullet points.",
    "polite_decline": "Write a short, polite reply declining a meeting on Friday at 3pm, "
    "and suggest next week instead.",
    "quick_fact": "What is the capital of Australia, and why isn't it Sydney? Answer in two sentences.",
    "logic_puzzle": "Alice, Bob and Carol each have one meeting, on Monday, Tuesday or Wednesday, "
    "each on a different day. Alice's isn't on Monday. Bob's is the day after Alice's. "
    "Which day is Carol's? Explain step by step.",
    # reworded Sep 30: without "in a transformer LLM", all four setups explained a generic key-value store
    "long_explanation": "Explain the KV cache in a transformer LLM to a new engineer in about 300 words.",
    "json_extraction": "Extract the name, company and meeting date from this message as JSON: "
    "\"Hi, this is Sam Ortiz from Acme Robotics. Can we meet on October 21 to review the pilot?\"",
}

SCENARIOS = tuple(PROMPTS)


_FINAL_ANSWER = (
    r"answer is\D{0,12}?(\d+)",
    r"(?:has|have|left with|still has|remain(?:s|ing)?)\D{0,25}?(\d+)\s+sheep",
    r"(\d+)\s+sheep\s+(?:left|remain)",
    r"sheep left\s*=\s*(\d+)",
)


def sheep_verdict(text: str) -> bool:
    """Correct iff the last stated answer is 9 ("the farmer has 9 sheep left", "the answer is 9").
    Falls back to the first sentence's last number when the answer isn't phrased either way.
    Markdown emphasis around the number ("**9** sheep") is stripped first."""
    text = re.sub(r"[*_`]+", "", text)
    hits = sorted((m.start(), m.group(1)) for p in _FINAL_ANSWER for m in re.finditer(p, text, flags=re.I))
    if hits:
        return hits[-1][1] == "9"
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    stated = re.findall(r"\d+", first)
    return bool(stated) and stated[-1] == "9"


_DAY = r"(monday|tuesday|wednesday)"
_FINAL_DAY = (
    r"carol'?s?\b[^.\n]{0,80}?\b" + _DAY,
    r"answer\b[^.\n]{0,20}?\b" + _DAY,
)


def puzzle_verdict(text: str) -> bool:
    """Correct iff the last stated day for Carol (or "the answer is ...") is Monday. The only
    solution is Alice on Tuesday, Bob on Wednesday, Carol on Monday."""
    hits = sorted((m.start(), m.group(1).lower()) for p in _FINAL_DAY for m in re.finditer(p, text, re.I))
    return bool(hits) and hits[-1][1] == "monday"


_EXPECTED_JSON = {"name": "sam ortiz", "company": "acme robotics", "date": "october 21"}


def json_verdict(text: str) -> bool:
    """Correct iff the answer holds a JSON object with exactly the stated values: name Sam Ortiz,
    company Acme Robotics, date October 21. A date with an invented year doesn't count."""
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            data = json.loads(match.group(0))
        except ValueError:
            continue
        found = {}
        for key, value in data.items():
            k = key.lower()
            field = next((f for f in ("company", "date", "name") if f in k), None)
            if field and isinstance(value, str):
                found[field] = " ".join(value.lower().split())
        if found == _EXPECTED_JSON:
            return True
    return False


GRADERS = {
    "complex_reasoning": sheep_verdict,
    "logic_puzzle": puzzle_verdict,
    "json_extraction": json_verdict,
}


def _verdict(scenario: str, text: str) -> str | None:
    grader = GRADERS.get(scenario)
    if grader is None:
        return None
    return "pass" if grader(text) else "fail"


def _active_quality_dir() -> Path:
    from app import tracks  # the track picks the folder; imported here to keep quality.py free of app state

    return tracks.active().quality_dir


def _find_capture(quality_dir: Path, key: str, scenario: str) -> Path | None:
    for vdir in VARIANT_DIRS.get(key, (key,)):
        for name in SCENARIO_ALIASES.get(scenario, (scenario,)):
            path = quality_dir / vdir / f"{name}.json"
            if path.is_file():
                return path
    return None


def _load_capture(quality_dir: Path, key: str, scenario: str) -> dict | None:
    """A capture for this setup and scenario, only if it answers the exact preset prompt."""
    path = _find_capture(quality_dir, key, scenario)
    if not path:
        return None
    try:
        data = json.loads(path.read_text())
        data["response_text"] = str(data["response_text"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning(f"Skipping bad capture {path}: {e}")
        return None
    if data.get("prompt") not in (None, PROMPTS.get(scenario)):
        logger.warning(f"Skipping {path}: it answers a different prompt")
        return None
    return data


def get_comparison(scenario: str, variants: list[str], quality_dir: Path | None = None) -> dict:
    quality_dir = quality_dir or _active_quality_dir()
    captured = {}
    prompt, temperature = None, None
    for key in variants:
        data = _load_capture(quality_dir, key, scenario)
        if data is None:
            continue
        text = data["response_text"]
        prompt = prompt or data.get("prompt")
        temperature = data.get("temperature") if temperature is None else temperature
        captured[key] = {
            "text": text,
            "model": data.get("model"),
            "usage": data.get("usage"),
            "verdict": _verdict(scenario, text),
        }

    if captured:
        return {"source": "captured", "prompt": prompt, "temperature": temperature, "responses": captured}
    return {"source": "not_captured", "prompt": PROMPTS[scenario], "temperature": None, "responses": {}}


def recorded_answer(key: str, scenario: str, quality_dir: Path | None = None) -> dict | None:
    """One setup's recorded answer with what was measured when it was recorded. The Sep 29
    captures have no timings, so those fields are None for them."""
    data = _load_capture(quality_dir or _active_quality_dir(), key, scenario)
    if data is None:
        return None
    return {
        "text": data["response_text"],
        "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
        "captured_at": data.get("captured_at"),
        "ttft_ms": data.get("ttft_ms"),
        "total_ms": data.get("total_ms"),
        "tokens_per_second": data.get("tokens_per_second"),
    }
