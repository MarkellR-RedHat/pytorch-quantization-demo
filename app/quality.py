"""Side-by-side quality comparison from captured model output only"""

import json
import logging
import re
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

SCENARIO_ALIASES = {
    "complex_reasoning": ("complex_reasoning", "reasoning", "sheep"),
    "code_generation": ("code_generation", "code"),
    "summarization": ("summarization", "summary"),
}

VARIANT_DIRS = {
    "FP16": ("FP16", "BF16", "fp16", "bf16"),
    "FP8": ("FP8", "fp8"),
    "INT4": ("INT4", "INT4_AWQ", "int4", "int4_awq"),
    "SPEC_DECODE": ("SPEC_DECODE", "SPEC", "spec_decode", "spec"),
}

# The exact prompts behind the Sep 29 captures in quality/<VARIANT>/<scenario>.json. There is no
# made-up fallback text: a setup with no capture shows no answer. The spec decode captures match
# the BF16 ones on the sheep riddle only; the code and summary answers diverge partway through.
PROMPTS = {
    "complex_reasoning": "A farmer has 17 sheep. All but 9 run away. "
    "How many sheep does the farmer have left? Explain your reasoning step by step.",
    "code_generation": "Write a Python function that returns the second largest number in a list. "
    "Handle edge cases.",
    "summarization": "Summarize the key trade-offs of model quantization for production LLM deployments "
    "in 3 bullet points.",
}

SCENARIOS = tuple(PROMPTS)


_FINAL_ANSWER = (
    r"answer is\D{0,12}?(\d+)",
    r"(?:has|have|left with|still has|remain(?:s|ing)?)\D{0,25}?(\d+)\s+sheep",
    r"(\d+)\s+sheep\s+(?:left|remain)",
)


def sheep_verdict(text: str) -> bool:
    """Correct iff the last stated answer is 9 ("the farmer has 9 sheep left", "the answer is 9").
    Falls back to the first sentence's last number when the answer isn't phrased either way."""
    hits = sorted((m.start(), m.group(1)) for p in _FINAL_ANSWER for m in re.finditer(p, text, flags=re.I))
    if hits:
        return hits[-1][1] == "9"
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    stated = re.findall(r"\d+", first)
    return bool(stated) and stated[-1] == "9"


def _verdict(scenario: str, text: str) -> str | None:
    if scenario != "complex_reasoning":
        return None
    return "pass" if sheep_verdict(text) else "fail"


def _find_capture(quality_dir: Path, key: str, scenario: str) -> Path | None:
    for vdir in VARIANT_DIRS.get(key, (key,)):
        for name in SCENARIO_ALIASES[scenario]:
            path = quality_dir / vdir / f"{name}.json"
            if path.is_file():
                return path
    return None


def get_comparison(scenario: str, variants: list[str], quality_dir: Path | None = None) -> dict:
    quality_dir = quality_dir or settings.resolve(settings.quality_dir)
    captured = {}
    prompt, temperature = None, None
    for key in variants:
        path = _find_capture(quality_dir, key, scenario)
        if not path:
            continue
        try:
            data = json.loads(path.read_text())
            text = str(data["response_text"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"Skipping bad capture {path}: {e}")
            continue
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
