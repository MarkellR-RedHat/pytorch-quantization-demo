"""Side-by-side quality comparison: captured model output when available, illustrative text otherwise"""

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

_SHEEP_FP16 = (
    "The farmer has 9 sheep left.\n\n"
    "Step 1: The farmer starts with 17 sheep.\n"
    "Step 2: The phrase 'all but 9' means every sheep except 9 ran away.\n"
    "Step 3: So 17 - 9 = 8 sheep ran away.\n"
    "Step 4: The 9 sheep that didn't run away are still with the farmer.\n\n"
    "The answer is 9 sheep. The tricky part is that 'all but 9' means 9 remain, not that 9 ran away."
)
_CODE_FP16 = (
    "```python\ndef second_largest(nums):\n    if len(nums) < 2:\n"
    "        raise ValueError(\"Need at least two distinct values\")\n"
    "    unique = sorted(set(nums), reverse=True)\n    if len(unique) < 2:\n"
    "        raise ValueError(\"Need at least two distinct values\")\n    return unique[1]\n```\n\n"
    "This handles duplicates by using `set()` first, then sorts in descending order. It raises an error "
    "if there aren't at least two distinct values in the list."
)
_SUMMARY_FP16 = (
    "- **Speed vs. Quality**: Lower precision (INT8/INT4) increases throughput and reduces latency, but "
    "aggressive quantization can degrade output quality on complex reasoning, nuanced instructions, and "
    "multi-step tasks.\n"
    "- **Cost vs. Capability**: Quantized models need less GPU memory and compute, which cuts infrastructure "
    "cost, but may require additional validation pipelines to catch quality regressions before they "
    "affect users.\n"
    "- **Simplicity vs. Optimization**: Full precision models work out of the box, while quantized "
    "deployments require careful calibration, evaluation datasets, and ongoing monitoring."
)

# Illustrative only. With greedy decoding (temperature 0) speculative decoding returns exactly the
# target model's tokens, so the spec decode text is identical to the baseline text by construction.
ILLUSTRATIVE = {
    "complex_reasoning": {
        "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? "
        "Explain your reasoning step by step.",
        "FP16": _SHEEP_FP16,
        "INT4": "The farmer has 8 sheep left.\n\nStep 1: The farmer starts with 17 sheep.\n"
        "Step 2: 9 sheep run away.\nStep 3: 17 - 9 = 8 sheep remain.\n\nThe answer is 8 sheep.",
        "SPEC_DECODE": _SHEEP_FP16,
    },
    "code_generation": {
        "prompt": "Write a Python function that returns the second largest number in a list. "
        "Handle edge cases.",
        "FP16": _CODE_FP16,
        "INT4": "```python\ndef second_largest(nums):\n    nums.sort()\n    return nums[-2]\n```\n\n"
        "Sort the list and return the second to last element.",
        "SPEC_DECODE": _CODE_FP16,
    },
    "summarization": {
        "prompt": "Summarize the key trade-offs of model quantization for production LLM deployments "
        "in 3 bullet points.",
        "FP16": _SUMMARY_FP16,
        "INT4": "- Speed vs Quality: Smaller models run faster but may lose accuracy\n"
        "- Cost vs Capability: Less GPU memory needed, lower costs\n"
        "- Simplicity vs Optimization: Quantization requires extra setup",
        "SPEC_DECODE": _SUMMARY_FP16,
    },
}

SCENARIOS = tuple(ILLUSTRATIVE)


def sheep_verdict(text: str) -> bool:
    """Correct iff the stated answer is 9: 'answer is N' wins, else the first sentence's last number."""
    stated = re.findall(r"answer is\D{0,12}?(\d+)", text, flags=re.IGNORECASE)
    if not stated:
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

    data = ILLUSTRATIVE[scenario]
    return {
        "source": "illustrative",
        "prompt": data["prompt"],
        "temperature": 0,
        "responses": {
            key: {"text": data[key], "model": None, "usage": None, "verdict": _verdict(scenario, data[key])}
            for key in variants
            if key in data
        },
    }
