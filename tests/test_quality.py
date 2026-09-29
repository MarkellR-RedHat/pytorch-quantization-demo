"""Quality panel: captured output wins over illustrative text, and the sheep checker is right"""

import json

import pytest

from app.quality import ILLUSTRATIVE, get_comparison, sheep_verdict

VARIANTS = ["FP16", "INT4", "SPEC_DECODE"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The farmer has 9 sheep left.\n\nStep 1: 17 sheep.", True),
        ("The farmer has 8 sheep left.\n\nStep 1: 17 - 9 = 8.", False),
        ("Let's think. 17 - 9 = 8 ran away. The answer is 9.", True),
        ("There are 17 sheep. The answer is **8**.", False),
        ("No idea.", False),
    ],
)
def test_sheep_verdict(text, expected):
    assert sheep_verdict(text) is expected


def test_illustrative_when_nothing_captured(tmp_path):
    result = get_comparison("complex_reasoning", VARIANTS, tmp_path)
    assert result["source"] == "illustrative"
    assert result["temperature"] == 0
    verdicts = {k: v["verdict"] for k, v in result["responses"].items()}
    assert verdicts == {"FP16": "pass", "INT4": "fail", "SPEC_DECODE": "pass"}
    assert get_comparison("code_generation", VARIANTS, tmp_path)["responses"]["FP16"]["verdict"] is None


def test_spec_decode_illustration_matches_baseline():
    # Greedy speculative decoding returns the target model's exact tokens.
    for scenario in ILLUSTRATIVE.values():
        assert scenario["SPEC_DECODE"] == scenario["FP16"]


def test_captured_output_is_used(tmp_path):
    folder = tmp_path / "BF16"
    folder.mkdir()
    (folder / "sheep.json").write_text(json.dumps({
        "prompt": "A farmer has 17 sheep...",
        "temperature": 0,
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "response_text": "The farmer has 9 sheep left.",
        "usage": {"completion_tokens": 8},
        "raw": {},
    }))
    (tmp_path / "INT4").mkdir()
    (tmp_path / "INT4" / "complex_reasoning.json").write_text("not json")
    result = get_comparison("complex_reasoning", VARIANTS, tmp_path)
    assert result["source"] == "captured"
    assert result["temperature"] == 0
    assert set(result["responses"]) == {"FP16"}
    assert result["responses"]["FP16"] == {
        "text": "The farmer has 9 sheep left.",
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "usage": {"completion_tokens": 8},
        "verdict": "pass",
    }
