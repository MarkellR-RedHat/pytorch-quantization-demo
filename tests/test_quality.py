"""Quality panel: only captured output is shown, and the sheep checker is right"""

import json

import pytest

from app.quality import PROMPTS, get_comparison, sheep_verdict

VARIANTS = ["FP16", "INT4", "SPEC_DECODE"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The farmer has 9 sheep left.\n\nStep 1: 17 sheep.", True),
        ("The farmer has 8 sheep left.\n\nStep 1: 17 - 9 = 8.", False),
        ("Let's think. 17 - 9 = 8 ran away. The answer is 9.", True),
        ("There are 17 sheep. The answer is **8**.", False),
        ("No idea.", False),
        ("Step 1: The farmer has 17 sheep.\nStep 2: 8 run away.\n\nSo the farmer has 9 sheep left.", True),
        ("The farmer starts with 17 sheep. 9 run away, so the farmer has 8 sheep left.", False),
    ],
)
def test_sheep_verdict(text, expected):
    assert sheep_verdict(text) is expected


def test_nothing_is_shown_when_nothing_was_captured(tmp_path):
    for scenario in PROMPTS:
        result = get_comparison(scenario, VARIANTS, tmp_path)
        assert result["source"] == "not_captured"
        assert result["responses"] == {}
        assert result["prompt"] == PROMPTS[scenario]


def test_prompts_match_the_captures():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "quality"
    for variant in VARIANTS:
        for scenario, prompt in PROMPTS.items():
            assert json.loads((root / variant / f"{scenario}.json").read_text())["prompt"] == prompt


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


def test_every_captured_sheep_answer_grades_correct():
    """The Sep 29 captures: greedy and 20 sampled answers per setup, all stating 9."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "quality"
    for variant in VARIANTS:
        greedy = json.loads((root / variant / "complex_reasoning.json").read_text())["response_text"]
        sampled = root / variant / "complex_reasoning_samples_t0.7.json"
        samples = json.loads(sampled.read_text())["responses"]
        assert sheep_verdict(greedy), variant
        assert sum(sheep_verdict(t) for t in samples) == len(samples) == 20, variant
