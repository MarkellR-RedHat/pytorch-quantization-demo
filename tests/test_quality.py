"""Quality panel: only captured output is shown, and the sheep checker is right"""

import json

import pytest

from app.quality import PROMPTS, get_comparison, json_verdict, puzzle_verdict, sheep_verdict

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


def test_every_capture_answers_its_preset_prompt():
    """Presets can be uncaptured (they show "not captured"), but a capture that exists must answer the
    exact preset prompt, and the Sep 29 three stay captured for the three setups."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "quality"
    for path in root.glob("*/*.json"):
        if path.stem in PROMPTS:
            assert json.loads(path.read_text())["prompt"] == PROMPTS[path.stem], path
    for variant in VARIANTS:
        for scenario in ("complex_reasoning", "code_generation", "summarization"):
            assert (root / variant / f"{scenario}.json").is_file(), (variant, scenario)


def test_uncaptured_preset_reports_not_captured():
    result = get_comparison("logic_puzzle", VARIANTS)
    if not result["responses"]:
        assert result["source"] == "not_captured"
        assert result["prompt"] == PROMPTS["logic_puzzle"]


def test_a_capture_of_a_different_prompt_is_ignored(tmp_path):
    (tmp_path / "FP16").mkdir()
    (tmp_path / "FP16" / "quick_fact.json").write_text(json.dumps({
        "prompt": "What is the capital of Australia?", "response_text": "Canberra.",
    }))
    assert get_comparison("quick_fact", ["FP16"], tmp_path)["source"] == "not_captured"


def test_capture_script_sends_the_same_prompts():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "capture_presets.py"
    spec = importlib.util.spec_from_file_location("capture_presets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PROMPTS == PROMPTS
    assert module.MAX_TOKENS == 1024


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Alice is Tuesday, Bob is Wednesday.\n\nCarol's meeting is on Monday.", True),
        ("If Carol's were on Tuesday, Alice... So the answer is Monday.", True),
        ("Step 1: Carol's can't be Wednesday. Therefore Carol's meeting is on **Monday**.", True),
        ("Carol's meeting is on Wednesday.", False),
        ("Alice is on Tuesday and Bob on Wednesday.", False),
    ],
)
def test_puzzle_verdict(text, expected):
    assert puzzle_verdict(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"name": "Sam Ortiz", "company": "Acme Robotics", "date": "October 21"}\n```', True),
        ('{"sender_name": "Sam Ortiz", "company_name": "Acme Robotics", "date": "october 21"}', True),
        ('{"name": "Sam Ortiz", "company": "Acme Robotics", "date": "2026-10-21"}', False),
        ('{"name": "Sam", "company": "Acme Robotics", "date": "October 21"}', False),
        ("Name: Sam Ortiz, company: Acme Robotics, date: October 21", False),
    ],
)
def test_json_verdict(text, expected):
    assert json_verdict(text) is expected


def test_captured_output_is_used(tmp_path):
    folder = tmp_path / "BF16"
    folder.mkdir()
    (folder / "sheep.json").write_text(json.dumps({
        "prompt": PROMPTS["complex_reasoning"],
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
