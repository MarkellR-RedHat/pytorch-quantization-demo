"""The example router on the Ask scene sends each preset to the lane the talk gives it (slides 6 and 8)"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.ask import PRESETS
from app.quality import PROMPTS

ROOT = Path(__file__).resolve().parent.parent
node = shutil.which("node")

# BF16 = the hardest questions, the everyday lane = FP8 on Hopper when it's deployed (INT4 where 73 GB of
# weights won't fit), Spec = latency-sensitive long answers
LANES = {
    "reasoning": "BF16",
    "puzzle": "BF16",
    "summary": "EVERYDAY",
    "decline": "EVERYDAY",
    "fact": "EVERYDAY",
    "json": "EVERYDAY",
    "code": "SPEC_DECODE",
    "explain": "SPEC_DECODE",
}


def route_source() -> str:
    js = (ROOT / "static" / "js" / "presenter.js").read_text()
    match = re.search(r"\n    function route\(q, hasFP8\) \{\n.*?\n    \}\n", js, re.S)
    assert match, "route() not found in presenter.js"
    return match.group(0)


def lanes(has_fp8: bool) -> dict:
    prompts = {key: PROMPTS[scenario] for key, (scenario, _label) in PRESETS.items()}
    src = route_source() + f"\nconst P = {json.dumps(prompts)};\n" + (
        f"const keys = Object.entries(P).map(([k, q]) => [k, route(q, {json.dumps(has_fp8)}).key]);\n"
        "console.log(JSON.stringify(Object.fromEntries(keys)));"
    )
    out = subprocess.run([node, "-e", src], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(node is None, reason="node is not installed")
@pytest.mark.parametrize("has_fp8", [True, False])
def test_every_preset_lands_on_its_lane(has_fp8):
    assert set(LANES) == set(PRESETS)
    everyday = "FP8" if has_fp8 else "INT4"
    assert lanes(has_fp8) == {k: (everyday if v == "EVERYDAY" else v) for k, v in LANES.items()}


def js_function(name: str) -> str:
    js = (ROOT / "static" / "js" / "presenter.js").read_text()
    match = re.search(rf"\n    function {name}\(\w+\) \{{\n.*?\n    \}}\n", js, re.S)
    assert match, f"{name}() not found in presenter.js"
    return match.group(0)


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_badge_names_live_and_recorded_setups():
    def badge(cfg):
        src = js_function("badgeState") + f"\nconsole.log(JSON.stringify(badgeState({json.dumps(cfg)})));"
        out = subprocess.run([node, "-e", src], capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    three = [{"label": "BF16", "mode": "live"}, {"label": "INT4 (Red Hat W4A16)", "mode": "live"},
             {"label": "Spec Decode", "mode": "live"}]
    assert badge({"mode": "live", "variants": three}) == {"kind": "live", "text": "Live models"}
    three[2]["mode"] = "recorded"
    assert badge({"mode": "live", "variants": three}) == {
        "kind": "mixed", "text": "Live: BF16, INT4 (Red Hat W4A16) · Recorded: Spec Decode"}
    assert badge({"mode": "simulated", "variants": three}) == {"kind": "sim", "text": "Replay"}
