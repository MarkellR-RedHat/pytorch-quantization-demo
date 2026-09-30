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

# BF16 = the hardest questions, INT4 = everyday chat and easy questions, Spec = latency-sensitive long answers
LANES = {
    "reasoning": "FP16",
    "puzzle": "FP16",
    "summary": "INT4",
    "decline": "INT4",
    "fact": "INT4",
    "json": "INT4",
    "code": "SPEC_DECODE",
    "explain": "SPEC_DECODE",
}


def route_source() -> str:
    js = (ROOT / "static" / "js" / "presenter.js").read_text()
    match = re.search(r"\n    function route\(q\) \{\n.*?\n    \}\n", js, re.S)
    assert match, "route() not found in presenter.js"
    return match.group(0)


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_every_preset_lands_on_its_lane():
    assert set(LANES) == set(PRESETS)
    prompts = {key: PROMPTS[scenario] for key, (scenario, _label) in PRESETS.items()}
    src = route_source() + f"\nconst P = {json.dumps(prompts)};\n" + (
        "const keys = Object.entries(P).map(([k, q]) => [k, route(q).key]);\n"
        "console.log(JSON.stringify(Object.fromEntries(keys)));"
    )
    out = subprocess.run([node, "-e", src], capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(out.stdout) == LANES
