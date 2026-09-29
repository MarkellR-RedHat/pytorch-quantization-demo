"""The arena's claims, checked: the browser flies the same networks as PyTorch, and the
numbers quoted on screen are what the exported policies actually do."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
POLICY = json.loads((ROOT / "static" / "arena" / "policy.json").read_text())
GOLDEN = ROOT / "tests" / "data" / "arena_golden.json"

PARITY_JS = """
const A = require(process.argv[1] + '/static/js/arena-core.js');
const P = require(process.argv[1] + '/static/arena/policy.json');
const G = require(process.argv[1] + '/tests/data/arena_golden.json');
A.configure(P.world);
const out = {};
for (const [k, g] of Object.entries(G)) {
  const L = P.variants[k].layers, c = new A.Course(g.seed);
  let y = 250, vy = 0, flips = 0, maxd = 0;
  for (let f = 0; f < g.y.length; f++) {
    const a = A.act(L, y, vy, c.ahead(2));
    if (a !== g.actions[f]) flips++;
    [y, vy] = A.birdStep(y, vy, a); c.step();
    maxd = Math.max(maxd, Math.abs(y - g.y[f]));
  }
  out[k] = { frames: g.y.length, flips, maxd };
}
console.log(JSON.stringify(out));
"""

SPEC_JS = """
const A = require(process.argv[1] + '/static/js/arena-core.js');
const P = require(process.argv[1] + '/static/arena/policy.json');
A.configure(P.world);
const T = P.variants.BF16.layers, D = P.draft.layers, K = 4;
const ca = new A.Course(77), cb = new A.Course(77);
let ya = 250, va = 0, yb = 250, vb = 0, frames = 0, passes = 0, accepted = 0, maxd = 0;
while (frames < 3000) {
  const r = A.speculate(T, D, yb, vb, cb.pipes, K);
  passes++; accepted += r.accepted;
  for (const a of r.commit) {
    const ar = A.act(T, ya, va, ca.ahead(2));
    [ya, va] = A.birdStep(ya, va, ar); ca.step();
    [yb, vb] = A.birdStep(yb, vb, a); cb.step();
    frames++; maxd = Math.max(maxd, Math.abs(ya - yb));
  }
}
console.log(JSON.stringify({ acceptance: accepted / (passes * K), fpp: frames / passes, maxd }));
"""

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="node is not installed")


def run_js(src: str) -> dict:
    out = subprocess.run(
        [node, "-e", src, str(ROOT)], capture_output=True, text=True, timeout=120, check=True
    )
    return json.loads(out.stdout)


@needs_node
def test_browser_flies_the_same_trajectories_as_pytorch():
    for key, r in run_js(PARITY_JS).items():
        assert r["frames"] > 500, key
        assert r["flips"] == 0, f"{key}: browser chose a different action than Python"
        assert r["maxd"] == 0, f"{key}: browser path drifted from Python by {r['maxd']} px"


@needs_node
def test_speculative_decoding_is_lossless_in_the_browser():
    r = run_js(SPEC_JS)
    assert r["maxd"] == 0, "spec decode bird left the BF16 path"
    assert 0.5 < r["acceptance"] < 1.0
    assert r["fpp"] > 1.0


def test_policy_has_every_variant_the_dashboard_draws():
    assert set(POLICY["variants"]) == {"BF16", "FP8", "INT4_RTN", "INT4_AWQ"}
    assert POLICY["draft"]["layers"]
    shapes = [len(layer["W"][0]) for layer in POLICY["variants"]["BF16"]["layers"]]
    assert shapes[0] == len(POLICY["features"])


def test_quoted_results_match_the_exported_evaluation():
    """The dashboard and slides say: BF16, FP8 and AWQ clear every hard gap, RTN does not."""
    ev = {k: v["eval"] for k, v in POLICY["variants"].items()}
    for key in ("BF16", "FP8", "INT4_AWQ"):
        assert ev[key]["hard"]["cleared"] == ev[key]["hard"]["attempted"], key
        assert ev[key]["easy"]["cleared"] == ev[key]["easy"]["attempted"], key
    rtn = ev["INT4_RTN"]
    assert rtn["easy"]["cleared"] == rtn["easy"]["attempted"]
    assert rtn["hard"]["cleared"] < 0.8 * rtn["hard"]["attempted"]
    assert POLICY["spec"]["max_path_difference_px"] == 0


def test_int4_weights_really_have_at_most_16_levels_per_group():
    group = POLICY["group_size"]
    for key in ("INT4_RTN",):
        for layer in POLICY["variants"][key]["layers"]:
            for row in layer["W"]:
                for j in range(0, len(row), group):
                    assert len(set(row[j : j + group])) <= 16, key


def test_param_count_on_screen_matches_the_network():
    layers = POLICY["variants"]["BF16"]["layers"]
    n = sum(len(layer["W"]) * len(layer["W"][0]) + len(layer["b"]) for layer in layers)
    assert n == POLICY["params"]["target"] == 4673
