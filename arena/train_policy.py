"""Train and quantize the policy networks that fly the birds in the Quantization Arena.

Every bird on the presenter screen is flown by a real neural network. This script:

1. Trains a 6-64-64-1 MLP (the "target") in PyTorch by imitation learning (DAgger)
   from a rule-based expert, plus a tiny 6-4-1 "draft" model for speculative decoding.
2. Quantizes the target's weights four ways: BF16, FP8 (E4M3, per-channel), INT4
   round-to-nearest (group size 16), and INT4 with activation-aware scaling (AWQ-style,
   group size 16). Biases stay in high precision, as in W4A16 serving formats.
3. Evaluates every variant on held-out courses and exports the dequantized weights
   to static/arena/policy.json, which the browser runs as-is.

The physics here must stay identical to static/js/arena-core.js. tests/test_arena.py
checks that the browser code reproduces the Python trajectories frame for frame.

Run:  pip install -r arena/requirements.txt && python arena/train_policy.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "arena" / "policy.json"
GOLDEN = ROOT / "tests" / "data" / "arena_golden.json"

# World, in logical pixels. The canvas scales this to whatever size it is drawn at.
W, H = 960.0, 540.0
GROUND = 500.0
BX = 220.0
BW, BH = 34.0, 26.0
G, FLAP, VMAX = 0.45, -7.4, 10.0
SPEED = 4.0
PIPE_W = 78.0
SPACING = 330.0
GAP_EASY, GAP_HARD = 170.0, 112.0
MARGIN = 70.0
HARD_FRAC = 0.3
SPEC_K = 4
GROUP = 16

# Feature 0 is left in raw pixels on purpose. It plays the role of the outlier
# activation channels that LLMs develop (LLM.int8(), SmoothQuant, AWQ): a few input
# dimensions with a much larger range than the rest. With every feature normalized,
# all of the INT4 schemes fly perfectly, which is the lesson in miniature.
FEATURES = [
    "gap bottom - bird y (raw px, outlier channel)",
    "bird y - gap top (/100)",
    "vertical velocity (/10)",
    "distance to gap (/300)",
    "next gap bottom - bird y (/100)",
    "bird y - next gap top (/100)",
]


class Mulberry32:
    """Tiny seeded PRNG, bit-identical to the JavaScript version in arena-core.js."""

    def __init__(self, seed: int):
        self.a = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0


class Course:
    def __init__(self, seed: int, hard_frac: float = HARD_FRAC):
        self.rng = Mulberry32(seed)
        self.hard_frac = hard_frac
        self.next_id = 0
        self.pipes: list[dict] = [self.make(W + 40.0)]

    def make(self, x: float, hard: bool | None = None) -> dict:
        r1 = self.rng.random()
        r2 = self.rng.random()
        if hard is None:
            hard = r1 < self.hard_frac
        gap = GAP_HARD if hard else GAP_EASY
        lo, hi = MARGIN + gap / 2, GROUND - MARGIN - gap / 2
        c = lo + r2 * (hi - lo)
        self.next_id += 1
        return {"id": self.next_id, "x": x, "top": c - gap / 2, "bot": c + gap / 2, "hard": hard}

    def step(self) -> None:
        for p in self.pipes:
            p["x"] -= SPEED
        self.pipes = [p for p in self.pipes if p["x"] + PIPE_W > -10]
        if self.pipes[-1]["x"] < W - SPACING:
            self.pipes.append(self.make(self.pipes[-1]["x"] + SPACING))

    def ahead(self, n: int = 2) -> list[dict]:
        return [p for p in self.pipes if p["x"] + PIPE_W >= BX - BW / 2][:n]


def bird_step(y: float, vy: float, flap: int) -> tuple[float, float]:
    vy = FLAP if flap else min(vy + G, VMAX)
    return y + vy, vy


def dead(y: float, pipes: list[dict]) -> bool:
    if y - BH / 2 < 0 or y + BH / 2 > GROUND:
        return True
    for p in pipes:
        if BX + BW / 2 > p["x"] and BX - BW / 2 < p["x"] + PIPE_W:
            if y - BH / 2 < p["top"] or y + BH / 2 > p["bot"]:
                return True
    return False


def features(y: float, vy: float, pipes_ahead: list[dict]) -> list[float]:
    p0 = pipes_ahead[0]
    p1 = pipes_ahead[1] if len(pipes_ahead) > 1 else p0
    return [
        p0["bot"] - y,
        (y - p0["top"]) / 100.0,
        vy / VMAX,
        (p0["x"] - BX) / 300.0,
        (p1["bot"] - y) / 100.0,
        (y - p1["top"]) / 100.0,
    ]


def expert(y: float, vy: float, course: Course, margin: float = 10.0) -> int:
    p = course.ahead(1)[0]
    return 1 if (y + vy > p["bot"] - margin - BH / 2 and vy >= -1.0) else 0


# ---------------------------------------------------------------- training


def mlp(hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = len(FEATURES)
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def rollout(net, seed, frames, xs, ys, beta, rng):
    course = Course(seed)
    y, vy = 250.0, 0.0
    for f in range(frames):
        feat = features(y, vy, course.ahead(2))
        a_exp = expert(y, vy, course)
        xs.append(feat)
        ys.append(a_exp)
        if net is None or rng.random() < beta:
            a = a_exp
        else:
            with torch.no_grad():
                a = int(net(torch.tensor([feat], dtype=torch.float32)).item() > 0)
        y, vy = bird_step(y, vy, a)
        course.step()
        if dead(y, course.pipes):
            course = Course(seed * 1000 + f)
            y, vy = 250.0, 0.0


def fit(net, xs, ys, epochs=60):
    x = torch.tensor(xs, dtype=torch.float32)
    t = torch.tensor(ys, dtype=torch.float32)[:, None]
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 512):
            idx = perm[i : i + 512]
            opt.zero_grad()
            loss_fn(net(x[idx]), t[idx]).backward()
            opt.step()
    with torch.no_grad():
        return ((net(x) > 0).float() == t).float().mean().item()


def train(hidden, iters, tag):
    net = mlp(hidden)
    rng = random.Random(7)
    xs, ys = [], []
    for it in range(iters):
        beta = 1.0 if it == 0 else 0.3 / it
        for s in range(6):
            rollout(None if it == 0 else net, it * 100 + s, 2500, xs, ys, beta, rng)
        acc = fit(net, xs, ys)
        print(f"  {tag}: DAgger round {it}, {len(xs)} states, agreement with expert {acc:.3f}")
    return net


# ---------------------------------------------------------------- quantization


def to_bf16(w):
    return torch.tensor(w, dtype=torch.float32).to(torch.bfloat16).double().numpy()


def to_fp8(w):
    w = np.asarray(w)
    s = np.abs(w).max(1, keepdims=True) / 448.0
    s[s == 0] = 1
    q = torch.tensor(w / s, dtype=torch.float32).to(torch.float8_e4m3fn).double().numpy()
    return q * s


def int4_group(w, g=GROUP):
    """Asymmetric round-to-nearest INT4 with one scale and zero point per group of g inputs."""
    w = np.asarray(w)
    out = np.empty_like(w)
    for j in range(0, w.shape[1], g):
        blk = w[:, j : j + g]
        mn, mx = blk.min(1, keepdims=True), blk.max(1, keepdims=True)
        s = (mx - mn) / 15.0
        s[s == 0] = 1e-8
        z = np.round(-mn / s)
        out[:, j : j + g] = (np.clip(np.round(blk / s) + z, 0, 15) - z) * s
    return out


def layer_inputs(layers, xs):
    acts, h = [], np.asarray(xs, dtype=np.float64)
    for i, (w, b) in enumerate(layers):
        acts.append(h)
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return acts


def awq(layers, calib, g=GROUP, grid=20):
    """AWQ-style: scale each input channel by mean|x|^alpha before quantizing, fold the
    scale back out, and grid-search alpha per layer to minimize the layer's output error."""
    out, alphas = [], []
    for (w, b), x in zip(layers, layer_inputs(layers, calib), strict=True):
        xm = np.abs(x).mean(0) + 1e-8
        ref = x @ w.T
        best = None
        for k in range(grid + 1):
            alpha = k / grid
            s = xm**alpha
            s = s / math.sqrt(s.max() * s.min())
            wq = int4_group(w * s, g=min(g, w.shape[1])) / s
            err = float(((x @ wq.T - ref) ** 2).mean())
            if best is None or err < best[0]:
                best = (err, wq, alpha)
        out.append((best[1], b))
        alphas.append(best[2])
    return out, alphas


def forward(layers, x):
    h = np.asarray(x, dtype=np.float64)
    for i, (w, b) in enumerate(layers):
        h = w @ h + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return float(h[0])


# ---------------------------------------------------------------- evaluation


HELD_OUT_SEEDS = tuple(range(500, 520))


def evaluate(layers, seeds=HELD_OUT_SEEDS, frames=3000):
    """Fly each course; on a crash, respawn in the next gap like the browser does."""
    stats = {"easy": [0, 0], "hard": [0, 0]}  # [cleared, attempted]
    for seed in seeds:
        course = Course(seed)
        y, vy = 250.0, 0.0
        scored = set()
        for _ in range(frames):
            a = int(forward(layers, features(y, vy, course.ahead(2))) > 0)
            y, vy = bird_step(y, vy, a)
            course.step()
            for p in course.pipes:
                if p["x"] + PIPE_W < BX - BW / 2 and p["id"] not in scored:
                    scored.add(p["id"])
                    k = "hard" if p["hard"] else "easy"
                    stats[k][0] += 1
                    stats[k][1] += 1
            if dead(y, course.pipes):
                here = [p for p in course.ahead(2) if p["x"] <= BX + BW / 2]
                for p in here:
                    if p["id"] not in scored:
                        scored.add(p["id"])
                        stats["hard" if p["hard"] else "easy"][1] += 1
                nxt = [p for p in course.ahead(2) if p["id"] not in scored]
                y = (nxt[0]["top"] + nxt[0]["bot"]) / 2 if nxt else GROUND / 2
                vy = 0.0
                # the browser gives a respawned bird a grace period; mirror that by skipping this gap
                for p in here:
                    scored.add(p["id"])
    return {k: {"cleared": v[0], "attempted": v[1]} for k, v in stats.items()}


def spec_check(target, draft, seed=77, frames=4000):
    """Greedy speculative decoding: the draft proposes K actions along its own imagined
    rollout, the target checks all K states in one batched pass, and we keep the agreeing
    prefix plus the target's own action at the first disagreement (or a bonus action)."""
    ca, cb = Course(seed), Course(seed)
    ya = yb = 250.0
    va = vb = 0.0
    proposed = accepted = passes = done = 0
    max_diff = 0.0
    while done < frames:
        pipes = [dict(p) for p in cb.pipes]
        ys, vs = yb, vb
        states, acts = [], []
        for _ in range(SPEC_K):
            ahead = [p for p in pipes if p["x"] + PIPE_W >= BX - BW / 2][:2]
            states.append((ys, vs, [dict(p) for p in ahead]))
            a = int(forward(draft, features(ys, vs, ahead)) > 0)
            acts.append(a)
            ys, vs = bird_step(ys, vs, a)
            for p in pipes:
                p["x"] -= SPEED
        ahead = [p for p in pipes if p["x"] + PIPE_W >= BX - BW / 2][:2]
        states.append((ys, vs, ahead))
        verdict = [int(forward(target, features(s[0], s[1], s[2])) > 0) for s in states]
        passes += 1
        n = 0
        while n < SPEC_K and acts[n] == verdict[n]:
            n += 1
        proposed += SPEC_K
        accepted += n
        commit = acts[:n] + [verdict[n]]
        for a in commit:
            a_ref = int(forward(target, features(ya, va, ca.ahead(2))) > 0)
            ya, va = bird_step(ya, va, a_ref)
            ca.step()
            yb, vb = bird_step(yb, vb, a)
            cb.step()
            done += 1
            max_diff = max(max_diff, abs(ya - yb))
    return {
        "k": SPEC_K,
        "acceptance_rate": accepted / proposed,
        "frames_per_target_pass": done / passes,
        "max_path_difference_px": max_diff,
    }


def golden(layers, seed=123, frames=1500):
    course = Course(seed)
    y, vy = 250.0, 0.0
    ys, acts = [], []
    for _ in range(frames):
        a = int(forward(layers, features(y, vy, course.ahead(2))) > 0)
        y, vy = bird_step(y, vy, a)
        course.step()
        ys.append(y)
        acts.append(a)
        if dead(y, course.pipes):
            break
    return {"seed": seed, "y": ys, "actions": acts}


def bytes_for(layers, bits, per_group=None):
    n_w = sum(w.size for w, _ in layers)
    n_b = sum(b.size for _, b in layers)
    extra = 0
    if per_group:
        # one fp16 scale and one fp16 zero point per group
        extra = sum(w.shape[0] * math.ceil(w.shape[1] / per_group) for w, _ in layers) * 4
    return int(n_w * bits / 8 + n_b * 2 + extra)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    print("Training target policy (6-64-64-1)")
    target_net = train([64, 64], iters=6, tag="target")
    print("Training draft policy (6-4-1)")
    torch.manual_seed(1)
    draft_net = train([4], iters=1, tag="draft")

    def raw(net):
        return [(m.weight.detach().double().numpy(), m.bias.detach().double().numpy())
                for m in net if isinstance(m, nn.Linear)]

    t_fp32, d_fp32 = raw(target_net), raw(draft_net)

    calib = []
    for s in range(3):
        course = Course(900 + s)
        y, vy = 250.0, 0.0
        for f in range(1500):
            calib.append(features(y, vy, course.ahead(2)))
            y, vy = bird_step(y, vy, expert(y, vy, course))
            course.step()
            if dead(y, course.pipes):
                course, y, vy = Course(9000 + f), 250.0, 0.0

    awq_layers, alphas = awq(t_fp32, calib)
    variants = {
        "BF16": ([(to_bf16(w), b) for w, b in t_fp32], 16, None),
        "FP8": ([(to_fp8(w), b) for w, b in t_fp32], 8, None),
        "INT4_RTN": ([(int4_group(w), b) for w, b in t_fp32], 4, GROUP),
        "INT4_AWQ": (awq_layers, 4, GROUP),
    }
    draft = [(to_bf16(w), b) for w, b in d_fp32]

    print("Evaluating on 20 held-out courses")
    out = {"version": 1, "world": {
        "W": W, "H": H, "GROUND": GROUND, "BX": BX, "BW": BW, "BH": BH, "G": G, "FLAP": FLAP,
        "VMAX": VMAX, "SPEED": SPEED, "PIPE_W": PIPE_W, "SPACING": SPACING, "GAP_EASY": GAP_EASY,
        "GAP_HARD": GAP_HARD, "MARGIN": MARGIN, "HARD_FRAC": HARD_FRAC},
        "features": FEATURES, "group_size": GROUP, "awq_alpha_per_layer": alphas, "variants": {}}
    for key, (layers, bits, grp) in variants.items():
        ev = evaluate(layers)
        out["variants"][key] = {
            "bits": bits,
            "weight_bytes": bytes_for(layers, bits, grp),
            "layers": [{"W": w.tolist(), "b": b.tolist()} for w, b in layers],
            "eval": ev,
        }
        e, h = ev["easy"], ev["hard"]
        print(f"  {key:9s} easy {e['cleared']}/{e['attempted']}  hard {h['cleared']}/{h['attempted']}")
    out["draft"] = {"bits": 16, "weight_bytes": bytes_for(draft, 16),
                    "layers": [{"W": w.tolist(), "b": b.tolist()} for w, b in draft]}
    out["spec"] = spec_check(variants["BF16"][0], draft)
    print(f"  SPEC      {out['spec']}")
    out["params"] = {"target": int(sum(w.size + b.size for w, b in t_fp32)),
                     "draft": int(sum(w.size + b.size for w, b in d_fp32))}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps({k: golden(v[0]) for k, v in variants.items()}))
    print(f"Wrote {OUT.relative_to(ROOT)} and {GOLDEN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
