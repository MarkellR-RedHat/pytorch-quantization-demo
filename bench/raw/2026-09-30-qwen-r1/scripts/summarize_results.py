#!/usr/bin/env python3
"""Turn a results folder (Llama round 2, or the Qwen track) into the numbers the slides use.

Reads, if present (bench/ or sweeps/, eval/ or evals/, either spelling works):
  <root>/bench/<VARIANT>/single-t0.json, single-t0.7.json    (vllm bench serve, concurrency 1)
  <root>/bench/<VARIANT>/c<N>.json, sharegpt-c<N>.json         (vllm bench serve sweep)
  <root>/eval/<VARIANT>/**/results_*.json                      (lm_eval, gsm8k_cot_llama)
  <root>/eval/<VARIANT>-mmlu_pro/**/results_*.json             (lm_eval, mmlu_pro)
  <root>/logs/spec-metrics-{before,after}-t<T>[-k<K>].txt      (vLLM /metrics snapshots; -k is spec tokens)
  <root>/logs/*-version.txt                                     (vLLM version line per setup)
  <root>/logs/*-startup.txt                                     ("Model loading took", "GPU KV cache size", kernel lines)

Usage:
  python summarize_results.py results-2                      # Llama track (default)
  python summarize_results.py results-qwen --track qwen      # Qwen track: BF16 on a full H200, FP8 and INT4 on 71 GB slices, INT4_35 on a 35 GB slice
  python summarize_results.py results-2 --json out.json

--track qwen changes the "per GPU" column to "per H200": a 71 GB slice counts as half a card (two fit) and a
35 GB slice as a third (three fit), so slice throughput is multiplied by 2 or 3 before it is compared with BF16.

It never guesses. Anything missing is reported as missing, and every number printed
names the file it came from.
"""
import argparse
import glob
import json
import os
import re
import sys

TRACKS = {
    # GPUS = H200 cards (or fractions of one) each setup occupies; throughput / GPUS = tokens per H200
    "llama": {
        "GPUS": {"FP16": 2, "BF16": 2, "INT4": 1, "INT4_RH": 1, "SPEC_DECODE": 2, "FP8": 1},
        "LABEL": {"FP16": "BF16", "BF16": "BF16", "INT4": "INT4 AWQ", "INT4_RH": "INT4 Red Hat", "SPEC_DECODE": "Spec decode", "FP8": "FP8"},
        "BASE": ("FP16", "BF16"),
    },
    "qwen": {
        "GPUS": {"BF16": 1, "FP8": 1 / 2, "INT4": 1 / 2, "INT4_35": 1 / 3, "SPEC_DECODE": 1},
        "LABEL": {"BF16": "BF16 (full H200)", "FP8": "FP8 (71 GB slice)", "INT4": "INT4 (71 GB slice)", "INT4_35": "INT4 (35 GB slice)", "SPEC_DECODE": "Spec decode (MTP)"},
        "BASE": ("BF16",),
    },
}
GPUS = TRACKS["llama"]["GPUS"]
LABEL = TRACKS["llama"]["LABEL"]
BASE = TRACKS["llama"]["BASE"]


def base_variant(rows):
    return next((v for v in BASE if v in rows), None)


def subdir(root, *names):
    for n in names:
        if os.path.isdir(os.path.join(root, n)):
            return os.path.join(root, n)
    return os.path.join(root, names[0])


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def pick(d, *keys):
    for k in keys:
        if d is not None and d.get(k) is not None:
            return d[k]
    return None


def bench_row(path):
    d = load(path)
    if d is None:
        return None
    return {
        "file": path,
        "concurrency": pick(d, "max_concurrency"),
        "completed": pick(d, "completed"),
        "output_tok_s": pick(d, "output_throughput"),
        "request_s": pick(d, "request_throughput"),
        "mean_ttft_ms": pick(d, "mean_ttft_ms"),
        "median_ttft_ms": pick(d, "median_ttft_ms"),
        "mean_tpot_ms": pick(d, "mean_tpot_ms"),
        "median_tpot_ms": pick(d, "median_tpot_ms"),
        "mean_e2el_ms": pick(d, "mean_e2el_ms"),
        "p95_tpot_ms": pick(d, "p95_tpot_ms"),
        "p99_tpot_ms": pick(d, "p99_tpot_ms"),
    }


def fmt(x, nd=1):
    return "missing" if x is None else f"{x:.{nd}f}"


def single_stream(root, out):
    print("\n== One request at a time (ShareGPT, 30 prompts) ==")
    for t in ("0", "0.7"):
        rows = {}
        for v in GPUS:
            r = bench_row(os.path.join(subdir(root, "bench", "sweeps"), v, f"single-t{t}.json"))
            if r:
                rows[v] = r
        if not rows:
            print(f"  temperature {t}: no files")
            continue
        print(f"  temperature {t}:")
        for v, r in rows.items():
            # For one stream, 1000 / mean TPOT is decode speed; output_throughput also includes prefill and gaps.
            decode = 1000.0 / r["mean_tpot_ms"] if r["mean_tpot_ms"] else None
            print(f"    {LABEL[v]:12s} output {fmt(r['output_tok_s'])} tok/s | decode {fmt(decode)} tok/s (1000/TPOT)"
                  f" | TTFT mean {fmt(r['mean_ttft_ms'],0)} ms | {r['completed']} done | {r['file']}")
        bv = base_variant(rows)
        base = rows.get(bv) if bv else None
        if base and base["output_tok_s"]:
            for v in GPUS:
                if v != bv and v in rows and rows[v]["output_tok_s"]:
                    ratio = rows[v]["output_tok_s"] / base["output_tok_s"]
                    print(f"    -> {LABEL[v]} is {ratio:.2f}x BF16 speed for one request")
                    out.setdefault("single_stream", {}).setdefault(t, {})[v + "_vs_BF16"] = round(ratio, 3)
        out.setdefault("single_stream_raw", {})[t] = rows


def sweep(root, out):
    for ds, prefix in (("random prompts", "c"), ("ShareGPT", "sharegpt-c")):
        print(f"\n== Load sweep, {ds} ==")
        table = {}
        for v in GPUS:
            for f in glob.glob(os.path.join(subdir(root, "bench", "sweeps"), v, f"{prefix}*.json")):
                m = re.search(r"c(\d+)\.json$", os.path.basename(f))
                if not m or (prefix == "c" and os.path.basename(f).startswith("sharegpt")):
                    continue
                r = bench_row(f)
                if r:
                    table.setdefault(int(m.group(1)), {})[v] = r
        if not table:
            print("  no files")
            continue
        present = [v for v in GPUS if any(v in table[c] for c in table)]
        print(f"  {'at once':>7} | " + " | ".join(f"{LABEL[v]:>30s}" for v in present))
        for c in sorted(table):
            cells = []
            for v in present:
                r = table[c].get(v)
                if not r or r["output_tok_s"] is None:
                    cells.append(f"{'missing':>30s}")
                else:
                    p95 = r.get("p95_tpot_ms")
                    cells.append(f"{r['output_tok_s']:6.0f} tok/s {r['output_tok_s']/GPUS[v]:6.0f}/H200 p95 {fmt(p95, 0):>4s}ms")
            print(f"  {c:>7} | " + " | ".join(cells))
            bv = base_variant(table[c])
            b = table[c].get(bv) if bv else None
            for iv in present:
                i = table[c].get(iv)
                if iv != bv and b and i and b["output_tok_s"] and i["output_tok_s"]:
                    per_gpu = (i["output_tok_s"] / GPUS[iv]) / (b["output_tok_s"] / GPUS[bv])
                    out.setdefault("sweep", {}).setdefault(ds, {}).setdefault(c, {})[f"{iv}_per_H200_vs_BF16_per_H200"] = round(per_gpu, 2)
                    print(f"          {LABEL[iv]} per H200 is {per_gpu:.2f}x BF16 per H200 at {c} at once")
        out.setdefault("sweep_raw", {})[ds] = table


def evals(root, out):
    print("\n== Accuracy (lm_eval) ==")
    found = False
    for v in GPUS:
        for sub in (v, v + "-mmlu_pro"):
            for f in sorted(glob.glob(os.path.join(subdir(root, "eval", "evals"), sub, "**", "results_*.json"), recursive=True)):
                d = load(f) or {}
                for task, metrics in (d.get("results") or {}).items():
                    for k, val in metrics.items():
                        if isinstance(val, (int, float)) and ("exact_match" in k or k.startswith("acc")) and "stderr" not in k:
                            found = True
                            print(f"  {LABEL[v]:12s} {task:22s} {k:32s} {val:.4f} | {f}")
                            out.setdefault("eval", {}).setdefault(v, {})[f"{task}:{k}"] = val
    if not found:
        print("  no files")
        return
    bv = base_variant(out["eval"])
    b = out["eval"].get(bv, {}) if bv else {}
    for iv in out["eval"]:
        if iv == bv:
            continue
        i = out["eval"].get(iv, {})
        for key in b:
            if key in i and b[key]:
                print(f"  -> {LABEL[iv]} recovers {100*i[key]/b[key]:.1f}% of BF16 on {key}")


def spec_metrics(root, out):
    print("\n== Spec decode acceptance, per temperature ==")
    pat = re.compile(r"^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+]+)")

    def read(path):
        vals = {}
        try:
            for line in open(path):
                m = pat.match(line.strip())
                if m:
                    vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(2))
        except OSError:
            return None
        return vals

    any_found = False
    runs = []
    for f in sorted(glob.glob(os.path.join(root, "logs", "spec-metrics-before-t*.txt"))):
        m = re.search(r"before-t([0-9.]+)(-k\d+)?\.txt$", os.path.basename(f))
        if m:
            runs.append((m.group(1), m.group(2) or ""))
    if not runs:
        runs = [("0", ""), ("0.7", "")]
    for t, k in runs:
        tag = f"temperature {t}" + (f", spec tokens {k[2:]}" if k else "")
        a = read(os.path.join(root, "logs", f"spec-metrics-before-t{t}{k}.txt"))
        z = read(os.path.join(root, "logs", f"spec-metrics-after-t{t}{k}.txt"))
        if not a or not z:
            print(f"  {tag}: missing before/after snapshot")
            continue
        drafts = z.get("drafts", 0) - a.get("drafts", 0)
        dtok = z.get("draft_tokens", 0) - a.get("draft_tokens", 0)
        acc = z.get("accepted_tokens", 0) - a.get("accepted_tokens", 0)
        if drafts <= 0:
            print(f"  {tag}: no drafts between snapshots")
            continue
        any_found = True
        mal = 1 + acc / drafts
        rate = acc / dtok if dtok else None
        print(f"  {tag}: mean acceptance length {mal:.2f} tokens per target pass"
              f" | draft acceptance rate {fmt(None if rate is None else 100*rate)}% (not alpha)")
        out.setdefault("spec_acceptance", {})[t + k] = {"mean_acceptance_length": round(mal, 2), "draft_acceptance_rate": rate}
    if not any_found:
        print("  no usable snapshots")


def versions(root, out):
    print("\n== vLLM version per setup ==")
    files = sorted(glob.glob(os.path.join(root, "logs", "*-version.txt")))
    if not files:
        print("  no *-version.txt files")
        return
    seen = set()
    for f in files:
        try:
            lines = [l.strip() for l in open(f) if l.strip()]
        except OSError:
            continue
        line = next((l for l in lines if re.search(r"\d+\.\d+\.\d+", l)), lines[0] if lines else "")
        m = re.search(r"\d+\.\d+\.\d+[^\s,'\"]*", line)
        ver = m.group(0) if m else line
        seen.add(ver)
        print(f"  {os.path.basename(f):32s} {ver} | {f}")
        out.setdefault("versions", {})[os.path.basename(f)] = ver
    if len(seen) > 1:
        print("  WARNING: setups ran different vLLM versions, so the comparison isn't apples to apples")


def startup(root, out):
    print("\n== Startup logs: weights loaded, KV cache, kernels ==")
    files = sorted(glob.glob(os.path.join(root, "logs", "*-startup.txt")))
    if not files:
        print("  no *-startup.txt files")
        return
    for f in files:
        try:
            text = open(f, errors="replace").read()
        except OSError:
            continue
        facts = {}
        m = re.search(r"Model loading took ([0-9.]+) GiB", text)
        if m:
            facts["loaded_gib"] = float(m.group(1))
        m = re.search(r"GPU KV cache size: ([0-9,]+) tokens", text)
        if m:
            facts["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        m = re.search(r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x", text)
        if m:
            facts["max_concurrency"] = f"{m.group(2)}x at {m.group(1)}"
        kernels = sorted(set(re.findall(r"\b(\w*(?:Machete|Marlin|Cutlass|CutlassFP8|FlashAttn|FLASH_ATTN|FlashInfer|DeepGEMM|Triton)\w*)", text)))
        if kernels:
            facts["kernels"] = kernels[:6]
        m = re.search(r"(MIG[^\n]{0,60}|mig-\dg\.\d+gb)", text)
        if m:
            facts["mig"] = m.group(1).strip()
        print(f"  {os.path.basename(f):48s} " + ", ".join(f"{k}={v}" for k, v in facts.items()) + f" | {f}")
        out.setdefault("startup", {})[os.path.basename(f)] = facts


def main():
    global GPUS, LABEL, BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results-2")
    ap.add_argument("--track", choices=sorted(TRACKS), default="llama")
    ap.add_argument("--json")
    a = ap.parse_args()
    GPUS, LABEL, BASE = TRACKS[a.track]["GPUS"], TRACKS[a.track]["LABEL"], TRACKS[a.track]["BASE"]
    if not os.path.isdir(a.root):
        sys.exit(f"{a.root} not found")
    out = {"track": a.track}
    single_stream(a.root, out)
    sweep(a.root, out)
    evals(a.root, out)
    spec_metrics(a.root, out)
    versions(a.root, out)
    startup(a.root, out)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
