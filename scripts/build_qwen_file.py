#!/usr/bin/env python3
"""Build benchmark_results.qwen.json from a results-qwen/ folder in the layout RUN-QWEN.md asks for.

    python scripts/build_qwen_file.py bench/raw/2026-10-qwen-r1       # writes benchmark_results.qwen.json
    python scripts/build_qwen_file.py <folder> --out <file>               # anywhere else, for a dry run

The Qwen track ran on vLLM 0.24: BF16 and the MTP speculator on one full H200, FP8 on a 71 GB MIG slice
(two per H200), and the INT4 card on a 35 GB slice (three per H200). The INT4 card's speed, load numbers
and recordings come from the 35 GB run (INT4_35 in the raw folder); its accuracy comes from the same
checkpoint on a 71 GB slice (INT4 in the raw folder), since accuracy is a property of the checkpoint,
not the slice, and that 71 GB run is kept under the card as the apples-to-apples footnote against FP8.

Every throughput number is per device (a full H200 or one slice). The `per_h200` block next to it is
that number times the slices per card, arithmetic and not a measurement, and is labeled so; the app
never divides a per-H200 number by a per-slice one.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_benchmark_file import ROOT, load, single_stream  # noqa: E402

DEVICES = {  # resource -> (device name, slices per H200)
    "nvidia.com/gpu": ("H200", 1),
    "nvidia.com/mig-3g.71gb": ("71 GB slice", 2),
    "nvidia.com/mig-2g.35gb": ("35 GB slice", 3),
}
PER_H200_NOTE = "per device times the slices per H200: arithmetic, not a measurement"
INT4_BUILD = "Red Hat's LLM Compressor W4A16 build (AWQ smoothing + GPTQ, compressed-tensors)"

SETUPS = {
    "BF16": {"raw": "BF16", "isvc": "qwen-bf16", "checkpoint": "Qwen/Qwen3.8-27B", "quantization": None,
             "resource": "nvidia.com/gpu"},
    "FP8": {"raw": "FP8", "isvc": "qwen-fp8", "checkpoint": "Qwen/Qwen3.8-27B-FP8",
            "quantization": "FP8 (W8A8)", "resource": "nvidia.com/mig-3g.71gb", "build": "Qwen's FP8 build"},
    "INT4": {"raw": "INT4_35", "isvc": "qwen-int4-35gb", "checkpoint": "RedHatAI/Qwen3.8-27B-INT4",
             "quantization": "W4A16", "resource": "nvidia.com/mig-2g.35gb", "build": INT4_BUILD,
             "accuracy_from": "INT4"},
    "SPEC_DECODE": {"raw": "SPEC_DECODE", "isvc": "qwen-mtp", "checkpoint": "Qwen/Qwen3.8-27B",
                    "quantization": None, "resource": "nvidia.com/gpu",
                    "speculation": "MTP, the model's own head", "num_speculative_tokens": 4},
}
# the same INT4 checkpoint on a 71 GB slice: the evals ran here, and it's the like-for-like line against FP8
INT4_71 = {"raw": "INT4", "isvc": "qwen-int4", "checkpoint": "RedHatAI/Qwen3.8-27B-INT4",
           "quantization": "W4A16", "resource": "nvidia.com/mig-3g.71gb", "build": INT4_BUILD}


def pod_file(raw: Path, isvc: str, suffix: str) -> Path:
    [path] = raw.glob(f"logs/{isvc}-predictor-*-{suffix}.txt")
    return path


def startup_log(raw: Path, isvc: str) -> dict:
    text = pod_file(raw, isvc, "startup-full").read_text()
    find = lambda pattern: (m := re.search(pattern, text)) and m.group(1)  # noqa: E731
    info = {
        "weights_gib_per_gpu": float(find(r"Model loading took ([\d.]+) GiB")),
        "kv_cache_tokens": int(find(r"GPU KV cache size: ([\d,]+) tokens").replace(",", "")),
        "attention": f"FlashAttention {find(r'Using FlashAttention version (\d)')}",
        "kernels": sorted(set(re.findall(r"\b(\w+Kernel)\b", text))),
        "non_default_args": find(r"non-default args: (\{.*\})"),
    }
    if m := re.search(r"speculative_config=SpeculativeConfig\(method='(\w+)'.*?num_spec_tokens=(\d+)", text):
        info["spec_method"], info["spec_tokens_from_log"] = m.group(1), int(m.group(2))
    # thinking turned off server-side: --default-chat-template-kwargs={"enable_thinking": false} shows up in
    # the non-default args as default_chat_template_kwargs; absent, the setting came per request or not at all
    args = info["non_default_args"] or ""
    if m := re.search(r"enable_thinking['\"]?\s*:\s*(True|False|true|false)", args):
        info["thinking_in_server_args"] = m.group(1).lower() == "true"
    return info


def versions(raw: Path, isvc: str) -> dict:
    lines = pod_file(raw, isvc, "version").read_text().splitlines()
    out = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("vllm", "torch", "transformers"):
            out[parts[0]] = parts[1]
    gpu_line = next((line for line in lines if line.startswith("NVIDIA")), "")
    fields = [x.strip() for x in gpu_line.split(",")]
    out["gpu"] = fields[0] if fields else None
    out["driver"] = fields[1] if len(fields) > 1 else None
    if len(fields) > 2 and fields[2].endswith("MiB"):
        out["device_memory_mib"] = int(fields[2].split()[0])
    return out


def image_digest(raw: Path, isvc: str) -> str | None:
    path = next(raw.glob(f"logs/{isvc}-predictor-*-image.txt"), None)
    return path.read_text().strip() if path else None


def accuracy(raw: Path, folder: str) -> dict | None:
    gsm = next(raw.glob(f"evals/{folder}/*/results_*.json"), None)
    mmlu = next(raw.glob(f"evals/{folder}-mmlu_pro/*/results_*.json"), None)
    if not gsm:
        return None
    g = load(gsm)
    task = next(k for k in g["results"] if k.startswith("gsm8k"))
    gr = g["results"][task]
    out = {
        "gsm8k": {
            "task": task, "metric": "exact_match, flexible-extract",
            "score": round(100 * gr["exact_match,flexible-extract"], 2),
            "stderr": round(100 * gr["exact_match_stderr,flexible-extract"], 2),
            "questions": g["n-samples"][task]["effective"],
            "fewshot": g["configs"][task]["num_fewshot"],
        }
    }
    if mmlu:
        m = load(mmlu)
        mr = m["results"]["mmlu_pro"]
        subjects = sorted(k for k in m["results"] if k != "mmlu_pro")
        out["mmlu_pro"] = {
            "task": "mmlu_pro", "metric": "exact_match, custom-extract",
            "score": round(100 * mr["exact_match,custom-extract"], 2),
            "stderr": round(100 * mr["exact_match_stderr,custom-extract"], 2),
            "questions": sum(m["n-samples"][s]["effective"] for s in subjects),
            "subjects": len(subjects),
            "per_subject": int(m["config"]["limit"]),
            "fewshot": m["configs"][subjects[0]]["num_fewshot"],
            "note": "the first 20 questions of each subject (lm_eval --limit 20), not a random sample",
        }
    return out


def counters(path: Path) -> dict:
    """Every vllm:spec_decode counter in a /metrics snapshot, keyed by metric name (per-position ones by
    position)."""
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"vllm:spec_decode_(\w+)(?:\{([^}]*)\})?\s+([\d.e+-]+)", line)
        if not m:
            continue
        name, labels, value = m.groups()
        pos = re.search(r'position="(\d+)"', labels or "")
        key = f"{name}[{pos.group(1)}]" if pos else name
        out[key] = out.get(key, 0.0) + float(value)
    return out


def acceptance(raw: Path, temperature: str, k: int) -> dict | None:
    """The MTP counters' movement over one single-stream run: the before and after snapshots RUN-QWEN takes
    per temperature and per number of speculative tokens. Unknown counter names are kept raw, so nothing
    0.24 reports is lost even if this doesn't know what to call it."""
    before = raw / "logs" / f"spec-metrics-before-t{temperature}-k{k}.txt"
    after = raw / "logs" / f"spec-metrics-after-t{temperature}-k{k}.txt"
    if not (before.is_file() and after.is_file()):
        return None
    b, a = counters(before), counters(after)
    delta = {key: a[key] - b.get(key, 0.0) for key in a}

    def find(part):
        return next((v for key, v in delta.items() if part in key and "[" not in key), None)

    drafts, draft_tokens, accepted = find("num_drafts"), find("num_draft_tokens"), find("num_accepted_tokens")
    out = {"spec_tokens": k, "counters": {key: int(v) for key, v in delta.items()}}
    if drafts and draft_tokens and accepted is not None:
        out.update({
            "drafts": int(drafts), "draft_tokens": int(draft_tokens), "accepted": int(accepted),
            "rate": round(accepted / draft_tokens, 4),
            "mean_acceptance_length": round(1 + accepted / drafts, 2),
        })
        per_pos = {int(key.split("[")[1][:-1]): v for key, v in delta.items()
                   if "accepted_tokens_per_pos[" in key}
        if per_pos:
            out["accepted_per_position"] = [round(per_pos[p] / drafts, 4) for p in sorted(per_pos)]
    return out


def snapshot_keys(raw: Path) -> set[tuple[str, int]]:
    """(temperature, K) for every before/after pair of spec counter snapshots in logs/."""
    keys = set()
    for path in (raw / "logs").glob("spec-metrics-before-t*-k*.txt"):
        m = re.fullmatch(r"spec-metrics-before-t([\d.]+)-k(\d+)\.txt", path.name)
        if m and (raw / "logs" / f"spec-metrics-after-t{m.group(1)}-k{m.group(2)}.txt").is_file():
            keys.add((m.group(1), int(m.group(2))))
    return keys


def sweep_note(raw: Path, folder: str) -> str | None:
    failed = sorted(p.name for p in (raw / "sweeps" / folder).glob("*-failed.txt"))
    if not failed:
        return None
    return "sweep points that failed and are kept as evidence: " + ", ".join(failed)


def setup(raw: Path, spec: dict) -> dict:
    out = {k: v for k, v in spec.items() if k not in ("raw", "isvc", "accuracy_from")}
    name, slices = DEVICES[spec["resource"]]
    out["device"] = {"name": name, "count": 1, "per_h200": slices}
    out["gpus"] = 1  # one device, a full card or one slice; every per-GPU number in the app is per device
    out["slices_per_h200"] = slices
    out["tensor_parallel_size"] = 1
    out["dtype"] = "bfloat16"
    t0, t07 = single_stream(raw, spec["raw"], "0"), single_stream(raw, spec["raw"], "0.7")
    out.update(t0)
    out["temperature"] = 0
    out["at_temperature_0_7"] = t07
    out.update(startup_log(raw, spec["isvc"]))
    out["versions"] = versions(raw, spec["isvc"])
    out["image_digest"] = image_digest(raw, spec["isvc"])
    if acc := accuracy(raw, spec.get("accuracy_from", spec["raw"])):
        out["accuracy"] = acc
        if "accuracy_from" in spec:
            out["accuracy_note"] = (
                f"measured on the same checkpoint on a {DEVICES[INT4_71['resource']][0]}: accuracy is a "
                "property of the checkpoint, not the slice"
            )
    if "speculation" in spec:
        # every -t<T>-k<K> snapshot pair the run made, whatever the K values (RUN-QWEN asks for 1, 2 and
        # 4; an extra pass at 8 or any other K is picked up the same way), and the single-stream file
        # for every K other than the main one (single-t0.json is the main K's run)
        main_k = spec["num_speculative_tokens"]
        out["acceptance"] = {}
        for t, k in sorted(snapshot_keys(raw)):
            if a := acceptance(raw, t, k):
                out["acceptance"][f"t{t}-k{k}"] = a
        main = out["acceptance"].get(f"t0-k{main_k}")
        if main and "rate" in main:
            out["draft_acceptance_rate"] = main["rate"]
            out["mean_acceptance_length"] = main["mean_acceptance_length"]
        out["spec_tokens_measured"] = sorted({k for _t, k in snapshot_keys(raw)} | {main_k})
        for path in sorted((raw / "sweeps" / spec["raw"]).glob("single-t0-k*.json")):
            k = int(re.fullmatch(r"single-t0-k(\d+)\.json", path.name).group(1))
            out[f"throughput_tps_k{k}"] = round(load(path)["output_throughput"], 1)
    if note := sweep_note(raw, spec["raw"]):
        out["sweep_note"] = note
    out["sweep_dir"] = spec["raw"]
    out["captures_dir"] = spec["raw"]
    return out


def run_date(stamp: str | None) -> str | None:
    """vllm bench serve stamps results YYYYMMDD-HHMMSS; the app wants YYYY-MM-DD."""
    if stamp and len(stamp) >= 8 and stamp[:8].isdigit():
        return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    return stamp


def two_slices(raw: Path, folder: str) -> dict | None:
    """sweeps/<V>/c<N>-two-slices-{a,b}.json: the same setup on two slices of one card, loaded at the same
    time. Their output throughputs added are a measured per-H200 number for that load, not arithmetic."""
    pairs = {}
    for path in (raw / "sweeps" / folder).glob("c*-two-slices-*.json"):
        m = re.fullmatch(r"c(\d+)-two-slices-(\w+)\.json", path.name)
        if m:
            pairs.setdefault(int(m.group(1)), {})[m.group(2)] = load(path)
    for c in sorted(pairs, reverse=True):
        runs = pairs[c]
        if len(runs) >= 2:
            per_slice = {k: round(runs[k]["output_throughput"], 1) for k in sorted(runs)}
            return {
                "concurrency_per_slice": c,
                "slices_loaded": len(runs),
                "output_tokens_per_second": round(sum(r["output_throughput"] for r in runs.values()), 1),
                "per_slice": per_slice,
                "tpot_p95_ms": {k: runs[k].get("p95_tpot_ms") for k in sorted(runs)},
                "files": [f"sweeps/{folder}/c{c}-two-slices-{k}.json" for k in sorted(runs)],
                "note": f"{len(runs)} slices of one H200 loaded together at {c} requests each: measured",
            }
    return None


def finish(raw: Path, variants: list[dict], baseline: dict) -> None:
    """Ratios from the unrounded single-stream numbers, and the labeled per-H200 arithmetic (or the
    two-slices measurement where the run made one)."""
    for v in variants:
        v["speed_vs_baseline"] = round(v["_tps"] / baseline["_tps"], 3)
        t07, base07 = v["at_temperature_0_7"]["_tps"], baseline["at_temperature_0_7"]["_tps"]
        v["speed_vs_baseline_t0_7"] = round(t07 / base07, 3)
        n = v["slices_per_h200"]
        v["per_h200"] = {
            "slices": n,
            "throughput_tps": round(v["_tps"] * n, 1),
            "note": PER_H200_NOTE if n > 1 else "one full H200, the measured number",
        }
        if n > 1 and (measured := two_slices(raw, v["sweep_dir"])):
            v["per_h200"]["measured"] = measured
    for v in variants:
        del v["_tps"], v["at_temperature_0_7"]["_tps"]


def thinking_setting(raw: Path, variants: list[dict]) -> str | None:
    """What the run says about Qwen's thinking mode: the notes.txt line, and whether the pods themselves
    carried enable_thinking in their args (server-side, the same for every request)."""
    notes = (raw / "notes.txt").read_text() if (raw / "notes.txt").is_file() else ""
    line = next((ln.strip() for ln in notes.splitlines() if ln.lower().startswith("thinking")), None)
    flags = {v.get("thinking_in_server_args") for v in variants}
    if flags == {False}:
        server = "off server-side in every pod (--default-chat-template-kwargs enable_thinking=false)"
    elif flags == {True}:
        server = "on server-side in every pod"
    elif None in flags and len(flags) == 1:
        server = "not set in the pods' args (per request, if at all)"
    else:
        server = "NOT THE SAME IN EVERY POD: " + ", ".join(
            f"{v['checkpoint']}={v.get('thinking_in_server_args')}" for v in variants)
    return f"{line}; {server}" if line else server


def build(raw: Path) -> dict:
    variants = {key: setup(raw, spec) for key, spec in SETUPS.items()}
    variants["INT4"]["int4_71"] = setup(raw, INT4_71)
    variants["INT4"]["int4_71"]["note"] = (
        "the same checkpoint on a 71 GB slice, the profile FP8 ran on, so this line and FP8's get the same "
        "share of the GPU's compute; the card above is the 35 GB slice"
    )
    everything = list(variants.values()) + [variants["INT4"]["int4_71"]]
    finish(raw, everything, variants["BF16"])
    ver = variants["BF16"]["versions"]
    thinking = thinking_setting(raw, everything)
    where = raw.relative_to(ROOT) if raw.is_relative_to(ROOT) else raw
    return {
        "track": "qwen",
        "model": "Qwen/Qwen3.8-27B",
        "gpu": "NVIDIA H200 (141 GB): a full card or a MIG slice per setup, named under each setup's device",
        "vllm_version": ver.get("vllm"),
        "torch_version": ver.get("torch"),
        "transformers_version": ver.get("transformers"),
        "driver_version": ver.get("driver"),
        "date": run_date(load(raw / "sweeps" / "BF16" / "single-t0.json").get("date")),
        "source": f"built by scripts/build_qwen_file.py from {where}",
        "thinking": thinking,
        "per_h200_note": PER_H200_NOTE,
        "notes": (
            "Qwen3.8-27B on vLLM 0.24.0+rhaiv.13 (the Llama track ran on 0.18; 0.24 is needed for MTP). "
            "Single stream: vllm bench serve inside each pod, 30 ShareGPT prompts one at a time, "
            "temperature 0 (headline) and 0.7, --max-model-len 32768. BF16 and the MTP speculator on one "
            "full H200; FP8 on a 71 GB MIG slice (two per H200); the INT4 card on a 35 GB slice (three per "
            "H200), with its accuracy "
            "from the same checkpoint on a 71 GB slice and that run kept under it as the like-for-like line "
            "against FP8. Every throughput is per device; per_h200 is that number times the slices per card, "
            "arithmetic and not a measurement. Accuracy is lm_eval: GSM8K (gsm8k_cot) on all questions and "
            "MMLU-Pro on the first 20 of each subject. Spec decode counters were read before and after each "
            "single-stream run, per temperature and per number of speculative tokens (k). Thinking mode: "
            f"{thinking}."
        ),
        "single_stream": {"dataset": "ShareGPT", "prompts": 30, "concurrency": 1, "where": "inside the pod"},
        "variants": variants,
    }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw = Path(args[0] if args else ROOT / "bench" / "raw" / "2026-10-qwen-r1").resolve()
    out_path = ROOT / "benchmark_results.qwen.json"
    if "--out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--out") + 1])
    result = build(raw)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    for key, v in result["variants"].items():
        print(f"{key:12s} {v['throughput_tps']:6.1f} tok/s per {v['device']['name']:12s} "
              f"{v['speed_vs_baseline']:.2f}x")
