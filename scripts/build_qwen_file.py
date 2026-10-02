#!/usr/bin/env python3
"""Build benchmark_results.qwen.json from a run folder in the layout under bench/raw/2026-09-30-qwen-r1/
(its scripts/RUN-QWEN.md).

    python scripts/build_qwen_file.py                                   # bench/raw/2026-09-30-qwen-r1
    python scripts/build_qwen_file.py <folder> --out <file>            # anywhere else, for a dry run

The Qwen track ran on vLLM 0.24: BF16 and the MTP speculator on one full H200, FP8 and INT4 on 71 GB
MIG slices (two per H200, the same profile, so the two are like for like), and INT4 once more on a
35 GB slice (three per H200) as the "it fits there too" footnote under the INT4 card.

Every throughput is per device (a full H200 or one slice). The `per_h200` block next to it is that
number times the slices per card: arithmetic, not a measurement (two slices were not loaded at
once), and labeled so; the app never divides a per-H200 number by a per-slice one.

What the file carries that the sheet of headline numbers doesn't: which pod each setup's log came
from, whether that pod ran with CUDA graphs or in eager mode, the thinking-mode setting, every
sweep point's timing window and which other runs on the same pod overlapped it (a point that shared
its pod with another sweep is a lower bound, and the app labels it), the first pass set aside and why,
every number of speculative tokens the run measured, and the other eval files when a task was run
more than once.
"""

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_benchmark_file import ROOT, load, single_stream  # noqa: E402

DEVICES = {  # resource -> (device name, slices per H200)
    "nvidia.com/gpu": ("H200", 1),
    "nvidia.com/mig-3g.71gb": ("71 GB slice", 2),
    "nvidia.com/mig-2g.35gb": ("35 GB slice", 3),
}
PER_H200_NOTE = "per device times the slices per H200: arithmetic, not a measurement"
INT4_BUILD = "Red Hat's build: W4A16, GPTQ via LLM Compressor, with AWQ smoothing, compressed-tensors format"

# Each setup names the pod whose logs it reads, since some ran on more than one pod.
SETUPS = {
    "BF16": {"raw": "BF16", "pod": "qwen-bf16-predictor-796d45bfd9-wgmdz", "checkpoint": "Qwen/Qwen3.8-27B",
             "quantization": None, "resource": "nvidia.com/gpu"},
    "FP8": {"raw": "FP8", "pod": "qwen-fp8-predictor-5446d888bb-r9nmt", "checkpoint": "Qwen/Qwen3.8-27B-FP8",
            "quantization": "FP8 (W8A8)", "resource": "nvidia.com/mig-3g.71gb", "build": "Qwen's FP8 build",
            "pod_note": "the measured pod, thinking off; qwen-fp8-predictor-85858648f7-gvdvc was the Phase 0 "
                        "smoke pod without the thinking flag"},
    "INT4": {"raw": "INT4", "pod": "qwen-int4-predictor-5d5c69c57b-qcrcc",
             "checkpoint": "RedHatAI/Qwen3.8-27B-INT4", "quantization": "W4A16",
             "resource": "nvidia.com/mig-3g.71gb", "build": INT4_BUILD},
    "SPEC_DECODE": {"raw": "SPEC_DECODE", "pod": "qwen-mtp-predictor-67986cf5bf-47s94",
                    "checkpoint": "Qwen/Qwen3.8-27B", "quantization": None, "resource": "nvidia.com/gpu",
                    "speculation": "MTP, the model's own head", "num_speculative_tokens": 4},
}
# the same INT4 checkpoint on a 35 GB slice: single-stream and the sweep, no evals (accuracy is the
# checkpoint's), the footnote under the INT4 card
INT4_35 = {"raw": "INT4_35", "pod": "qwen-int4-35gb-predictor-7ccb746499-qcbgv",
           "checkpoint": "RedHatAI/Qwen3.8-27B-INT4", "quantization": "W4A16",
           "resource": "nvidia.com/mig-2g.35gb", "build": INT4_BUILD,
           "pod_note": "the later of the two 35 GB pods, the one with the GPU record; both ran in eager mode"}

# A setup's first-pass/ folder holds the Sep 30 afternoon sweep, redone in the evening with one client
# per pod. Nothing is read from it; the note says why it was redone.
FIRST_PASS = (
    "the afternoon sweep: its random and ShareGPT runs shared the pod for stretches (the bench client "
    "runs inside the pod, so two runs at once share one server), and BF16's and INT4's single-t0.7 ran "
    "while the GSM8K eval (lm_eval, 32 concurrent) hit the same pod, doubling and tripling the per-token "
    "time; redone in the evening with one client per pod, which is what the numbers come from"
)


def pod_file(raw: Path, pod: str, suffix: str) -> Path:
    return raw / "logs" / f"{pod}-{suffix}.txt"


def startup_log(raw: Path, pod: str) -> dict:
    text = pod_file(raw, pod, "startup-full").read_text()

    def find(pattern):
        m = re.search(pattern, text)
        return m.group(1) if m else None

    args = find(r"non-default args: (\{.*\})") or ""
    info = {
        "pod": pod,
        "weights_gib_per_gpu": float(find(r"Model loading took ([\d.]+) GiB")),
        "kv_cache_tokens": int(find(r"GPU KV cache size: ([\d,]+) tokens").replace(",", "")),
        "attention": "FlashAttention " + str(find(r"Using FlashAttention version (\d)")),
        "kernels": sorted(set(re.findall(r"\b(\w+Kernel)\b", text))),
        # the one quantized-linear kernel the log names, for the dashboard's note (none for BF16 and MTP)
        "kernel": next(iter(sorted(set(re.findall(r"\b(\w+LinearKernel)\b", text)))), None),
        "non_default_args": args,
        # eager mode means no CUDA graphs: the MIG slices that hit the NVML profiling bug ran this way
        "enforce_eager": "'enforce_eager': True" in args,
        "cuda_graphs_captured": bool(re.search(r"Capturing CUDA graph|Graph capturing finished", text)),
    }
    if m := re.search(r"speculative_config=SpeculativeConfig\(method='(\w+)'.*?num_spec_tokens=(\d+)", text):
        info["spec_method"], info["spec_tokens_from_log"] = m.group(1), int(m.group(2))
    if m := re.search(r"enable_thinking['\"]?\s*:\s*(True|False|true|false)", args):
        info["thinking_in_server_args"] = m.group(1).lower() == "true"
    return info


def parse_versions(text: str) -> dict:
    """A pod's version file: "vllm X / torch Y / transformers Z" plus an nvidia-smi line, or, from the
    measured pods, the bare vLLM version alone (the record helper's python one-liner printed only that)."""
    lines = text.splitlines()
    out = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("vllm", "torch", "transformers"):
            out[parts[0]] = parts[1]
        elif len(parts) == 1 and re.fullmatch(r"\d+\.\d+\.\d+\S*", parts[0]):
            out.setdefault("vllm", parts[0])
    gpu_line = next((line for line in lines if line.startswith("NVIDIA")), "")
    fields = [x.strip() for x in gpu_line.split(",")]
    if fields and fields[0]:
        out["gpu"] = fields[0]
    if len(fields) > 1:
        out["driver"] = fields[1]
    if len(fields) > 2 and fields[2].endswith("MiB"):
        out["device_memory_mib"] = int(fields[2].split()[0])
    return out


def versions(raw: Path, pod: str) -> dict:
    """The pod's own versions, with torch, transformers and the driver filled in from another pod's full
    version file when the pod's own file has only the vLLM version (every pod ran the same image)."""
    out = parse_versions(pod_file(raw, pod, "version").read_text())
    if "torch" not in out or "driver" not in out:
        for other in sorted((raw / "logs").glob("*-version.txt")):
            full = parse_versions(other.read_text())
            if "torch" in full and "driver" in full:
                if full.get("vllm") != out.get("vllm"):
                    continue
                for key in ("torch", "transformers", "driver", "gpu"):
                    out.setdefault(key, full.get(key))
                out["versions_from"] = other.name.removesuffix("-version.txt")
                break
    return out


def image_digest(raw: Path, pod: str) -> str | None:
    path = raw / "logs" / f"{pod}-image.txt"
    return path.read_text().strip() if path.is_file() else None


def gsm8k(raw: Path, path: Path) -> dict:
    g = load(path)
    task = next(k for k in g["results"] if k.startswith("gsm8k"))
    gr = g["results"][task]
    return {
        "task": task, "metric": "exact_match, flexible-extract",
        "score": round(100 * gr["exact_match,flexible-extract"], 2),
        "stderr": round(100 * gr["exact_match_stderr,flexible-extract"], 2),
        "strict": round(100 * gr["exact_match,strict-match"], 2),
        "questions": g["n-samples"][task]["effective"],
        "fewshot": g["configs"][task]["num_fewshot"],
        "file": str(path.relative_to(raw)),
    }


def accuracy(raw: Path, folder: str) -> dict | None:
    """The latest results file of each task, with any earlier runs of the same task listed."""
    gsm = sorted(raw.glob(f"evals/{folder}/*/results_*.json"))
    mmlu = sorted(raw.glob(f"evals/{folder}-mmlu_pro/*/results_*.json"))
    if not gsm:
        return None
    out = {"gsm8k": gsm8k(raw, gsm[-1])}
    if len(gsm) > 1:
        out["gsm8k"]["other_runs"] = [gsm8k(raw, p) for p in gsm[:-1]]
        out["gsm8k"]["note"] = (f"the task was run {len(gsm)} times against the same pod (two of them at the "
                                "same time); the latest file is the score, the others are listed")
    if mmlu:
        m = load(mmlu[-1])
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
            "file": str(mmlu[-1].relative_to(raw)),
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
    """The MTP counters' movement over one single-stream run: the before and after snapshots the run list
    takes per temperature and per number of speculative tokens. Unknown counter names are kept raw, so nothing
    0.24 reports is lost even if this doesn't know what to call it."""
    before = raw / "logs" / f"spec-metrics-before-t{temperature}-k{k}.txt"
    after = raw / "logs" / f"spec-metrics-after-t{temperature}-k{k}.txt"
    if not (before.is_file() and after.is_file()):
        return None
    b, a = counters(before), counters(after)
    delta = {key: a[key] - b.get(key, 0.0) for key in a}

    def find(part):  # the _total counters, never the _created ones 0.24 reports beside them
        return next((v for key, v in delta.items()
                     if part in key and "[" not in key and "created" not in key), None)

    drafts, draft_tokens, accepted = find("num_drafts"), find("num_draft_tokens"), find("num_accepted_tokens")
    out = {"spec_tokens": k, "counters": {key: int(v) for key, v in delta.items()}}
    if drafts and draft_tokens and accepted is not None:
        out.update({
            "drafts": int(drafts), "draft_tokens": int(draft_tokens), "accepted": int(accepted),
            "rate": round(accepted / draft_tokens, 4),
            "mean_acceptance_length": round(1 + accepted / drafts, 2),
        })
        per_pos = {int(key.split("[")[1][:-1]): v for key, v in delta.items()
                   if "accepted_tokens_per_pos" in key and "[" in key and "created" not in key}
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


def sweep_runs(raw: Path, folder: str) -> dict:
    """Every vllm bench serve run in the setup's folder with its window (the pod's clock) and which other
    runs on the same pod overlapped it. vllm bench serve stamps `date` when it writes the result, after
    the run, and `duration` is the run's length, so the window is [date - duration, date]. The bench
    client runs inside the pod, so two runs at once share one server: an overlapped point is a lower
    bound."""
    runs = {}
    for path in sorted((raw / "sweeps" / folder).glob("*.json")):
        d = load(path)
        if not d.get("date") or d.get("duration") is None:
            continue
        end = datetime.strptime(d["date"], "%Y%m%d-%H%M%S")
        runs[path.name] = {"start": end - timedelta(seconds=float(d["duration"])), "end": end,
                           "concurrency": d.get("max_concurrency")}
    out = {}
    for name, r in runs.items():
        others = [
            {"file": other, "concurrency": o["concurrency"]}
            for other, o in runs.items() if other != name and o["start"] < r["end"] and r["start"] < o["end"]
        ]
        out[name] = {
            "start": r["start"].isoformat(timespec="seconds") + "Z",
            "end": r["end"].isoformat(timespec="seconds") + "Z",
            "overlapped_with": others,
        }
    return out


def sweep_note(raw: Path, folder: str) -> str | None:
    failed = sorted(p.name for p in (raw / "sweeps" / folder).glob("*-failed.txt"))
    if not failed:
        return None
    return "sweep points that failed and are kept as evidence: " + ", ".join(failed)


def setup(raw: Path, spec: dict) -> dict:
    out = {k: v for k, v in spec.items() if k not in ("raw", "pod")}
    name, slices = DEVICES[spec["resource"]]
    out["device"] = {"name": name, "count": 1, "per_h200": slices}
    out["gpus"] = 1  # one device, a full card or one slice; every per-GPU number in the app is per device
    out["slices_per_h200"] = slices
    out["tensor_parallel_size"] = 1
    out["dtype"] = "bfloat16"
    out.update(single_stream(raw, spec["raw"], "0"))
    out["temperature"] = 0
    out["at_temperature_0_7"] = single_stream(raw, spec["raw"], "0.7")
    out.update(startup_log(raw, spec["pod"]))
    out["versions"] = versions(raw, spec["pod"])
    out["image_digest"] = image_digest(raw, spec["pod"])
    if acc := accuracy(raw, spec["raw"]):
        out["accuracy"] = acc
    if "speculation" in spec:
        # every -t<T>-k<K> snapshot pair the run made, whatever the K values, and the single-stream
        # file for every K other than the main one (single-t0.json is the main K's run)
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
    out["sweep_runs"] = sweep_runs(raw, spec["raw"])
    overlapped = sorted(n for n, r in out["sweep_runs"].items() if r["overlapped_with"])
    if overlapped:
        out["overlap_note"] = (
            f"{len(overlapped)} of {len(out['sweep_runs'])} runs shared the pod with another vllm bench "
            "serve run (the random and ShareGPT sweeps ran side by side for stretches); those points are "
            "lower bounds and the app labels them: " + ", ".join(overlapped)
        )
    if (raw / "sweeps" / spec["raw"] / "first-pass").is_dir():
        out["first_pass"] = {"folder": f"sweeps/{spec['raw']}/first-pass", "note": FIRST_PASS}
    out["sweep_dir"] = spec["raw"]
    out["captures_dir"] = spec["raw"]
    return out


def finish(variants: list[dict], baseline: dict) -> None:
    """Ratios from the unrounded single-stream numbers, and the labeled per-H200 arithmetic."""
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
    for v in variants:
        del v["_tps"], v["at_temperature_0_7"]["_tps"]


def thinking_setting(raw: Path, variants: list[dict]) -> str:
    """What the run says about Qwen's thinking mode: the notes.txt line, and whether the pods themselves
    carried enable_thinking in their args (server-side, the same for every request)."""
    notes = (raw / "notes.txt").read_text() if (raw / "notes.txt").is_file() else ""
    line = next((ln.strip() for ln in notes.splitlines() if ln.lower().startswith("thinking")), None)
    if line:  # the notes say "Thinking mode: OFF (...)" or "thinking: off"; keep what follows the label
        line = re.sub(r"^thinking(?: mode)?\s*:\s*", "", line, flags=re.I)
    flags = {v.get("thinking_in_server_args") for v in variants}
    if flags == {False}:
        server = "off server-side in every pod (--default-chat-template-kwargs enable_thinking=false)"
    elif flags == {True}:
        server = "on server-side in every pod"
    elif flags == {None}:
        server = "not set in the pods' args (per request, if at all)"
    else:
        server = "NOT THE SAME IN EVERY POD: " + ", ".join(
            f"{v['pod']}={v.get('thinking_in_server_args')}" for v in variants)
    return f"{line}; {server}" if line else server


def run_date(stamp: str | None) -> str | None:
    """vllm bench serve stamps results YYYYMMDD-HHMMSS; the app wants YYYY-MM-DD."""
    if stamp and len(stamp) >= 8 and stamp[:8].isdigit():
        return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    return stamp


def build(raw: Path) -> dict:
    variants = {key: setup(raw, spec) for key, spec in SETUPS.items()}
    variants["INT4"]["int4_35"] = setup(raw, INT4_35)
    variants["INT4"]["int4_35"]["note"] = (
        "the same checkpoint on a 35 GB slice (three per H200): it loads and answers, in eager mode (the "
        "NVML CUDA-graph profiling bug on that slice), and never got under the 50 ms p95 budget under load"
    )
    everything = list(variants.values()) + [variants["INT4"]["int4_35"]]
    finish(everything, variants["BF16"])
    ver = variants["BF16"]["versions"]
    thinking = thinking_setting(raw, everything)
    eager = sorted(v["device"]["name"] for v in everything if v["enforce_eager"])
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
        "eager_devices": eager,
        "per_h200_note": PER_H200_NOTE,
        "notes": (
            "Qwen3.8-27B on vLLM 0.24.0+rhaiv.13 (the Llama track ran on 0.18; 0.24 is needed for MTP). "
            "Single stream: vllm bench serve inside each pod, 30 ShareGPT prompts one at a time, "
            "temperature 0 (headline) and 0.7, --max-model-len 32768. BF16 and the MTP speculator on one "
            "full H200 with CUDA graphs; FP8 and INT4 on 71 GB MIG slices (two per H200, the same profile, "
            "also with CUDA graphs); INT4 once more on a 35 GB slice (three per H200) in eager mode, the "
            "workaround for an NVML CUDA-graph profiling failure on that slice. Every throughput is per "
            "device; per_h200 is that number times the slices per card, arithmetic and not a measurement "
            "(two slices were not loaded at once). The sweeps and the BF16 and INT4 temperature-0.7 "
            "runs are the evening pass, one client per pod; the afternoon pass, where runs shared their "
            "pod, is kept under sweeps/<V>/first-pass/ and nothing is read from it (sweep_runs lists every "
            "run's window and any overlap, and the app labels an overlapped point as a lower bound). "
            "Accuracy is lm_eval: GSM8K (gsm8k_cot, not the Llama prompt "
            "format, so not comparable with the Llama track's figures) on all questions and MMLU-Pro on "
            "the first 20 of each subject. Spec decode counters were read before and after each "
            "single-stream run, per temperature and per number of speculative tokens (k). Thinking mode: "
            f"{thinking}. nvidia-smi inside a MIG slice reports no utilization, so the GPU-util CSVs of "
            "the slice setups are empty."
        ),
        "single_stream": {"dataset": "ShareGPT", "prompts": 30, "concurrency": 1, "where": "inside the pod"},
        "variants": variants,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build benchmark_results.qwen.json from a results folder")
    parser.add_argument("raw", nargs="?", default=str(ROOT / "bench" / "raw" / "2026-09-30-qwen-r1"))
    parser.add_argument("--out", default=str(ROOT / "benchmark_results.qwen.json"))
    opts = parser.parse_args()
    raw = Path(opts.raw).resolve()
    out_path = Path(opts.out)
    result = build(raw)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    for key, v in result["variants"].items():
        print(f"{key:12s} {v['throughput_tps']:6.1f} tok/s per {v['device']['name']:12s} "
              f"{v['speed_vs_baseline']:.2f}x  eager={v['enforce_eager']}")
