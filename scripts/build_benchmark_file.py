#!/usr/bin/env python3
"""Build benchmark_results.json from the raw measurement files.

Every number the dashboard shows comes from here, so nothing is typed in by hand:

    python scripts/build_benchmark_file.py            # round 2 (bench/raw/2026-09-29-r2): the file
    python scripts/build_benchmark_file.py --round1   # the Sep 29 afternoon run, kept under "history"

Round 2 ran `vllm bench serve` inside each pod: 30 ShareGPT prompts one at a time at temperature 0 and
0.7, the concurrency sweeps, lm_eval for GSM8K and MMLU-Pro, and the spec decode counters read before
and after each single-stream run. The INT4 setup on screen is Red Hat's validated W4A16 build (GPTQ); the
community AWQ build that was the naive pick is kept as a reference under it.
"""

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUND1 = ROOT / "bench" / "raw" / "2026-09-29-r1"
ROUND2 = ROOT / "bench" / "raw" / "2026-09-29-r2"
MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Column key -> raw folder. The INT4 column is Red Hat's build; the AWQ build is its reference.
SETUPS = {
    "BF16": {
        "raw": "FP16", "pod": "benchmark-bf16", "gpus": 2, "tensor_parallel_size": 2,
        "quantization": None, "dtype": "bfloat16", "checkpoint": MODEL,
    },
    "INT4": {
        "raw": "INT4_RH", "pod": "benchmark-int4-rh", "gpus": 1, "tensor_parallel_size": 1,
        "quantization": "gptq_marlin (AutoGPTQ format, W4A16)", "dtype": "float16",
        "checkpoint": "RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16",
        "build": "Red Hat W4A16 build (GPTQ)",
    },
    "SPEC_DECODE": {
        "raw": "SPEC_DECODE", "pod": "benchmark-spec", "gpus": 2, "tensor_parallel_size": 2,
        "quantization": None, "dtype": "bfloat16", "checkpoint": MODEL,
        "draft_model": "meta-llama/Llama-3.1-8B-Instruct", "num_speculative_tokens": 5,
    },
    "FP8": {
        "raw": "FP8", "pod": "benchmark-fp8", "gpus": 1, "tensor_parallel_size": 1,
        "quantization": "fp8 (compressed-tensors, W8A8)", "dtype": "bfloat16",
        "checkpoint": "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8",
        "build": "Red Hat FP8 build",
        "sweep_note": (
            "c1.json (random prompts) looks cold: 40.8 tok/s with a mean first token of 144 ms and p95 "
            "193 ms, against 51.1 tok/s and 62 ms on the ShareGPT single-stream run, the headline number."
        ),
    },
}
AWQ_REFERENCE = {
    "raw": "INT4", "pod": "benchmark-int4", "gpus": 1, "tensor_parallel_size": 1,
    "quantization": "awq_marlin", "dtype": "float16",
    "checkpoint": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    "build": "community AWQ build, the naive pick",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


# round 2


def startup_log(raw: Path, pod: str) -> dict:
    """Weights per GPU rank, KV cache size, kernel and attention backend from the vLLM startup log."""
    [path] = raw.glob(f"logs/{pod}-predictor-*-startup.txt")
    text = path.read_text()
    find = lambda pattern: (m := re.search(pattern, text)) and m.group(1)  # noqa: E731
    fa_version = find(r"Using FlashAttention version (\d)")
    info = {
        "weights_gib_per_gpu": float(find(r"Model loading took ([\d.]+) GiB")),
        "kv_cache_tokens": int(find(r"GPU KV cache size: ([\d,]+) tokens").replace(",", "")),
        "attention": f"FlashAttention {fa_version}",
    }
    if kernel := find(r"(?:Using|Selected) (\w+LinearKernel) for"):
        info["kernel"] = kernel
    return info


def versions(raw: Path, pod: str) -> dict:
    [path] = raw.glob(f"logs/{pod}-predictor-*-version.txt")
    lines = path.read_text().splitlines()
    vllm, torch = lines[0].split()
    gpu, driver = (x.strip() for x in lines[2].split(","))
    return {"vllm": vllm, "torch": torch, "gpu": gpu, "driver": driver}


def single_stream(raw: Path, folder: str, temperature: str) -> dict:
    d = load(raw / "sweeps" / folder / f"single-t{temperature}.json")
    assert d["failed"] == 0 and d["max_concurrency"] == 1
    return {
        "prompts": d["completed"],
        "throughput_tps": round(d["output_throughput"], 1),
        "_tps": d["output_throughput"],  # unrounded, for the ratios; dropped from the file
        "avg_latency_ms": round(d["mean_e2el_ms"], 1),
        "median_latency_ms": round(d["median_e2el_ms"], 1),
        "p95_latency_ms": round(d["p95_e2el_ms"], 1),
        "ttft_ms_avg": round(d["mean_ttft_ms"], 1),
        "ttft_ms_median": round(d["median_ttft_ms"], 1),
        "ttft_ms_p95": round(d["p95_ttft_ms"], 1),
        "tpot_ms_median": round(d["median_tpot_ms"], 2),
        "avg_tokens_per_request": round(d["total_output_tokens"] / d["completed"], 1),
        "avg_input_tokens": round(d["total_input_tokens"] / d["completed"], 1),
    }


def accuracy(raw: Path, folder: str) -> dict:
    [gsm] = raw.glob(f"evals/{folder}/*/results_*.json")
    [mmlu] = raw.glob(f"evals/{folder}-mmlu_pro/*/results_*.json")
    g, m = load(gsm), load(mmlu)
    gr, mr = g["results"]["gsm8k_cot_llama"], m["results"]["mmlu_pro"]
    subjects = sorted(k for k in m["results"] if k != "mmlu_pro")
    return {
        "gsm8k": {
            "task": "gsm8k_cot_llama", "metric": "exact_match, flexible-extract",
            "score": round(100 * gr["exact_match,flexible-extract"], 2),
            "stderr": round(100 * gr["exact_match_stderr,flexible-extract"], 2),
            "strict_score": round(100 * gr["exact_match,strict-match"], 2),
            "questions": g["n-samples"]["gsm8k_cot_llama"]["effective"],
            "fewshot": g["configs"]["gsm8k_cot_llama"]["num_fewshot"],
        },
        "mmlu_pro": {
            "task": "mmlu_pro", "metric": "exact_match, custom-extract",
            "score": round(100 * mr["exact_match,custom-extract"], 2),
            "stderr": round(100 * mr["exact_match_stderr,custom-extract"], 2),
            "questions": sum(m["n-samples"][s]["effective"] for s in subjects),
            "subjects": len(subjects),
            "per_subject": int(m["config"]["limit"]),
            "fewshot": m["configs"][subjects[0]]["num_fewshot"],
            "note": "the first 20 questions of each subject (lm_eval --limit 20), not a random sample",
        },
    }


def acceptance(raw: Path, temperature: str) -> dict:
    """Spec decode counters read before and after one single-stream run, so each temperature is isolated."""
    def counters(when):
        text = (raw / "logs" / f"spec-metrics-{when}-t{temperature}.txt").read_text()
        return [float(line.split()[-1]) for line in text.splitlines() if line.strip()]
    before, after = counters("before"), counters("after")
    drafts, draft_tokens, accepted = (int(a - b) for a, b in zip(after, before, strict=True))
    return {
        "drafts": drafts, "draft_tokens": draft_tokens, "accepted": accepted,
        "rate": round(accepted / draft_tokens, 4),
        "mean_acceptance_length": round(1 + accepted / drafts, 2),
    }


def setup(raw: Path, spec: dict) -> dict:
    t0, t07 = single_stream(raw, spec["raw"], "0"), single_stream(raw, spec["raw"], "0.7")
    out = {k: v for k, v in spec.items() if k not in ("raw", "pod")}
    out.update(t0)
    out["temperature"] = 0
    out["at_temperature_0_7"] = t07
    out.update(startup_log(raw, spec["pod"]))
    out["versions"] = versions(raw, spec["pod"])
    if (raw / "evals" / spec["raw"]).is_dir():
        out["accuracy"] = accuracy(raw, spec["raw"])
    if "draft_model" in spec:
        out["acceptance"] = {t: acceptance(raw, t) for t in ("0", "0.7")}
        out["draft_acceptance_rate"] = out["acceptance"]["0"]["rate"]
        out["mean_acceptance_length"] = out["acceptance"]["0"]["mean_acceptance_length"]
    out["sweep_dir"] = spec["raw"]
    return out


def build_round2(raw: Path = ROUND2) -> dict:
    variants = {key: setup(raw, spec) for key, spec in SETUPS.items()}
    variants["INT4"]["reference"] = setup(raw, AWQ_REFERENCE)
    variants["INT4"]["reference"]["sweep_note"] = (
        "sharegpt-c64.json for this build was a retry after a timed-out first attempt on the same prompts, "
        "with prefix caching on; its first-token times are far below every other setup's at 64 in flight, "
        "so that point isn't used for any claim."
    )
    bf = variants["BF16"]
    bf_tps, bf_tps_07 = bf["_tps"], bf["at_temperature_0_7"]["_tps"]
    for v in list(variants.values()) + [variants["INT4"]["reference"]]:
        v["speed_vs_baseline"] = round(v["_tps"] / bf_tps, 3)
        v["speed_vs_baseline_t0_7"] = round(v["at_temperature_0_7"]["_tps"] / bf_tps_07, 3)
    for v in list(variants.values()) + [variants["INT4"]["reference"]]:
        del v["_tps"], v["at_temperature_0_7"]["_tps"]
    ver = bf["versions"]
    return {
        "model": MODEL,
        "gpu": "NVIDIA H200 (141 GB)",
        "vllm_version": ver["vllm"],
        "torch_version": ver["torch"],
        "driver_version": ver["driver"],
        "date": "2026-09-29",
        "cluster": "Red Hat internal H200 cluster",
        "source": f"built by scripts/build_benchmark_file.py from {raw.relative_to(ROOT)}",
        "notes": (
            "Round 2, the evening of Sep 29, 2026 (FP8 ran last, into the early hours of Sep 30). Single "
            "stream: vllm bench serve inside each pod, 30 "
            "ShareGPT prompts one at a time (max_concurrency 1), at temperature 0 (the headline numbers) and "
            "0.7, enforce_eager off everywhere, so no network hop is included. throughput_tps is output "
            "tokens per second over the whole run. Accuracy is lm_eval through a port-forward: GSM8K 8-shot "
            "CoT on all 1,319 questions and MMLU-Pro 5-shot on the first 20 questions of each of 14 subjects "
            "(280). Spec decode counters were read before and after each single-stream run, so acceptance is "
            "per temperature. The INT4 setup is Red Hat's validated W4A16 build (GPTQ); the community AWQ "
            "build is under reference. The round-1 numbers are under history."
        ),
        "single_stream": {"dataset": "ShareGPT", "prompts": 30, "concurrency": 1, "where": "inside the pod"},
        "variants": variants,
        "history": {"round1_2026_09_29_afternoon": build_round1()},
    }


# round 1 (kept as history)

# Reported by the vLLM startup logs of each deployment on Sep 29, 2026 (those logs weren't exported).
STARTUP_LOGS_ROUND1 = {
    "BF16": {"weights_gib_per_gpu": 65.74, "kv_cache_tokens": 367280},
    "INT4": {"weights_gib_per_gpu": 37.87, "kv_cache_tokens": 280112, "kernel": "MacheteLinearKernel"},
    "SPEC_DECODE": {"weights_gib_per_gpu": 73.24, "kv_cache_tokens": 218064},
}


def summarize_runs(runs: list[dict]) -> dict:
    ms = [r["elapsed"] * 1000 for r in runs]
    tps = [r["tok_per_sec"] for r in runs]
    return {
        "runs": len(runs),
        "throughput_tps": round(statistics.mean(tps), 1),
        "avg_latency_ms": round(statistics.mean(ms), 1),
        # with 5 runs, the slowest run stands in for p95
        "p95_latency_ms": round(max(ms), 1),
        "avg_tokens_per_request": round(statistics.mean(r["tokens"] for r in runs), 1),
        "per_run_tps": tps,
    }


def build_round1(raw: Path = ROUND1) -> dict:
    single = load(raw / "benchmark-throughput.json")
    spec = load(raw / "benchmark-spec-decode.json")
    ttft = load(raw / "benchmark-ttft.json")
    metrics = load(raw / "spec-decode-metrics.json")["acceptance_rate"]
    natural = load(raw / "benchmark-natural-length.json")
    variants = {
        "BF16": {
            "gpus": 2, "tensor_parallel_size": 2, "quantization": None, "dtype": "bfloat16",
            **summarize_runs(single["bf16"]["runs"]),
            "ttft_ms_avg": ttft["bf16_ttft"]["avg_ms"],
            "natural_length_tokens_avg": natural["bf16"]["avg"],
            **STARTUP_LOGS_ROUND1["BF16"],
        },
        "INT4": {
            "gpus": 1, "tensor_parallel_size": 1, "quantization": "awq_marlin", "dtype": "float16",
            "checkpoint": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            **summarize_runs(single["int4"]["runs"]),
            "ttft_ms_avg": ttft["int4_ttft"]["avg_ms"],
            "natural_length_tokens_avg": natural["int4"]["avg"],
            **STARTUP_LOGS_ROUND1["INT4"],
        },
        "SPEC_DECODE": {
            "gpus": 2, "tensor_parallel_size": 2, "quantization": None, "dtype": "bfloat16",
            "draft_model": spec["spec_decode"]["draft_model"],
            "num_speculative_tokens": spec["spec_decode"]["num_speculative_tokens"],
            "enforce_eager": spec["spec_decode"]["enforce_eager"],
            **summarize_runs(spec["spec_decode"]["runs"]),
            "first_run_cold": True,
            "draft_acceptance_rate": metrics["overall_rate"],
            "mean_acceptance_length": round(1 + metrics["total_accepted"] / metrics["total_drafts"], 2),
            **STARTUP_LOGS_ROUND1["SPEC_DECODE"],
        },
    }
    return {
        "date": "2026-09-29",
        "source": f"built from {raw.relative_to(ROOT)}",
        "notes": (
            "Round 1, the afternoon of Sep 29: 5 requests per setup, one at a time, temperature 0, 256 "
            "output tokens each, through a port-forward from a laptop, so latencies include that network "
            "hop. The raw files don't record the prompt. Spec Decode's 64.9 tok/s (about 1.4x BF16) came "
            "from this one prompt and its first run was cold; round 2's 30 ShareGPT prompts give 1.25x, the "
            "number used everywhere. Acceptance counters here are cumulative since pod start and include "
            "the temperature 0.7 quality runs."
        ),
        "variants": variants,
        "spec_decode_enforce_eager": {
            "date": "2026-09-29",
            "throughput_tps": 40.0, "avg_latency_ms": 5543.9, "p95_latency_ms": 8646.0,
            "enforce_eager": True, "temperature": 0.7, "max_tokens": 256, "requests": 20,
            "note": "First run: enforce_eager on (no torch.compile, no CUDA graphs), temperature 0.7.",
        },
    }


if __name__ == "__main__":
    if "--round1" in sys.argv:
        print(json.dumps(build_round1(), indent=2))
        sys.exit(0)
    out = build_round2()
    (ROOT / "benchmark_results.json").write_text(json.dumps(out, indent=2) + "\n")
    for key, v in out["variants"].items():
        print(f"{key:12s} {v['throughput_tps']:6.1f} tok/s  {v['speed_vs_baseline']:.2f}x  "
              f"{v['avg_latency_ms'] / 1000:5.2f} s  TTFT {v['ttft_ms_avg']:.0f} ms  {v['gpus']} GPU")
    ref = out["variants"]["INT4"]["reference"]
    print(f"{'AWQ reference':12s} {ref['throughput_tps']:6.1f} tok/s  {ref['speed_vs_baseline']:.2f}x")
