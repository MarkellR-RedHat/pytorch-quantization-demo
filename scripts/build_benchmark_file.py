#!/usr/bin/env python3
"""Build benchmark_results.json from the raw measurement files in bench/raw/<date>/.

Every number the dashboard shows comes from here, so nothing is typed in by hand. Run it after
dropping new raw files in:  python scripts/build_benchmark_file.py bench/raw/2026-09-29
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Reported by the vLLM startup logs of each deployment on Sep 29, 2026 (the logs themselves weren't
# exported). "Model loading took" is per GPU rank.
STARTUP_LOGS = {
    "FP16": {"weights_gib_per_gpu": 65.74, "kv_cache_tokens": 367280},
    "INT4": {"weights_gib_per_gpu": 37.87, "kv_cache_tokens": 280112, "kernel": "MacheteLinearKernel"},
    "SPEC_DECODE": {"weights_gib_per_gpu": 73.24, "kv_cache_tokens": 218064},
}


def summarize(runs: list[dict]) -> dict:
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


def main(raw_dir: Path) -> dict:
    load = lambda name: json.loads((raw_dir / name).read_text())  # noqa: E731
    single = load("benchmark-throughput.json")
    spec = load("benchmark-spec-decode.json")
    ttft = load("benchmark-ttft.json")
    metrics = load("spec-decode-metrics.json")["acceptance_rate"]
    natural = load("benchmark-natural-length.json")

    variants = {
        "FP16": {
            "gpus": 2, "tensor_parallel_size": 2, "quantization": None, "dtype": "bfloat16",
            **summarize(single["bf16"]["runs"]),
            "ttft_ms_avg": ttft["bf16_ttft"]["avg_ms"],
            "natural_length_tokens_avg": natural["bf16"]["avg"],
            **STARTUP_LOGS["FP16"],
        },
        "INT4": {
            "gpus": 1, "tensor_parallel_size": 1, "quantization": "awq_marlin", "dtype": "float16",
            "checkpoint": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            **summarize(single["int4"]["runs"]),
            "ttft_ms_avg": ttft["int4_ttft"]["avg_ms"],
            "natural_length_tokens_avg": natural["int4"]["avg"],
            **STARTUP_LOGS["INT4"],
        },
        "SPEC_DECODE": {
            "gpus": 2, "tensor_parallel_size": 2, "quantization": None, "dtype": "bfloat16",
            "draft_model": spec["spec_decode"]["draft_model"],
            "num_speculative_tokens": spec["spec_decode"]["num_speculative_tokens"],
            "enforce_eager": spec["spec_decode"]["enforce_eager"],
            **summarize(spec["spec_decode"]["runs"]),
            "first_run_cold": True,
            "draft_acceptance_rate": metrics["overall_rate"],
            "mean_acceptance_length": round(1 + metrics["total_accepted"] / metrics["total_drafts"], 2),
            **STARTUP_LOGS["SPEC_DECODE"],
        },
    }
    return {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "gpu": "NVIDIA H200 (141 GB)",
        "vllm_version": single.get("vllm_version", "0.18.0+rhaiv.14"),
        "date": "2026-09-29",
        "cluster": "Red Hat internal H200 cluster",
        "source": f"built by scripts/build_benchmark_file.py from {raw_dir.relative_to(ROOT)}",
        "notes": (
            "Single stream: 5 requests per setup, one at a time, same prompt, temperature 0, "
            "256 output tokens each, all with enforce_eager off and CUDA graphs on. Measured through a "
            "port-forward from a laptop, so latencies include that network hop. throughput_tps is the mean "
            "of per-run output tokens per second over the whole request. Spec Decode's first run was cold "
            "(49.6 tok/s). Acceptance counters are cumulative since pod start and include the temperature "
            "0.7 quality runs."
        ),
        "variants": variants,
        "history": {
            "SPEC_DECODE_enforce_eager": {
                "date": "2026-09-29",
                "throughput_tps": 40.0, "avg_latency_ms": 5543.9, "p95_latency_ms": 8646.0,
                "enforce_eager": True, "temperature": 0.7, "max_tokens": 256, "requests": 20,
                "note": "First run: enforce_eager on (no torch.compile, no CUDA graphs), temperature 0.7.",
            }
        },
    }


if __name__ == "__main__":
    raw = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "bench" / "raw" / "2026-09-29").resolve()
    out = main(raw)
    (ROOT / "benchmark_results.json").write_text(json.dumps(out, indent=2) + "\n")
    for key, v in out["variants"].items():
        secs = v["avg_latency_ms"] / 1000
        print(f"{key:12s} {v['throughput_tps']:6.1f} tok/s  {secs:5.2f} s  {v['gpus']} GPU")
