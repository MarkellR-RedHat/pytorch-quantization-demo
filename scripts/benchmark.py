#!/usr/bin/env python3
"""
Single-stream benchmark for one vLLM endpoint: TTFT, time per output token, end-to-end latency.

Every request uses temperature 0 and ignore_eos with a fixed max_tokens, so every variant generates the
same number of tokens and the comparison is fair. Warmup requests are discarded.

Usage:
    python scripts/benchmark.py --endpoint http://int4-llama70b:8080 --variant INT4 --model llama-70b-int4
    BENCH_ENDPOINT=http://... python scripts/benchmark.py --variant FP16 --model llama-70b-fp16

This measures one stream at a time. For behaviour under load use `vllm bench serve` (printed at the end).
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

PROMPTS = [
    "Explain the trade-offs of model quantization for production LLM deployments.",
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? "
    "Explain step by step.",
    "Write a Python function that returns the second largest number in a list. Handle edge cases.",
    "Compare microservices and monolithic architecture for a high-traffic web application.",
    "Summarize the key concepts of zero trust security in three bullet points.",
]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(pct / 100 * len(ordered))) - 1]


def stream_once(client: httpx.Client, url: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft = None
    chunks = 0
    usage = {}
    with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if data.get("usage"):
                usage = data["usage"]
            if data.get("choices") and data["choices"][0].get("text"):
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
    e2e = time.perf_counter() - start
    tokens = int(usage.get("completion_tokens") or chunks)
    ttft = ttft if ttft is not None else e2e
    tpot = (e2e - ttft) / (tokens - 1) if tokens > 1 else 0.0
    return {"e2e_ms": e2e * 1000, "ttft_ms": ttft * 1000, "tpot_ms": tpot * 1000, "tokens": tokens}


def run_benchmark(
    endpoint: str, variant: str, model: str, requests: int, warmup: int, max_tokens: int
) -> dict:
    url = f"{endpoint.rstrip('/')}/v1/completions"
    print(f"\nBenchmarking {variant} ({model}) at {url}: {warmup} warmup + {requests} measured, 1 stream")
    results, errors = [], 0
    with httpx.Client(timeout=300.0) as client:
        for i in range(warmup + requests):
            prompt = PROMPTS[i % len(PROMPTS)]
            try:
                r = stream_once(client, url, model, prompt, max_tokens)
            except (httpx.HTTPError, ValueError) as e:
                errors += 1
                print(f"  [ERROR] {e}")
                if errors > 5:
                    sys.exit("Too many errors, stopping.")
                continue
            tag = "warmup" if i < warmup else f"{i - warmup + 1}/{requests}"
            print(f"  [{tag}] e2e {r['e2e_ms']:.0f} ms, ttft {r['ttft_ms']:.0f} ms, {r['tokens']} tokens")
            if i >= warmup:
                results.append(r)

    if not results:
        sys.exit(f"No successful requests for {variant}")

    e2e = [r["e2e_ms"] for r in results]
    tokens = sum(r["tokens"] for r in results)
    summary = {
        "variant": variant,
        "model": model,
        "concurrency": 1,
        "temperature": 0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "warmup_requests": warmup,
        "total_requests": len(results),
        "errors": errors,
        "avg_latency_ms": round(statistics.mean(e2e), 1),
        "p50_latency_ms": round(percentile(e2e, 50), 1),
        "p95_latency_ms": round(percentile(e2e, 95), 1),
        "mean_ttft_ms": round(statistics.mean(r["ttft_ms"] for r in results), 1),
        "mean_tpot_ms": round(statistics.mean(r["tpot_ms"] for r in results), 2),
        "throughput_tps": round(tokens / (sum(e2e) / 1000), 1),
        "avg_tokens_per_request": round(tokens / len(results), 1),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Single-stream benchmark for one vLLM endpoint")
    parser.add_argument("--endpoint", default=os.environ.get("BENCH_ENDPOINT"), help="vLLM base URL")
    parser.add_argument("--variant", required=True, choices=["FP16", "FP8", "INT4", "SPEC_DECODE"])
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", help="output JSON (default bench/<VARIANT>/single_stream.json)")
    args = parser.parse_args()
    if not args.endpoint:
        parser.error("--endpoint or BENCH_ENDPOINT is required")

    summary = run_benchmark(
        args.endpoint, args.variant, args.model, args.requests, args.warmup, args.max_tokens
    )
    out = Path(args.out or Path(__file__).parent.parent / "bench" / args.variant / "single_stream.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")
    print("Model weight size: read 'Model weights take X GiB' from the vLLM server startup log.")
    print("\nFor behaviour under load, sweep concurrency with vLLM's own tool, for example:")
    for n in (1, 8, 32, 64):
        print(
            f"  vllm bench serve --backend vllm --base-url {args.endpoint} --model {args.model} "
            f"--dataset-name random --random-input-len 128 --random-output-len {args.max_tokens} "
            f"--max-concurrency {n} --num-prompts {max(50, n * 4)} --save-result "
            f"--result-filename bench/{args.variant}/c{n}.json"
        )


if __name__ == "__main__":
    main()
