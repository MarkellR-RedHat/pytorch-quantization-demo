#!/usr/bin/env python3
"""
Benchmark runner for capturing real H200 numbers.

Hits vLLM endpoints, captures latency/throughput/memory metrics,
and updates simulation.py baseline_metrics with real data.

Usage:
    python scripts/benchmark.py --endpoint http://benchmark-int4:8080 --variant INT4
    python scripts/benchmark.py --endpoint http://benchmark-fp16:8080 --variant FP16
    python scripts/benchmark.py --endpoint http://benchmark-spec:8080 --variant SPEC_DECODE
    python scripts/benchmark.py --all  # runs all variants sequentially
"""

import argparse
import json
import time
import statistics
import subprocess
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests


PROMPTS = [
    "Explain the trade-offs of model quantization for production LLM deployments.",
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain step by step.",
    "Write a Python function that returns the second largest number in a list. Handle edge cases.",
    "Compare and contrast microservices architecture with monolithic architecture for a high-traffic web application.",
    "Summarize the key concepts of zero trust security in three bullet points.",
]

CLUSTER_ENDPOINTS = {
    "INT4": "http://benchmark-int4.user-mrawls.svc.cluster.local:8080",
    "FP16": "http://benchmark-fp16.user-mrawls.svc.cluster.local:8080",
    "SPEC_DECODE": "http://benchmark-spec.user-mrawls.svc.cluster.local:8080",
}


def run_benchmark(endpoint: str, variant: str, num_requests: int = 20) -> dict:
    print(f"\n{'='*60}")
    print(f"Benchmarking {variant} at {endpoint}")
    print(f"{'='*60}")

    model_name = f"llama-70b-{variant.lower().replace('_', '-')}"
    url = f"{endpoint}/v1/completions"

    latencies = []
    token_counts = []
    errors = 0

    for i, prompt in enumerate(PROMPTS * (num_requests // len(PROMPTS) + 1)):
        if len(latencies) >= num_requests:
            break

        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7,
        }

        try:
            start = time.perf_counter()
            resp = requests.post(url, json=payload, timeout=120)
            elapsed_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()

            data = resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            latencies.append(elapsed_ms)
            token_counts.append(tokens)
            print(f"  [{len(latencies)}/{num_requests}] {elapsed_ms:.1f}ms, {tokens} tokens")
        except Exception as e:
            errors += 1
            print(f"  [ERROR] {e}")
            if errors > 5:
                print("Too many errors, stopping.")
                break

    if not latencies:
        print(f"No successful requests for {variant}")
        return {}

    metrics_url = f"{endpoint}/metrics"
    gpu_memory_gb = None
    try:
        resp = requests.get(metrics_url, timeout=10)
        for line in resp.text.split("\n"):
            if "vllm:gpu_cache_usage_perc" in line and not line.startswith("#"):
                gpu_pct = float(line.split()[-1])
                break
    except Exception:
        pass

    avg_latency = statistics.mean(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    total_tokens = sum(token_counts)
    total_time_s = sum(latencies) / 1000
    throughput = total_tokens / total_time_s if total_time_s > 0 else 0
    avg_tokens = statistics.mean(token_counts) if token_counts else 0

    results = {
        "variant": variant,
        "avg_latency_ms": round(avg_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "throughput_tps": round(throughput, 1),
        "avg_tokens_per_request": round(avg_tokens, 1),
        "total_requests": len(latencies),
        "errors": errors,
    }

    print(f"\n--- {variant} Results ---")
    print(f"  Avg Latency:  {results['avg_latency_ms']}ms")
    print(f"  P95 Latency:  {results['p95_latency_ms']}ms")
    print(f"  Throughput:   {results['throughput_tps']} tokens/sec")
    print(f"  Requests:     {results['total_requests']} ({errors} errors)")

    return results


def get_gpu_memory(endpoint: str) -> float | None:
    try:
        resp = requests.get(f"{endpoint}/metrics", timeout=10)
        for line in resp.text.split("\n"):
            if "gpu_memory_used_bytes" in line and not line.startswith("#"):
                bytes_used = float(line.split()[-1])
                return round(bytes_used / (1024**3), 1)
    except Exception:
        pass
    return None


def update_simulation(results: list[dict]):
    sim_path = Path(__file__).parent.parent / "app" / "simulation.py"
    if not sim_path.exists():
        print(f"simulation.py not found at {sim_path}")
        return

    print(f"\n{'='*60}")
    print("Updating simulation.py baseline_metrics")
    print(f"{'='*60}")

    for r in results:
        variant = r["variant"]
        latency = r["avg_latency_ms"]
        tps = r["throughput_tps"]
        print(f"  {variant}: latency={latency}ms, throughput={tps} tps")

    output_path = Path(__file__).parent.parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print("Update simulation.py baseline_metrics manually with these values.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM model variants on H200")
    parser.add_argument("--endpoint", help="vLLM endpoint URL")
    parser.add_argument("--variant", choices=["FP16", "INT4", "SPEC_DECODE"])
    parser.add_argument("--all", action="store_true", help="Run all variants")
    parser.add_argument("--requests", type=int, default=20, help="Number of requests per variant")
    args = parser.parse_args()

    if args.all:
        results = []
        for variant, endpoint in CLUSTER_ENDPOINTS.items():
            r = run_benchmark(endpoint, variant, args.requests)
            if r:
                r["gpu_memory_gb"] = get_gpu_memory(endpoint)
                results.append(r)
        if results:
            update_simulation(results)
    elif args.endpoint and args.variant:
        r = run_benchmark(args.endpoint, args.variant, args.requests)
        if r:
            r["gpu_memory_gb"] = get_gpu_memory(args.endpoint)
            update_simulation([r])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
