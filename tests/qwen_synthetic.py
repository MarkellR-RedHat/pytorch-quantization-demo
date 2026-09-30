"""A synthetic results-qwen/ folder in the layout RUN-QWEN.md asks the work laptop for, so the Qwen builder
and the app can be tested before the real files land. Every number here is made up and marked so."""

import json
from pathlib import Path

from app.quality import PROMPTS

IMAGE = "registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:c056e61672b6aea489ad5dde0bd2f8497230f5333e87f7cf6c494eba3bfdc808"

# setup -> (isvc name, checkpoint, device resource, loaded GiB, kv tokens, single-stream tok/s at t0, at t0.7)
SETUPS = {
    "BF16": ("qwen-bf16", "Qwen/Qwen3.8-27B", "nvidia.com/gpu", 51.89, 403613, 60.0, 59.0),
    "SPEC_DECODE": ("qwen-mtp", "Qwen/Qwen3.8-27B", "nvidia.com/gpu", 51.89, 403613, 78.0, 72.0),
    "FP8": ("qwen-fp8", "Qwen/Qwen3.8-27B-FP8", "nvidia.com/mig-3g.71gb", 27.3, 120000, 61.0, 60.5),
    "INT4": ("qwen-int4", "RedHatAI/Qwen3.8-27B-INT4", "nvidia.com/mig-3g.71gb", 15.4, 150000, 52.0, 51.0),
    "INT4_35": ("qwen-int4-35gb", "RedHatAI/Qwen3.8-27B-INT4", "nvidia.com/mig-2g.35gb", 15.4, 40000, 40.0, 39.5),
}
SWEEP = {  # setup -> concurrency -> (total output tok/s, p95 tpot ms); None = the point failed
    "BF16": {1: 58, 8: 400, 16: 720, 32: 1200, 64: 1900},
    "SPEC_DECODE": {1: 75, 8: 420, 16: 700, 32: 1050, 64: 1400},
    "FP8": {1: 59, 8: 380, 16: 640, 32: 950, 64: 1250},
    "INT4": {1: 50, 8: 330, 16: 560, 32: 800, 64: 980},
    "INT4_35": {1: 39, 8: 240, 16: 380, 32: 480, 64: None},
}
TPOT = {1: 18, 8: 21, 16: 24, 32: 31, 64: 44}


def bench_json(model: str, tput: float, tpot: float, n: int, c: int) -> dict:
    return {
        "date": "20261002-010203", "backend": "vllm", "model_id": model, "num_prompts": n, "max_concurrency": c,
        "completed": n, "failed": 0, "total_input_tokens": n * 512, "total_output_tokens": n * 256,
        "output_throughput": tput, "mean_ttft_ms": 40.0 + c, "median_ttft_ms": 35.0 + c, "p95_ttft_ms": 60.0 + c,
        "mean_tpot_ms": tpot - 1, "median_tpot_ms": tpot - 1.5, "p95_tpot_ms": tpot, "p99_tpot_ms": tpot + 4,
        "mean_e2el_ms": 256 * tpot, "median_e2el_ms": 250 * tpot, "p95_e2el_ms": 270 * tpot,
    }


def startup_text(isvc: str, checkpoint: str, gib: float, kv: int, spec: bool) -> str:
    args = {"port": 8080, "model": checkpoint, "max_model_len": 32768, "served_model_name": [isvc],
            "default_chat_template_kwargs": {"enable_thinking": False}}
    if spec:
        args.update({"spec_method": "mtp", "spec_tokens": 4})
    kernel = "Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8" if "FP8" in checkpoint else (
        "Using MacheteLinearKernel for GPTQMarlinLinearMethod" if "INT4" in checkpoint else "")
    lines = [
        f"INFO [utils.py:1] non-default args: {args}",
        f"INFO [core.py:114] Initializing a V1 LLM engine (v0.24.0+rhaiv.13) with config: model='{checkpoint}', "
        + ("speculative_config=SpeculativeConfig(method='mtp', model='" + checkpoint + "', num_spec_tokens=4), " if spec else "speculative_config=None, ")
        + "dtype=torch.bfloat16, max_seq_len=32768, tensor_parallel_size=1, quantization=None",
        "INFO [flash_attn.py:599] Using FlashAttention version 3",
        kernel,
        f"INFO [model_runner.py:1] Model loading took {gib} GiB memory and 12.3 seconds",
        f"INFO [kv_cache_utils.py:1] GPU KV cache size: {kv:,} tokens",
        "INFO [kv_cache_utils.py:2] Maximum concurrency for 32,768 tokens per request: 12.3x",
    ]
    if spec:
        lines.append("WARNING [config.py:1] Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer")
    return "\n".join(line for line in lines if line) + "\n"


def write(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("manifests", "evals", "logs", "captures", "live", "scripts"):
        (root / sub).mkdir(exist_ok=True)
    for key, (isvc, checkpoint, resource, gib, kv, tps0, tps07) in SETUPS.items():
        pod = f"{isvc}-predictor-abc12-xyz34"
        sweeps = root / "sweeps" / key
        sweeps.mkdir(parents=True, exist_ok=True)
        for t, tps in (("0", tps0), ("0.7", tps07)):
            d = bench_json(isvc, tps, 1000 / tps, 30, 1)
            d.update({"total_input_tokens": 6434, "total_output_tokens": 5900, "median_e2el_ms": 2200.0, "mean_e2el_ms": 3300.0, "p95_e2el_ms": 9000.0})
            (sweeps / f"single-t{t}.json").write_text(json.dumps(d))
        for c, tput in SWEEP[key].items():
            n = max(64, c * 8)
            if tput is None:
                (sweeps / f"c{c}-failed.txt").write_text("torch.OutOfMemoryError: CUDA out of memory\n")
                (sweeps / f"sharegpt-c{c}-failed.txt").write_text("torch.OutOfMemoryError: CUDA out of memory\n")
                continue
            (sweeps / f"c{c}.json").write_text(json.dumps(bench_json(isvc, tput, TPOT[c], n, c)))
            (sweeps / f"sharegpt-c{c}.json").write_text(json.dumps(bench_json(isvc, tput * 0.9, TPOT[c] - 2, n, c)))
            if key == "FP8" and c == 32:  # two slices of one card loaded together: a measured per-H200 point
                for side in ("a", "b"):
                    (sweeps / f"c{c}-two-slices-{side}.json").write_text(json.dumps(bench_json(isvc, tput * 0.93, TPOT[c] + 3, n, c)))
        if key == "SPEC_DECODE":
            for k, rate in ((1, 0.90), (2, 0.82), (4, 0.72)):
                for t in ("0", "0.7") if k == 4 else ("0",):
                    r = rate if t == "0" else rate - 0.08
                    drafts, draft_tokens = 1300, 1300 * k
                    accepted = round(draft_tokens * r)
                    for when, base in (("before", 0), ("after", 1)):
                        text = "\n".join([
                            f'vllm:spec_decode_num_drafts_total{{engine="0",model_name="{isvc}"}} {base * drafts}.0',
                            f'vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="{isvc}"}} {base * draft_tokens}.0',
                            f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="{isvc}"}} {base * accepted}.0',
                        ] + [
                            f'vllm:spec_decode_num_accepted_tokens_per_pos{{engine="0",model_name="{isvc}",position="{p}"}} '
                            f"{base * round(drafts * r ** (p + 1))}.0"
                            for p in range(k)
                        ]) + "\n"
                        (root / "logs" / f"spec-metrics-{when}-t{t}-k{k}.txt").write_text(text)
                    if k != 4:
                        (sweeps / f"single-t0-k{k}.json").write_text(json.dumps(bench_json(isvc, tps0 * (0.85 + 0.05 * k), 14.0, 30, 1)))
            (root / "logs" / "spec-metrics-full.txt").write_text("# HELP vllm:spec_decode_num_drafts_total ...\n")
        (root / "logs" / f"{pod}-startup-full.txt").write_text(startup_text(isvc, checkpoint, gib, kv, key == "SPEC_DECODE"))
        (root / "logs" / f"{pod}-startup.txt").write_text(startup_text(isvc, checkpoint, gib, kv, key == "SPEC_DECODE"))
        (root / "logs" / f"{pod}-version.txt").write_text("vllm 0.24.0+rhaiv.13\ntorch 2.10.0\ntransformers 5.16.1\nname, driver_version, memory.total [MiB]\nNVIDIA H200, 580.126.20, "
                                                          + ("143771 MiB" if resource == "nvidia.com/gpu" else "72704 MiB" if "71gb" in resource else "36352 MiB") + "\n")
        (root / "logs" / f"{pod}-image.txt").write_text(IMAGE + "\n")
        (root / "logs" / f"{key}-gpu-util-c64.csv").write_text("index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB]\n0, NVIDIA H200, 97 %, 60000 MiB, 143771 MiB\n")
        (root / "logs" / f"{pod}-gpu.txt").write_text(json.dumps({"limits": {resource: "1"}, "requests": {resource: "1", "cpu": "4", "memory": "32Gi"}})
                                                      + "\nGPU 0: NVIDIA H200 (UUID: GPU-x)\nname, memory.total [MiB], memory.used [MiB]\nNVIDIA H200, "
                                                      + ("143771 MiB" if resource == "nvidia.com/gpu" else "72704 MiB" if "71gb" in resource else "36352 MiB") + ", 60000 MiB\n")
        (root / "manifests" / f"isvc-{isvc}.yaml").write_text(f"apiVersion: serving.kserve.io/v1beta1\nkind: InferenceService\nmetadata:\n  name: {isvc}\n")
        if key in ("BF16", "FP8", "INT4"):
            for task, folder, score, n in (("gsm8k_cot", key, 0.93 + 0.005 * len(key), 1319), ("mmlu_pro", f"{key}-mmlu_pro", 0.70 - 0.01 * len(key), 280)):
                out = root / "evals" / folder / isvc
                out.mkdir(parents=True, exist_ok=True)
                results = {task: {f"exact_match,{'flexible-extract' if task == 'gsm8k_cot' else 'custom-extract'}": score,
                                  f"exact_match_stderr,{'flexible-extract' if task == 'gsm8k_cot' else 'custom-extract'}": 0.0061 if task == "gsm8k_cot" else 0.0281,
                                  "exact_match,strict-match": score - 0.002, "exact_match_stderr,strict-match": 0.0062}}
                samples = {task: {"original": 1319 if task == "gsm8k_cot" else 0, "effective": n}}
                configs = {task: {"num_fewshot": 8 if task == "gsm8k_cot" else 5}}
                if task == "mmlu_pro":
                    for s in ("biology", "business", "chemistry", "computer_science", "economics", "engineering", "health",
                              "history", "law", "math", "other", "philosophy", "physics", "psychology"):
                        results[f"mmlu_pro_{s}"] = {"exact_match,custom-extract": score, "exact_match_stderr,custom-extract": 0.1}
                        samples[f"mmlu_pro_{s}"] = {"original": 500, "effective": 20}
                        configs[f"mmlu_pro_{s}"] = {"num_fewshot": 5}
                (out / "results_2026-10-02T01-02-03.json").write_text(json.dumps({
                    "results": results, "n-samples": samples, "configs": configs, "config": {"limit": None if task == "gsm8k_cot" else 20.0},
                }))
        if True:  # every setup is recorded, INT4_35 included (the INT4 card takes its recordings from it)
            cap = root / "captures" / key
            cap.mkdir(parents=True, exist_ok=True)
            for scenario, prompt in PROMPTS.items():
                (cap / f"{scenario}.json").write_text(json.dumps({
                    "variant": key, "model": isvc, "max_tokens": 1024, "endpoint": "port-forward", "stream": True,
                    "scenario": scenario, "prompt": prompt, "temperature": 0,
                    "response_text": "The farmer has 9 sheep left. Carol's meeting is on Monday. {\"name\": \"Sam Ortiz\", \"company\": \"Acme Robotics\", \"date\": \"October 21\"}",
                    "usage": {"completion_tokens": 40}, "finish_reason": "stop", "ttft_ms": 300.0, "total_ms": 1200.0,
                    "tokens_per_second": tps0, "captured_at": "2026-10-02T01:02:03-04:00",
                }))
            (cap / "logic_puzzle_samples_t0.7.json").write_text(json.dumps({
                "variant": key, "n": 5, "temperature": 0.7, "responses": ["Carol's meeting is on Monday."] * 5,
            }))
    (root / "notes.txt").write_text("SYNTHETIC results-qwen for tests: every number is made up.\nthinking: off\n")
    return root


if __name__ == "__main__":
    import sys

    print(write(Path(sys.argv[1] if len(sys.argv) > 1 else "results-qwen-synthetic")))
