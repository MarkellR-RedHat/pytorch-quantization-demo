"""Capture every Ask preset from one vLLM endpoint, streamed, with timings. Standard library only.

usage: python3 capture_presets.py <VARIANT> <chat-completions URL> <served model name>
writes results-2/quality/<VARIANT>/<scenario>.json, plus 5 samples at 0.7 for the logic puzzle
"""

import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

PROMPTS = {
    "complex_reasoning": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? "
    "Explain your reasoning step by step.",
    "code_generation": "Write a Python function that returns the second largest number in a list. Handle edge cases.",
    "summarization": "Summarize the key trade-offs of model quantization for production LLM deployments "
    "in 3 bullet points.",
    "polite_decline": "Write a short, polite reply declining a meeting on Friday at 3pm, and suggest next week instead.",
    "quick_fact": "What is the capital of Australia, and why isn't it Sydney? Answer in two sentences.",
    "logic_puzzle": "Alice, Bob and Carol each have one meeting, on Monday, Tuesday or Wednesday, each on a "
    "different day. Alice's isn't on Monday. Bob's is the day after Alice's. Which day is Carol's? "
    "Explain step by step.",
    "long_explanation": "Explain the KV cache to a new engineer in about 300 words.",
    "json_extraction": "Extract the name, company and meeting date from this message as JSON: \"Hi, this is Sam "
    "Ortiz from Acme Robotics. Can we meet on October 21 to review the pilot?\"",
}
MAX_TOKENS = 1024
SAMPLED = "logic_puzzle"
N_SAMPLES = 5


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ask(url: str, model: str, prompt: str, temperature: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.perf_counter()
    first, text, usage, finish = None, [], None, None
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    first = first or time.perf_counter()
                    text.append(piece)
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    tokens = (usage or {}).get("completion_tokens")
    return {
        "response_text": "".join(text),
        "usage": usage,
        "finish_reason": finish,
        "ttft_ms": round((first - start) * 1000, 1) if first else None,
        "total_ms": round((end - start) * 1000, 1),
        "tokens_per_second": round(tokens / (end - start), 1) if tokens else None,
    }


def main(variant: str, url: str, model: str) -> None:
    out = Path("results-2/quality") / variant
    out.mkdir(parents=True, exist_ok=True)
    meta = {"variant": variant, "model": model, "max_tokens": MAX_TOKENS, "endpoint": "port-forward", "stream": True}
    for scenario, prompt in PROMPTS.items():
        result = ask(url, model, prompt, 0)
        record = {**meta, "scenario": scenario, "prompt": prompt, "temperature": 0, **result, "captured_at": now()}
        (out / f"{scenario}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{variant:12s} {scenario:18s} {result['usage'] and result['usage'].get('completion_tokens')} tokens "
              f"ttft {result['ttft_ms']} ms  total {result['total_ms']} ms  finish {result['finish_reason']}")
    samples = [ask(url, model, PROMPTS[SAMPLED], 0.7) for _ in range(N_SAMPLES)]
    record = {**meta, "scenario": SAMPLED, "prompt": PROMPTS[SAMPLED], "temperature": 0.7, "n": N_SAMPLES,
              "responses": [s["response_text"] for s in samples], "runs": samples, "captured_at": now()}
    (out / f"{SAMPLED}_samples_t0.7.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{variant:12s} {SAMPLED} x{N_SAMPLES} at 0.7 saved")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
