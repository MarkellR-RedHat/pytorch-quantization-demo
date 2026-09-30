# Run: the Qwen3.8-27B track

One package for everything the Qwen track needs for the repo, the demo and the slides. Same shape as RUN-TODAY-FINAL.md. Everything runs on **vLLM 0.24.0+rhaiv.13**, the image behind the cluster's `vllm-cuda-runtime-template` (`registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:c056e61672b6aea489ad5dde0bd2f8497230f5333e87f7cf6c494eba3bfdc808`), pinned in every manifest below. The Llama track ran on 0.18.0+rhaiv.14 because that was the benchmark runtime at the time. The reason for 0.24 is speculative decoding: Qwen3.8-27B carries its own MTP head and 0.24 serves it with `--spec-method=mtp` (the 0.18 CLI has neither the method nor the `--spec-method/--spec-model/--spec-tokens` flags; DSpark, the Red Hat speculator, needs 0.29+ and doesn't load on any build we have). `--max-model-len 32768` everywhere: Qwen3.8's native context is 262K, 32K is plenty for the demo and keeps the KV cache inside a slice.

**GPU budget:** never more than one full H200 plus two MIG slices at once; no standing Kueue reservation (the pods borrow slices from the unreserved cohort through the namespace's queue); delete each pod the moment its tests finish. Expected use: Phase 0 about 30 minutes on two slices; Phase A about 75 minutes on the full H200 for BF16, then about 90 minutes for BF16+MTP (the k=1 and k=2 passes add the extra 15); Phase B about 75 minutes on two 71 GB slices in parallel, then INT4_35 about 70 minutes on one 35 GB slice after the 71 GB INT4 pod is deleted (still at most two slices at once). About 3 full-H200-hours plus 5 slice-hours, about 6 to 7 hours of wall clock.

**Harness parity with the Llama track, kept exact:** the same 30 ShareGPT prompts for single-stream (`--num-prompts 30 --max-concurrency 1`, temperature 0 and 0.7), the same `vllm bench serve` flags (`--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99`), the same random sweep (512 in / 256 out with `--ignore-eos`; not 1024), the same ShareGPT sweep, the same concurrency set (1, 8, 16, 32, 64) for every setup. Nothing in `vllm bench serve` was renamed between 0.18 and 0.24 as far as the feasibility run saw; if a flag is rejected, drop it, note the exact flag in `notes.txt`, and keep everything else the same.

The four setups and their names in every file: **BF16** = `Qwen/Qwen3.8-27B` on one full H200; **SPEC_DECODE** = the same checkpoint with its MTP head on the same full H200 (`--spec-method=mtp --spec-tokens=4`); **FP8** = `Qwen/Qwen3.8-27B-FP8` on a 71 GB slice; **INT4** = `RedHatAI/Qwen3.8-27B-INT4` on a 71 GB slice, the same profile as FP8 so both get the same share of the GPU's compute: it carries the evals (accuracy is a property of the checkpoint, not the slice) and is the apples-to-apples footnote against FP8. **INT4_35** = the same INT4 checkpoint on a 35 GB slice, which is the INT4 card on the slides and in the demo ("when 71 GB is too much", 3 slices per H200): REQUIRED, the full set except evals (deployment record, warm-up, the eight preset recordings and puzzle samples, single-stream at 0 and 0.7, the sweep with failed points kept, gpu-util at c=64, startup logs). Spec decode counter snapshots are named `spec-metrics-{before,after}-t<T>-k<K>.txt` for K in 1, 2, 4 (5a's `summarize_results.py --track qwen` reads the `-k` suffix and assumes k=4 without it).

Results go in `results-qwen/`, laid out like the repo's `bench/raw/2026-09-29-r2/` so the folder drops in:

```
results-qwen/
  manifests/   the manifests as applied, the runtime if one was used, secret shapes with values blanked
  sweeps/<V>/  c<N>.json, sharegpt-c<N>.json, single-t0.json, single-t0.7.json (V = BF16, SPEC_DECODE, FP8, INT4, INT4_35); a failed point is c<N>-failed.txt, never a missing file
  evals/<V>/ and evals/<V>-mmlu_pro/  (V = BF16, FP8, INT4, SPEC_DECODE if run; none for INT4_35, see B2)
  logs/        startup logs (full and grep), versions, image digests, GPU facts, spec counters, gpu-util CSVs, phase-0 answers
  captures/<V>/  the eight presets and the logic-puzzle samples (V = BF16, SPEC_DECODE, FP8, INT4, INT4_35)
  live/        screenshots and errors from the live check
  scripts/     every script actually used, including any ad hoc loop
  notes.txt, summary.json
```

## What to send back, and what each file unblocks

| File in `results-qwen/` | Unblocks |
|---|---|
| `manifests/*.yaml`, `logs/<pod>-image.txt`, `logs/<pod>-gpu.txt`, `logs/<pod>-kueue.txt` | The deployment record: the manifests in the repo are what ran, the slice sizes on the slides are what the pod saw |
| `logs/<pod>-startup-full.txt`, `logs/<pod>-startup.txt`, `logs/<pod>-version.txt` | Weights per device, KV cache, kernels, attention backend, MTP lines, versions |
| `logs/phase0-*.txt`, `logs/phase0-answers.txt` | Gate 5: the four load-and-delete facts that decide slide 4 and the "which slice" column |
| `sweeps/<V>/c*.json`, `sharegpt-c*.json` | **Required, all five setups.** Under load, the slice-vs-full-GPU story; the INT4 card's load numbers are INT4_35's |
| `sweeps/<V>/single-t0.json`, `single-t0.7.json` | Single-request speed and in-pod TTFT per setup |
| `logs/spec-metrics-full.txt`, `logs/spec-metrics-{before,after}-t<T>-k<K>.txt`, `sweeps/SPEC_DECODE/single-t0-k{1,2}.json` | MTP acceptance per temperature, and how it moves with 1, 2 and 4 speculative tokens |
| `evals/<V>`, `evals/<V>-mmlu_pro` | Accuracy per setup; the INT4 card's accuracy is the 71 GB INT4 run's (accuracy doesn't depend on the slice), so INT4_35 has none |
| `logs/<V>-gpu-util-c64.csv` | Utilization during the 64-in-flight runs |
| `logs/mig-profiles.txt` | The MIG profiles as the cluster reports them, so "two 71 GB slices per H200" cites the cluster |
| `captures/<V>/*.json` | **Required, INT4_35 included.** The eight preset recordings per setup: the Qwen track's fallback and its recorded-by-plan columns; the INT4 card's recordings are INT4_35's |
| `live/*.png`, `live/errors.txt`, `live/preflight.txt` | Proof the live path works against 0.24 (last step, or Oct 19; see E) |
| `notes.txt`, `summary.json`, `scripts/` | Provenance: what ran where, what failed, what was retried, the scripts that produced everything; summary.json from `summarize_results.py results-qwen --track qwen` |

## Before you start

- Pull the repo (`git pull`). The four Qwen manifests are under `kubernetes/models/qwen/` once the `tracks` branch lands on main; until then use the copies in this file (same content plus the queue label). The capture script is `scripts/capture_presets.py` at current main (it takes an output-dir argument); a copy is inline below in case the pull is behind.
- Keep the namespace, cluster hostname, node names, usernames and tokens out of every file and screenshot. The `scrub` helper below strips them from `oc get -o yaml` exports; read what it writes before zipping.
- **Warm-up rule:** 5 throwaway requests on every pod before any measured run, and again after any restart. The Llama round had a cold c=1 on FP8 and on INT4_RH and a cold first capture on spec decode; none of that this time.
- **Retry rule:** if a `vllm bench serve` run fails and has to be rerun, restart the pod first (`oc delete pod`) or add `--seed <new number>`, and write it in `notes.txt`. The Llama round's AWQ ShareGPT c=64 retry hit a warm prefix cache and had to be excluded.
- **Thinking mode:** Qwen3 models can answer in a "thinking" mode. Everything here (benchmarks, evals, captures) must use the same setting. Decide once in Phase 0 by looking at the first answer: if it contains a `<think>` block, a `reasoning_content` field, or runs to the cap, turn thinking off everywhere (`EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}` in the capture script, the same key through `--extra-body` / `gen_kwargs` where a tool supports it) and write "thinking: off" in `notes.txt`; otherwise write "thinking: on (default, no think blocks seen)". Never mix.
- Tokenizer: `Qwen/Qwen3.8-27B` (not gated, no HF token needed). The FP8 and INT4 checkpoints ship their own tokenizers; `--tokenizer <that checkpoint id>` works inside their pods.
- The multimodal loader: Qwen3.8-27B is the VL architecture. The feasibility run loaded it on a full H200 at 51.89 GiB with 67.82 GiB left for KV cache. On a slice, if the startup log shows the vision encoder taking memory, add `--limit-mm-per-prompt '{"image": 0}'` and note it.
- Timing comes from `vllm bench serve` inside the pods (in-pod TTFT is in the bench JSON). Nothing measured through a port-forward is used for timing; the captures' timings are labeled as recorded through a port-forward, as on the Llama track.

```bash
NS=your-namespace   # the only line to edit: the namespace the pods run in
mkdir -p results-qwen/{manifests,evals,logs,live,scripts,captures} results-qwen/sweeps/{BF16,SPEC_DECODE,FP8,INT4,INT4_35}
cp RUN-QWEN.md results-qwen/scripts/   # the run list is itself the record of what was run

inpod() { oc exec -n $NS "$1" -c kserve-container -- bash -lc "$2"; }
podof() { oc get pods -n $NS -l serving.kserve.io/inferenceservice=$1 -o jsonpath='{.items[0].metadata.name}'; }
waitready() { oc wait -n $NS --for=condition=Ready isvc/$1 --timeout=30m; }
scrub() {   # an oc get -o yaml export with the cluster-specific noise removed and the namespace blanked
  NS=$NS python3 -c 'import os,sys,yaml; d=yaml.safe_load(sys.stdin); d.pop("status",None); m=d.setdefault("metadata",{}); [m.pop(k,None) for k in ("managedFields","uid","resourceVersion","creationTimestamp","generation","selfLink","namespace")]; m.get("annotations",{}).pop("kubectl.kubernetes.io/last-applied-configuration",None); print(yaml.safe_dump(d, sort_keys=False).replace(os.environ["NS"],"<namespace>"))'
}
migprofiles() {   # once: the MIG profiles as the cluster reports them (node names cut)
  oc get nodes -o json | python3 -c 'import sys,json; [print(sorted((k,v) for k,v in n["status"]["allocatable"].items() if "mig" in k or k=="nvidia.com/gpu")) for n in json.load(sys.stdin)["items"] if any("mig" in k or k=="nvidia.com/gpu" for k in n["status"]["allocatable"])]' > results-qwen/logs/mig-profiles.txt
  # if a GPU pod allows it, append the driver's own list: inpod <pod> "nvidia-smi mig -lgip" >> results-qwen/logs/mig-profiles.txt
}
record() {   # record <isvc name> <pod>: the deployment record for one setup (A.1 to A.5)
  oc get isvc $1 -n $NS -o yaml | scrub > results-qwen/manifests/isvc-$1.yaml
  oc get pod $2 -n $NS -o jsonpath='{.status.containerStatuses[*].imageID}{"\n"}' > results-qwen/logs/$2-image.txt
  { oc get pod $2 -n $NS -o jsonpath='{.spec.containers[?(@.name=="kserve-container")].resources}{"\n"}'; inpod $2 "nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv"; } > results-qwen/logs/$2-gpu.txt
  oc get workloads -n $NS -o jsonpath='{range .items[*]}{.metadata.name}{" admitted-by="}{.status.admission.clusterQueue}{" flavors="}{.status.admission.podSetAssignments[*].flavors}{"\n"}{end}' | grep -i "$1" > results-qwen/logs/$2-kueue.txt || true
  oc logs -n $NS $2 -c kserve-container > results-qwen/logs/$2-startup-full.txt
  grep -E "non-default args|Initializing|Model loading took|KV cache|Maximum concurrency|LinearKernel|Selected|kernel|enforce_eager|CUDA graph|attention backend|FlashAttention|quantization|spec|MTP|mtp|Warming" results-qwen/logs/$2-startup-full.txt > results-qwen/logs/$2-startup.txt
  inpod $2 "python -c 'import vllm,torch,transformers;print(\"vllm\",vllm.__version__);print(\"torch\",torch.__version__);print(\"transformers\",transformers.__version__)'; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv" > results-qwen/logs/$2-version.txt
}
warmup() {   # warmup <pod> <served name>: 5 throwaway requests before anything measured
  for i in 1 2 3 4 5; do inpod $1 "curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":32}'" > /dev/null; done
}
getsharegpt() { inpod "$1" "curl -sL -o /tmp/sharegpt.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json && ls -la /tmp/sharegpt.json"; }
specmetrics() { inpod "$1" "curl -s localhost:8080/metrics | grep -E '^vllm:spec_decode'" > results-qwen/logs/spec-metrics-$2.txt; }
gpuutil() { oc exec -n $NS $1 -c kserve-container -- nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv -l 2 > results-qwen/logs/$2-gpu-util-c64.csv; }

# the two loops every setup runs (single-stream, then the sweep); POD/NAME/TOK/V set by the phase
single() {
  for T in 0 0.7; do
    inpod $POD "mkdir -p /tmp/results && vllm bench serve --backend vllm --base-url http://localhost:8080 --model $NAME --tokenizer $TOK \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --dataset-name sharegpt --dataset-path /tmp/sharegpt.json --num-prompts 30 --max-concurrency 1 --temperature $T \
      --save-result --result-dir /tmp/results --result-filename single-t$T.json"
  done
}
sweep() {
  for C in 1 8 16 32 64; do
    N=$(( C*8 > 64 ? C*8 : 64 ))
    [ $C = 64 ] && echo "start the gpuutil capture now in the other terminal: gpuutil $POD $V"
    inpod $POD "mkdir -p /tmp/results && vllm bench serve --backend vllm --base-url http://localhost:8080 --model $NAME --tokenizer $TOK \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --dataset-name random --random-input-len 512 --random-output-len 256 --ignore-eos \
      --num-prompts $N --max-concurrency $C --save-result --result-dir /tmp/results --result-filename c$C.json" \
      || { oc logs -n $NS $POD -c kserve-container --tail=50 > results-qwen/sweeps/$V/c$C-failed.txt; echo "c$C failed, see sweeps/$V/c$C-failed.txt"; }
    inpod $POD "vllm bench serve --backend vllm --base-url http://localhost:8080 --model $NAME --tokenizer $TOK \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --dataset-name sharegpt --dataset-path /tmp/sharegpt.json \
      --num-prompts $N --max-concurrency $C --save-result --result-dir /tmp/results --result-filename sharegpt-c$C.json" \
      || { oc logs -n $NS $POD -c kserve-container --tail=50 > results-qwen/sweeps/$V/sharegpt-c$C-failed.txt; echo "sharegpt-c$C failed, see the -failed.txt file"; }
  done
  oc cp $NS/$POD:/tmp/results results-qwen/sweeps/$V
}
# a sweep point that fails (for example c=64 on the 35 GB slice) stays in the table as a -failed.txt with the last 50 log lines,
# so the table shows where a slice tops out; apply the retry rule only if the failure was transient
evals() {   # evals <URL> <served name> <V>, on the laptop with a port-forward up
  lm_eval --model local-chat-completions \
    --model_args model=$2,base_url=$1/v1/chat/completions,num_concurrent=32,tokenized_requests=False,tokenizer=Qwen/Qwen3.8-27B \
    --tasks gsm8k_cot --apply_chat_template --fewshot_as_multiturn --output_path results-qwen/evals/$3
  lm_eval --model local-chat-completions \
    --model_args model=$2,base_url=$1/v1/chat/completions,num_concurrent=32,tokenized_requests=False,tokenizer=Qwen/Qwen3.8-27B \
    --tasks mmlu_pro --limit 20 --apply_chat_template --fewshot_as_multiturn --output_path results-qwen/evals/$3-mmlu_pro
}
```

Copy these helpers into `results-qwen/scripts/helpers.sh` (`declare -f inpod scrub migprofiles record warmup getsharegpt specmetrics gpuutil single sweep evals > results-qwen/scripts/helpers.sh`) so the scripts folder holds what actually ran. Evals use `gsm8k_cot` (the generic 8-shot chain-of-thought task; `gsm8k_cot_llama` carries Llama's prompt format) and `mmlu_pro --limit 20`, the same limit as the Llama track; if thinking mode makes answers run long, add `--gen_kwargs max_gen_toks=1024` and note it. Accuracy across tracks isn't the story; within a track it is. `summary.json` comes from 5a's `summarize_results.py` (in this folder) at the end: `python3 summarize_results.py results-qwen --track qwen`. It accepts both `sweeps/`/`bench/` and `evals/`/`eval/` spellings and reads the `-k` suffix on the spec counter files.

The capture script (also `scripts/capture_presets.py` in the repo):

```bash
cat > capture_presets.py <<'PYEOF'
"""Capture every Ask preset from one vLLM endpoint, streamed, with timings. Standard library only.

usage: python3 capture_presets.py <VARIANT> <chat-completions URL> <served model name> [<output dir>]
writes <output dir>/<VARIANT>/<scenario>.json (default quality/), plus 5 samples at 0.7 for the logic puzzle
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
    "long_explanation": "Explain the KV cache in a transformer LLM to a new engineer in about 300 words.",
    "json_extraction": "Extract the name, company and meeting date from this message as JSON: \"Hi, this is Sam "
    "Ortiz from Acme Robotics. Can we meet on October 21 to review the pilot?\"",
}
MAX_TOKENS = 1024  # the same cap the dashboard's live Ask uses
EXTRA_BODY = {}  # Phase 0 decides: set {"chat_template_kwargs": {"enable_thinking": False}} if answers carry a <think> block or reasoning_content
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
        **EXTRA_BODY,
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


def main(variant: str, url: str, model: str, out_dir: str = "quality") -> None:
    out = Path(out_dir) / variant
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
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    main(*sys.argv[1:])
PYEOF
cp capture_presets.py results-qwen/scripts/
```

The 20 sheep samples at temperature 0.7 that the Llama round recorded aren't needed: the dashboard no longer shows a 20-of-20 line, and the logic-puzzle samples cover the sampled-answer check. Every capture must end with `finish stop`; note any `length`.

## The manifests (also `kubernetes/models/qwen/` in the repo, without the queue label)

All raw containers on the 0.24 image, tensor parallel 1, `--max-model-len 32768`. The queue label is the LocalQueue this namespace submits through (the Sep 29 run used `reserved`). Of the two at the end, `isvc-qwen-int4-35gb.yaml` is a measured setup (B2) and is in the repo too; `isvc-qwen-bf16-71gb.yaml` is a Phase 0 smoke load only and isn't.

`isvc-qwen-bf16.yaml`:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: qwen-bf16
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
  labels:
    kueue.x-k8s.io/queue-name: reserved
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
    containers:
      - name: kserve-container
        image: registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:c056e61672b6aea489ad5dde0bd2f8497230f5333e87f7cf6c494eba3bfdc808
        args:
          - --port=8080
          - --model=Qwen/Qwen3.8-27B
          - --served-model-name=qwen-bf16
          - --max-model-len=32768
        env:
          - name: HF_HOME
            value: /tmp/hf_home
        resources:
          requests:
            cpu: 4
            memory: 32Gi
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

`isvc-qwen-mtp.yaml` (the same full GPU as BF16; deploy it only after BF16 is deleted):
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: qwen-mtp
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
  labels:
    kueue.x-k8s.io/queue-name: reserved
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
    containers:
      - name: kserve-container
        image: registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:c056e61672b6aea489ad5dde0bd2f8497230f5333e87f7cf6c494eba3bfdc808
        args:
          - --port=8080
          - --model=Qwen/Qwen3.8-27B
          - --served-model-name=qwen-mtp
          - --max-model-len=32768
          - --spec-method=mtp
          - --spec-tokens=4
        env:
          - name: HF_HOME
            value: /tmp/hf_home
        resources:
          requests:
            cpu: 4
            memory: 32Gi
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

`isvc-qwen-fp8.yaml`:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: qwen-fp8
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
  labels:
    kueue.x-k8s.io/queue-name: reserved
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
    containers:
      - name: kserve-container
        image: registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:c056e61672b6aea489ad5dde0bd2f8497230f5333e87f7cf6c494eba3bfdc808
        args:
          - --port=8080
          - --model=Qwen/Qwen3.8-27B-FP8
          - --served-model-name=qwen-fp8
          - --max-model-len=32768
        env:
          - name: HF_HOME
            value: /tmp/hf_home
        resources:
          requests:
            cpu: 4
            memory: 32Gi
            nvidia.com/mig-3g.71gb: 1
          limits:
            nvidia.com/mig-3g.71gb: 1
```

`isvc-qwen-int4.yaml` (the measured INT4, on the same 71 GB profile as FP8):
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: qwen-int4
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
  labels:
    kueue.x-k8s.io/queue-name: reserved
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
    containers:
      - name: kserve-container
        image: registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:c056e61672b6aea489ad5dde0bd2f8497230f5333e87f7cf6c494eba3bfdc808
        args:
          - --port=8080
          - --model=RedHatAI/Qwen3.8-27B-INT4
          - --served-model-name=qwen-int4
          - --max-model-len=32768
        env:
          - name: HF_HOME
            value: /tmp/hf_home
        resources:
          requests:
            cpu: 4
            memory: 32Gi
            nvidia.com/mig-3g.71gb: 1
          limits:
            nvidia.com/mig-3g.71gb: 1
```

Smoke-load only, Phase 0 (copy `isvc-qwen-int4.yaml` and `isvc-qwen-bf16.yaml` and change these lines):
- `isvc-qwen-int4-35gb.yaml`: name `qwen-int4-35gb`, served name `qwen-int4-35gb`, resource `nvidia.com/mig-2g.35gb: 1` in both places.
- `isvc-qwen-bf16-71gb.yaml`: name `qwen-bf16-71gb`, served name `qwen-bf16-71gb`, resource `nvidia.com/mig-3g.71gb: 1` in both places.

Estimates before a log exists (the feasibility report's "~35 GB INT4 / ~70 GB FP8" are the Llama 70B figures): 27B is about 54 GB in BF16 (measured 51.89 GiB), about 27 GB in FP8 and about 15 GB in INT4. Phase 0 measures all of it.

---

## PHASE 0: Gate 5, the four load-and-delete facts (two slices at a time, no full GPU)

These decide slide 4 and the "which slice" column. Load, record, one answer, delete; no benchmarks here.

```bash
migprofiles
# pair 1: FP8 and INT4, both on 71 GB slices (the measured configuration)
oc apply -n $NS -f isvc-qwen-fp8.yaml -f isvc-qwen-int4.yaml
waitready qwen-fp8; waitready qwen-int4
FP8_POD=$(podof qwen-fp8); INT4_POD=$(podof qwen-int4)
record qwen-fp8 $FP8_POD; record qwen-int4 $INT4_POD
for P in $FP8_POD:qwen-fp8 $INT4_POD:qwen-int4; do
  inpod ${P%%:*} "curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"${P##*:}\",\"messages\":[{\"role\":\"user\",\"content\":\"A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?\"}],\"temperature\":0,\"max_tokens\":400}'" >> results-qwen/logs/phase0-answers.txt; echo >> results-qwen/logs/phase0-answers.txt
done
for P in $FP8_POD $INT4_POD; do cp results-qwen/logs/$P-startup.txt results-qwen/logs/phase0-$P.txt; done
# keep these two up if Phase B follows today; otherwise: oc delete inferenceservice qwen-fp8 qwen-int4 -n $NS

# pair 2: the two "does it fit" checks. Only after pair 1 is deleted, or one at a time next to a single 71 GB slice.
oc apply -n $NS -f isvc-qwen-int4-35gb.yaml -f isvc-qwen-bf16-71gb.yaml
waitready qwen-int4-35gb; waitready qwen-bf16-71gb
INT4_35_POD=$(podof qwen-int4-35gb); BF16_71_POD=$(podof qwen-bf16-71gb)
record qwen-int4-35gb $INT4_35_POD; record qwen-bf16-71gb $BF16_71_POD
# one sheep answer each, as above, appended to phase0-answers.txt
for P in $INT4_35_POD $BF16_71_POD; do cp results-qwen/logs/$P-startup.txt results-qwen/logs/phase0-$P.txt; done
oc delete inferenceservice qwen-int4-35gb qwen-bf16-71gb -n $NS     # B2 deploys qwen-int4-35gb again, once the 71 GB INT4 pod is gone
```

Read the first answer here and decide the thinking-mode setting (Before you start). Note in `notes.txt`, per setup: loaded GiB ("Model loading took"), KV cache GiB and tokens, the kernel lines, whether the answer said 9, and for the two fit checks whether they came up at all at 32K. For BF16 on the 71 GB slice the line that matters is `GPU KV cache size: N tokens` (or the OOM text if it never gets there): "why not BF16 on a slice?" is the first question this track invites, and that number is the answer. If FP8 or INT4 failed to load on the 71 GB slice, stop and send the logs; the rest depends on it.

## PHASE A: BF16 on one full H200, then the same GPU as BF16 + MTP

```bash
oc apply -n $NS -f isvc-qwen-bf16.yaml
waitready qwen-bf16; BF16_POD=$(podof qwen-bf16); record qwen-bf16 $BF16_POD; getsharegpt $BF16_POD; warmup $BF16_POD qwen-bf16
oc port-forward -n $NS pod/$BF16_POD 18011:8080 &
```

### A1. Recordings (REQUIRED, about 5 minutes)

```bash
python3 capture_presets.py BF16 http://localhost:18011/v1/chat/completions qwen-bf16 results-qwen/captures
```

### A2. Single-stream, then the sweep

```bash
V=BF16; POD=$BF16_POD; NAME=qwen-bf16; TOK=Qwen/Qwen3.8-27B
single; sweep      # start gpuutil $POD $V in a second terminal when the loop says so; Ctrl+C when c=64 ends
```

### A3. Accuracy, on the laptop (`pip install "lm_eval[api]"` once)

```bash
evals http://localhost:18011 qwen-bf16 BF16
```

### A4. Swap the GPU to MTP

```bash
oc delete inferenceservice qwen-bf16 -n $NS
oc apply -n $NS -f isvc-qwen-mtp.yaml
waitready qwen-mtp; MTP_POD=$(podof qwen-mtp); record qwen-mtp $MTP_POD; getsharegpt $MTP_POD; warmup $MTP_POD qwen-mtp
oc port-forward -n $NS pod/$MTP_POD 18012:8080 &
inpod $MTP_POD "curl -s localhost:8080/metrics" > results-qwen/logs/spec-metrics-full.txt   # what 0.24 calls the counters, incl. any per-position ones
```

### A5. MTP: single-stream with acceptance, BEFORE its sweep

```bash
V=SPEC_DECODE; POD=$MTP_POD; NAME=qwen-mtp; TOK=Qwen/Qwen3.8-27B
for T in 0 0.7; do
  specmetrics $POD before-t$T-k4
  inpod $POD "mkdir -p /tmp/results && vllm bench serve --backend vllm --base-url http://localhost:8080 --model $NAME --tokenizer $TOK \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --dataset-name sharegpt --dataset-path /tmp/sharegpt.json --num-prompts 30 --max-concurrency 1 --temperature $T \
    --save-result --result-dir /tmp/results --result-filename single-t$T.json"
  specmetrics $POD after-t$T-k4
done
python3 capture_presets.py SPEC_DECODE http://localhost:18012/v1/chat/completions qwen-mtp results-qwen/captures
sweep
```

### A6. MTP with 1 and 2 speculative tokens (temperature 0 only, single-stream)

vLLM warns that more than one MTP token per step may lower acceptance; an expert will ask. For K in 1 and 2: edit the manifest's `--spec-tokens=4` to `K`, re-apply (`oc delete inferenceservice qwen-mtp -n $NS; oc apply -n $NS -f isvc-qwen-mtp.yaml; waitready qwen-mtp`), `MTP_POD=$(podof qwen-mtp); record qwen-mtp $MTP_POD` again (rename the exports to `isvc-qwen-mtp-k$K.yaml` and `<pod>-startup-full.txt` keeps the new pod's name), warm up, then:

```bash
K=1   # then 2
specmetrics $MTP_POD before-t0-k$K
inpod $MTP_POD "mkdir -p /tmp/results && vllm bench serve --backend vllm --base-url http://localhost:8080 --model qwen-mtp --tokenizer Qwen/Qwen3.8-27B \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --dataset-name sharegpt --dataset-path /tmp/sharegpt.json --num-prompts 30 --max-concurrency 1 --temperature 0 \
  --save-result --result-dir /tmp/results --result-filename single-t0-k$K.json"
specmetrics $MTP_POD after-t0-k$K
oc cp $NS/$MTP_POD:/tmp/results/single-t0-k$K.json results-qwen/sweeps/SPEC_DECODE/
```

Then `oc delete inferenceservice qwen-mtp -n $NS`. The k=4 snapshots from A5 are the `-t<T>-k4` files.

## PHASE B: FP8 and INT4 on 71 GB slices, in parallel (the full set for each)

```bash
# deploy again if Phase 0 deleted them (oc apply -n $NS -f isvc-qwen-fp8.yaml -f isvc-qwen-int4.yaml; waitready both; record again)
FP8_POD=$(podof qwen-fp8); INT4_POD=$(podof qwen-int4)
for P in $FP8_POD:qwen-fp8 $INT4_POD:qwen-int4; do getsharegpt ${P%%:*}; warmup ${P%%:*} ${P##*:}; done
oc port-forward -n $NS pod/$FP8_POD 18013:8080 &
oc port-forward -n $NS pod/$INT4_POD 18014:8080 &
python3 capture_presets.py FP8  http://localhost:18013/v1/chat/completions qwen-fp8  results-qwen/captures
python3 capture_presets.py INT4 http://localhost:18014/v1/chat/completions qwen-int4 results-qwen/captures
for V in FP8 INT4; do
  case $V in
    FP8)  POD=$FP8_POD;  NAME=qwen-fp8;  TOK=Qwen/Qwen3.8-27B-FP8;     URL=http://localhost:18013;;
    INT4) POD=$INT4_POD; NAME=qwen-int4; TOK=RedHatAI/Qwen3.8-27B-INT4; URL=http://localhost:18014;;
  esac
  single; sweep; evals $URL $NAME $V
done
oc delete inferenceservice qwen-int4 -n $NS     # FP8 stays up for B2 and the live check; INT4_35 takes the freed slice, so still two slices at most
```

### B2. INT4_35 (REQUIRED): the same INT4 on a 35 GB slice, the INT4 card's speed and recordings

No evals here: accuracy is a property of the checkpoint, not the slice, and the 71 GB INT4 run above already carries them; the INT4 card takes its speed, load numbers and recordings from INT4_35 and its accuracy from the 71 GB run, and the 71 GB run is the apples-to-apples footnote against FP8 (a 2g slice has a different share of the GPU's compute than the 3g slice FP8 runs on).

```bash
oc apply -n $NS -f isvc-qwen-int4-35gb.yaml
waitready qwen-int4-35gb; INT4_35_POD=$(podof qwen-int4-35gb); record qwen-int4-35gb $INT4_35_POD; getsharegpt $INT4_35_POD; warmup $INT4_35_POD qwen-int4-35gb
oc port-forward -n $NS pod/$INT4_35_POD 18015:8080 &
python3 capture_presets.py INT4_35 http://localhost:18015/v1/chat/completions qwen-int4-35gb results-qwen/captures
V=INT4_35; POD=$INT4_35_POD; NAME=qwen-int4-35gb; TOK=RedHatAI/Qwen3.8-27B-INT4
single; sweep      # gpuutil $POD $V in the second terminal at c=64, as in A2; a c=64 that OOMs stays as c64-failed.txt (no retry, it's the slice topping out)
# run E's qwenB live check now, with FP8 on 18013 and INT4_35 on 18015 up, then:
oc delete inferenceservice qwen-int4-35gb qwen-fp8 -n $NS
```

## E. Live check, the last step of Phase A and of Phase B

Conditional on main having the track selector (b2 will send the hash when it lands; `git log --oneline -1` after `git pull` shows it). If it does: `.env` with `SIMULATION_MODE=false`, `TRACK=qwen`, the Qwen endpoints that are up (18011 BF16, 18012 MTP for Phase A; 18013 FP8 and 18015 INT4_35 in the INT4 slot for Phase B) with their served names, `make serve`, then from this folder:

```bash
PHASE=qwenA   # qwenB after Phase B
python3 scripts/preflight.py > results-qwen/live/preflight-$PHASE.txt
python3 screenshot_presenter.py --url http://localhost:8000 --out results-qwen/live --phase $PHASE
```

`screenshot_presenter.py` is in this folder (`~/Downloads/pytorch-quant-tests/screenshot_presenter.py`, copy it into the package's `scripts/`); it clicks every preset, saves `<phase>-<preset>.png`, `<phase>-badge.txt`, `<phase>-recorded-mode.png` after pressing R, and `errors.txt`. If main doesn't have the selector when the pods are up, write "live check deferred to Oct 19" in `notes.txt` and move on; the Llama-track app can still take one Qwen pod in its BF16 slot (`MODEL_BF16_ENDPOINT=http://localhost:18011/v1/chat/completions`, `MODEL_BF16_NAME=qwen-bf16`) for a quick "it streams" screenshot.

## F. Before zipping

- `notes.txt`: phases and dates, what ran on which device (full H200, mig-3g.71gb, mig-2g.35gb), thinking-mode setting, warm-ups done, every failure and retry (and whether the pod was restarted or the seed changed), any flag 0.24 didn't accept, the spec-tokens 1/2/4 runs. No cluster name, node name, namespace, hostname or username anywhere in the folder: `grep -ril "<the cluster's name>\|<ns>\|ocp-\|$USER" results-qwen` should print nothing.
- `manifests/`: the `record` exports for all six InferenceServices (five measured, INT4_35 included, plus the BF16-on-71 GB smoke load) plus, if any ServingRuntime was used, `oc get servingruntime <name> -o yaml | scrub`, and the shapes of `hf-token` and `storage-config` if used (`oc get secret <name> -o yaml | scrub` with `data:` values replaced by `<blanked>`).
- `scripts/`: this file, `helpers.sh`, `capture_presets.py`, and any loop typed by hand.
- `summary.json`: `python3 summarize_results.py results-qwen --track qwen` (the script is in this folder; copy it into `scripts/` too).
- Zip `results-qwen/` and send it back.

## G. If something doesn't work

- **MTP fails to load** (`--spec-method=mtp`): try `--spec-method=qwen3_5_mtp` (the feasibility run's spelling); if that fails too, save the full log as `logs/mtp-failed-startup-full.txt`, continue with the other three setups, and the Qwen track ships with three lanes (say so in notes).
- **A slice can't be borrowed** (pod Pending with a Kueue "insufficient quota" event): `oc describe pod <pod> | tail -20 > logs/<pod>-pending.txt`, wait 15 minutes once, then try the other slice profile (`mig-2g.35gb` for INT4 only), and note it. FP8 needs the 71 GB slice; if it can't get one, run FP8 on the full H200 after the MTP work and mark it "full GPU" in notes, so the slides don't claim a slice that wasn't measured.
- **lm_eval task missing** (`gsm8k_cot` or `mmlu_pro`): `pip install -U "lm_eval[api]"`; if still missing, run `gsm8k` (plain) and note the task name used; never skip both.
- **FP8 kernel falls back** (startup log says weight-only FP8 or no `CutlassFP8`/`Fp8` kernel line): keep the measurements, copy the kernel line into notes; the slide wording depends on it.
- **A `vllm bench serve` run dies mid-sweep**: restart the pod, warm up again, rerun that concurrency level only, note the retry (the retry rule above).
- **Thinking blocks appear after Phase 0's decision**: stop, set thinking off everywhere, redo whatever was measured with it on; never mix.
- **Thinking can't be turned off** (the `chat_template_kwargs` key is rejected or ignored): keep it on everywhere, write "thinking: on, could not disable" in notes with the error text, and expect longer answers; the captures stay valid as long as `finish_reason` is `stop`.
- **A pod OOMs on a slice** (CUDA out of memory in the log, or the pod restarts during a sweep): save the full log as `logs/<pod>-oom-startup-full.txt` (the `record` helper's full log covers startup OOMs), record the concurrency level it died at as a `-failed.txt` point, and don't retry that level; the number is the slice topping out, which is a result. Continue with the remaining setups.
