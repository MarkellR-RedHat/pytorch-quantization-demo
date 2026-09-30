# Not Every Question Needs Two GPUs

Llama 3.1 70B Instruct served by vLLM on NVIDIA H200s four ways: BF16 on two GPUs, FP8 on one, INT4 on one, and speculative decoding with an 8B draft on two. The demo sends one question to all of them at once, streams the answers side by side with their timing, and shows what each setup gets you, what it costs under load, and where a router fits. Built for the Demo Theater at PyTorch Conference North America 2026, San Jose.

## Results

vLLM `0.18.0+rhaiv.14` (the build in Red Hat AI), PyTorch 2.10.0, driver 580.126.20, September 29, 2026. One request at a time is `vllm bench serve` inside the pod on 30 ShareGPT prompts at temperature 0. Under load is the same tool at 1 to 64 requests in flight on 512-token random prompts asking for 256, and the number kept is output tokens per second per GPU with the p95 time per output token at or under 50 ms (20 tokens per second per user). Accuracy is `lm_eval`: GSM8K 8-shot chain of thought on all 1,319 questions, MMLU-Pro 5-shot on the first 20 questions of each of 14 subjects (280, standard error about ±2.8 points).

| Setup | GPUs | Tokens/s, one request | First token | GSM8K | MMLU-Pro | Tokens/s per GPU under load | Weights per GPU |
|---|---|---|---|---|---|---|---|
| BF16, tensor parallel 2 | 2 | 49.7 (1.00×) | 33 ms | 94.8% | 66.8% | 981 at 64 in flight | 65.7 GiB |
| FP8, Red Hat's build | 1 | 51.1 (1.03×) | 62 ms | 95.2% | 66.1% | 1,485 at 64 | 67.7 GiB |
| INT4, Red Hat's validated W4A16 build (GPTQ) | 1 | 44.0 (0.89×) | 80 ms | 95.1% | 63.6% | 977 at 32 | 37.1 GiB |
| INT4, community AWQ build, the naive pick | 1 | 47.7 (0.96×) | 34 ms | 94.8% | 62.9% | 1,025 at 32 | 37.9 GiB |
| Spec decode, 70B + 8B draft, 5 tokens | 2 | 62.2 (1.25×) | 107 ms | same as BF16 by design | | 517 at 32 | 73.2 GiB |

- **FP8** matches BF16 one request at a time and serves about 1.5× BF16's tokens per GPU under load, with no measurable accuracy loss. On Hopper it is the everyday lane.
- **INT4** runs at 89% of BF16's speed on half the GPUs and about the same output per GPU under load, so its win is the replica that needs one GPU instead of two. It hits a ceiling near 1,050 tokens/s per GPU at 64 in flight (p95 53 to 55 ms) while BF16 keeps scaling: the finding from Red Hat's quantization study, W4A16 for bandwidth-bound serving and W8A8 once batching makes it compute-bound. On MMLU-Pro it scored 3 to 4 points lower on 280 questions, which is too few to call it, so the hardest questions stay on BF16 until it's tested further. GSM8K shows no loss.
- **Spec decode** is 1.25× faster per request (1.18× at temperature 0.7; 70.5% and 63.9% of drafted tokens accepted, 4.5 and 4.2 tokens per 70B pass), with a first token about 3× slower and about half of BF16's tokens per GPU under load. It is a latency tool.
- **Recorded answers:** all five setups answered the sheep riddle with 9, put Carol's meeting on Monday (at temperature 0 and in 5 of 5 samples at 0.7), and extracted all three JSON values. Spec decode's temperature-0 answers match BF16's word for word on 3 of 8 presets and diverge at a word choice, same substance, on the other 5.

Both INT4 builds run on vLLM's Machete kernel; FP8 on CutlassFP8ScaledMM; all four on FlashAttention 3. The AWQ build's ShareGPT run at 64 in flight was a retry on cached prompts and isn't used for any claim. The KV cache left on an H200: BF16 367K tokens, spec decode 218K, FP8 182K, INT4 283K.

## Run it

```bash
make setup
make run          # replay mode, no GPUs: http://localhost:8000/presenter
```

Three scenes, keys `1` `2` `3`: **Ask** streams a question (typed, or one of eight presets) to every setup; **Under load** replays the sweep with the 50 ms target drawn on each card; **Numbers** is the money slide. `R` flips every column between the live models and the recordings, `T` is the theme, `F` full screen. With `.env` pointing at vLLM endpoints (`SIMULATION_MODE=false`), Ask is live, and a column whose request fails or stalls plays its recording under a line that says so. A setup with `MODEL_<VARIANT>_MODE=recorded` never calls its endpoint. `make test` runs the suite; `scripts/preflight.py` checks the endpoints and recordings before a talk. The booth game is at `/arena` ([ARENA.md](ARENA.md)). Talk logistics are in [CONFERENCE_GUIDE.md](CONFERENCE_GUIDE.md).

## Deploy

The vLLM deployments are KServe InferenceServices in `kubernetes/models/`, one per setup, as they ran. They use raw deployment mode (a plain Deployment behind a Service), and on our cluster GPUs are scheduled through a Kueue queue, which the files leave out. The two INT4 files reference a ServingRuntime on our cluster that isn't in the repo yet; use your cluster's vLLM runtime or the raw-container form of the other three. Every file uses `--max-model-len 131072`, the model's full context and what was measured; 32768 is plenty for a demo and leaves more room for KV cache.

Not every setup needs a full H200:

| Setup | Weights as loaded | GPUs | 71 GB MIG slice |
|---|---|---|---|
| BF16, tensor parallel 2 | 65.7 GiB per GPU | 2 full H200s | no |
| Spec decode | 73.2 GiB per GPU | 2 full H200s | no |
| FP8 | 67.7 GiB (72.7 GB) | 1 full H200 | no, the weights alone exceed it |
| INT4 | 37.1 GiB | 1 GPU | untested (all runs were on full H200s); a 71 GB slice is 66.1 GiB, which after the weights, vLLM's 10% reserve and workspace leaves about 20 GiB for KV cache, and a 3g slice has 3/7 of the SMs, so no speed number here carries over |

The demo app itself: `.env` on a laptop (see `.env.example`), or the backup on OpenShift:

```bash
podman build -t quay.io/markellr-redhat/pytorch-quantization-demo:2026.10 .
oc create secret generic pytorch-quantization-demo --from-literal=PRESENTER_KEY=$(openssl rand -hex 16)
oc apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml -f kubernetes/route.yaml
```

One replica: metrics and websocket fan-out live in process memory.

## Where the evidence lives

`benchmark_results.json` is built from the raw files by `scripts/build_benchmark_file.py` and the dashboard reads it; a test rebuilds it and checks it matches. `bench/raw/2026-09-29-r2/` holds the delivered files: `sweeps/` (also what Under load plays), `evals/`, `logs/` (startup logs, versions, GPU utilization, spec decode counters), `captures/`, `live/` (the Sep 30 live-path check against the FP8 deployment: preflight output, error log, screenshots), and `notes.txt`. `bench/raw/2026-09-29-r1/` is the afternoon run that preceded it, kept as history in the benchmark file. `quality/<VARIANT>/` holds the recordings the demo plays (7 of the 8 presets; the reworded KV cache preset is recorded on Oct 19). The raw files were scrubbed of the cluster and node name and nothing else.

## Technical questions

**So why not just use FP8?**
On Hopper, for everyday traffic, that's the answer: same speed as BF16 per request, about 1.5× the tokens per GPU under load, and nothing lost that we could measure; Red Hat's card for the build says 99.9% of BF16 on OpenLLM v1. It needs FP8 tensor cores (Ada, Hopper and newer; on A100 vLLM falls back to a weight-only FP8 kernel, which is slower, so INT4 is the one-GPU option there), and its 72.7 GB of weights won't fit where INT4's 40 GB will. That's the split on the router slide: FP8 for everyday questions, INT4 where 73 GB doesn't fit, BF16 for the hardest questions, spec decode where latency matters.

**What changed for speculative decoding?**
The first run had `enforce_eager` on, which in vLLM 0.18 turns off `torch.compile` and CUDA graphs for the target and the draft, and measured 40.0 tokens/s, slower than BF16. With it off, spec decode is 1.25× BF16 at temperature 0 and 1.18× at 0.7, where fewer drafted tokens are accepted. It uses the same GPUs as BF16 and the draft takes memory from the KV cache. Under load it serves about half of BF16's tokens per GPU: vLLM batches verification across requests, so that isn't the reason; the cost is the compute spent on rejected draft tokens (about 30% at temperature 0) plus the draft's own passes, which stops paying off once the target is compute-bound. The startup log also notes async scheduling isn't supported with a draft-model speculator.

**Why isn't INT4 faster than BF16 per request?**
Decoding one request is limited by reading the weights: BF16 reads about 70 GB per GPU per token against 4.8 TB/s, INT4 about 40 GB, so by bandwidth INT4 should be about 1.75× faster. It ran at 89% (the AWQ build at 96%), about 36 to 40% of an H200's peak bandwidth against about 73% for BF16, and we haven't profiled the gap. Red Hat's study saw the same on H100: INT4 on one GPU only slightly faster than BF16 on two at one request (Kurtic et al., Table 5). Under load INT4 catches up per GPU until its ceiling.

**Is speculative decoding really lossless?**
Verification keeps the target's output distribution; vLLM calls it lossless up to hardware numerics. Comparing the temperature-0 recordings, 3 of the 8 presets are identical word for word (sheep riddle, decline a meeting, JSON extraction) and 5 diverge at a word choice with the same substance (positions estimated from characters): the Python function at about token 45, the summary at 103, the quick fact at 36, the KV cache explanation at 211, the logic puzzle at its third token ("To find out" versus "To determine"), both ending on Monday. Spec decode isn't guaranteed byte-identical to plain decoding, because the two paths batch and round differently.

**AWQ or GPTQ, and which checkpoints?**
Both give INT4 weights that vLLM serves with mixed-precision kernels, and Red Hat's study found GPTQ slightly ahead on harder benchmarks. The INT4 on screen is [RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16), Red Hat's validated W4A16 build: a GPTQ checkpoint in the AutoGPTQ format (`quant_method: gptq`, 4-bit weights, group size 128, 16-bit activations), loaded through `gptq_marlin` and run on Machete. The naive pick was `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`, an AutoAWQ checkpoint with group size 128, loaded through `awq_marlin` and also run on Machete. The FP8 build ([RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8)) is an LLM Compressor checkpoint in the `compressed-tensors` format.

**Does this hold for other models, or an 8B?**
The recipe is model-agnostic: the manifests, the tests and the dashboard take any Hugging Face model id, and Red Hat's study ([Kurtic et al., "Give Me BF16 or Give Me Death?"](https://arxiv.org/abs/2411.02355), ACL 2025) ran more than 500,000 evaluations across the 8B, 70B and 405B sizes. Smaller models lose more when quantized: Red Hat's W4A16 model cards report 97.4% of BF16 on OpenLLM v2 for the 70B and 96.1% for the 8B, so test an 8B on your own prompts. A second model wasn't measured for this talk.

**Can I route per request, and with what?**
The router on screen is an example rule, not a gateway. To build one, [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) classifies each request and picks the model, and [llm-d Router](https://github.com/llm-d/llm-d-router) (formerly the llm-d inference scheduler) picks the replica within a pool, preferring one with the prompt's prefix cached. INT4 plus spec decode, a draft on an INT4 target, is the obvious next lane and untested here.

**Where's PyTorch in all this?**
vLLM is a PyTorch Foundation project and compiles its models with torch.compile, LLM Compressor calibrates in PyTorch, and the arena's networks are trained and quantized in PyTorch.

## Credits and license

MIT. Built with Llama: the models served here are Llama 3.1 70B Instruct and Llama 3.1 8B Instruct, licensed under the [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE), Copyright © Meta Platforms, Inc. All Rights Reserved; the recorded answers in `quality/` are Llama 3.1 outputs. The arena's physics and game loop are adapted from [FlappyLearning](https://github.com/xviniette/FlappyLearning) by Vincent Bazia (MIT), and the Red Hat fonts are bundled under the SIL Open Font License. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Markell Rawls**, AI Developer Advocate, Red Hat
