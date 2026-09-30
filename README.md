# Not Every Question Needs Two GPUs

Everyone who has shipped a 70B model has had the same conversation with their GPU bill, and the first answer is usually to quantize it. This demo puts that decision side by side. It sends the same question to Llama 3.1 70B Instruct served by vLLM on NVIDIA H200s in four setups (BF16 on two GPUs, FP8 on one, INT4 from Red Hat's LLM Compressor on one, and speculative decoding with an 8B draft model on two), streams all the answers at once, and then shows what each setup gets you, what it costs, and where a router fits.

Built for the Demo Theater at PyTorch Conference North America 2026 in San Jose.

## Run it on your laptop

You don't need a GPU or a cluster. Without model endpoints the dashboard runs in replay mode, which plays back the speeds measured on the H200s, and everything (fonts included) is bundled, so it works with no network at all.

```bash
git clone https://github.com/MarkellR-RedHat/pytorch-quantization-demo.git
cd pytorch-quantization-demo
./scripts/setup.sh
./scripts/run-local.sh
```

Open http://localhost:8000/presenter. You need Python 3.11 or newer.

## What's on the screen

The presenter dashboard is a fixed 1920 by 1080 stage that scales to whatever it's plugged into, so nothing reflows or clips on a 720p projector.

**Ask** (press `1`). Type a question, or pick one of the presets, and it goes to every setup at once. The answers stream side by side with the time to first token, tokens per second, how many tokens each answer used, total time, and how many GPUs each setup uses. Answer length matters, because a setup that writes shorter answers can finish sooner while reasoning less. A line above the answers shows where a simple example router would send that question and why. With live endpoints connected these are real answers from the real deployments, and the first-token time is labeled "TTFT + network", because it crosses the VPN or a port-forward and isn't comparable to the benchmark's. Without them, a small "Replay" badge shows in the corner and each column plays back the speed that setup measured, without calling a model and without inventing a difference in the answers.

The eight preset questions, and the lane the example router gives each one (it matches the router slide):

| Button | Question | Router lane |
|---|---|---|
| Sheep riddle | A farmer has 17 sheep. All but 9 run away. How many are left? (step by step) | BF16, the hardest questions |
| Logic puzzle | Alice, Bob and Carol each have one meeting on a different day of Monday to Wednesday… Which day is Carol's? | BF16 |
| Three bullet summary | The trade-offs of quantization in 3 bullets | The everyday lane: FP8 when it's deployed, INT4 otherwise |
| Decline a meeting | A short, polite reply declining a Friday meeting | Everyday |
| Quick fact | The capital of Australia, and why it isn't Sydney, in two sentences | Everyday |
| Extract to JSON | Name, company and date from a one-line message | Everyday |
| Python function | Second largest number in a list, with edge cases | Spec Decode, long answers with someone waiting |
| Explain KV cache | The KV cache in a transformer LLM, for a new engineer, in about 300 words | Spec Decode (the long-answer, latency case) |

The exact wording lives in `app/quality.py` (`PROMPTS`), and `scripts/capture_presets.py` records every setup's answer to each one at temperature 0 with the same 1,024-token cap as live Ask.

**If a live request fails.** When a setup errors, sends no first token within 8 seconds, or goes quiet for 10 seconds mid-answer, that column plays its recorded answer to the preset instead. Any live text it already streamed stays above a red dashed line, the line says "Recorded <date> · live request failed after N tokens", and the column is badged RECORDED, so a recording is never passed off as live. A typed question has no recording, so the column says the live request failed. Press `R` to switch every column between the live models and replay in one keystroke.

**Under load** (press `2`, then `Space`). A replay of the `vllm bench serve` load test: 1, 8, 32, then 64 requests in flight at once against each setup (synthetic random-token prompts), with output tokens per second per GPU, total output tokens per second, the tail time per output token (TPOT, p95 when the run recorded it, otherwise vLLM's default p99), and median time per answer at each step. What sets cost is how many tokens a GPU serves while users still get a fast stream, so each card marks a latency target on its TPOT chart (`TPOT_TARGET_MS`, default 50 ms, which is 20 tokens per second per user) and ends with its best output tokens per GPU under that target. The result line compares INT4 and Spec Decode with BF16 on that number, and the Numbers screen's cost line uses it too. It plays back whatever sweep files are in `bench/<VARIANT>/c<N>.json` (the `vllm bench serve --save-result` output), and until those exist the scene stays out of the numbered flow, so `1` and `2` go to Ask and Numbers.

**Numbers** (press `3`, or `2` until the load test exists). The money slide: GPUs, tokens per second and mean time for one request, and accuracy for each setup (GSM8K on the card, MMLU-Pro with its sample size in the footnote), what each one is best for, what to watch out for, and when a router is worth adding.

**The arena** (http://localhost:8000/arena). A booth game where birds flown by the same small PyTorch network, stored at different precisions, fly a course with hard gaps. It isn't part of the talk, and the next section explains how it works.

## How the arena works, and how to check it

The arena is a booth demo, and I wanted it to hold up if someone who works on quantization kernels looks at it for more than ten seconds, so nothing in it is scripted.

| Bird | Weights | Hard gaps cleared* |
|---|---|---|
| BF16 | 16-bit brain float | 219 of 219 |
| FP8 | E4M3 with one scale per output row | 219 of 219 |
| INT4 AWQ | 4-bit, group size 16, activation-aware scaling | 219 of 219 |
| INT4 RTN | 4-bit, group size 16, plain round to nearest | 135 of 220 |
| Spec Decode | BF16 network plus a 4-unit draft network | same path as BF16, to the pixel |

\* Measured by `arena/train_policy.py` on 20 held-out courses. Every variant also clears every easy gap.

The birds are flown by a 6-64-64-1 MLP trained in PyTorch by imitation learning (DAgger) from a simple rule-based pilot. The four precision variants are the same trained weights quantized after training: BF16 and FP8 use PyTorch's own `torch.bfloat16` and `torch.float8_e4m3fn` casts, INT4 RTN is asymmetric round-to-nearest with one scale and zero point per group of 16 inputs, and INT4 AWQ scales each input channel by its mean activation raised to a power (searched per layer to minimize output error) before doing the same group rounding, which is the core idea of [AWQ](https://arxiv.org/abs/2306.00978). Biases stay in higher precision, as they do in W4A16 serving formats.

One input feature is fed in raw pixels while the other five are normalized, and that's deliberate. It plays the part of the outlier activation channels LLMs develop ([LLM.int8()](https://arxiv.org/abs/2208.07339), [SmoothQuant](https://arxiv.org/abs/2211.10438)), which is the reason plain INT4 rounding hurts and activation-aware scaling fixes it. If you normalize every feature, all of the INT4 variants fly perfectly, which is the same lesson from the other side.

The speculative decoding bird runs the BF16 network as the target. A 4-unit draft network guesses the next 4 moves along its own imagined path, the target scores all of those states at once, and the bird keeps the moves the target agrees with plus the target's own move at the first disagreement (or a bonus move when all four were right). That's greedy speculative decoding as described by [Leviathan et al.](https://arxiv.org/abs/2211.17192), so the committed moves are exactly the ones the target would have chosen alone. The sidebar shows the live acceptance rate, the moves per target pass, and the distance between the spec bird and a hidden BF16 bird flying the same spot, which stays at 0 px.

To check it yourself:

```bash
pip install -r arena/requirements.txt
python arena/train_policy.py      # retrains and re-exports static/arena/policy.json
pytest tests/test_arena.py        # browser engine vs PyTorch, frame for frame
```

On the same PyTorch version the retrain is byte-identical to the committed file. The tests run the JavaScript engine in Node and check that it makes the same decision as Python on every frame with zero drift, that the spec bird never leaves the BF16 path, that the INT4 weights really have at most 16 levels per group, and that the numbers quoted on screen match the exported evaluation.

The arena is a teaching model, and I'd say that out loud on stage. A 4,673-parameter policy isn't a 70B language model, and the toy draft agrees with its target about 93% of the time, which is higher than an 8B draft gets on a 70B target. What carries over is the mechanism.

## The real numbers

Measured on September 29, 2026 on NVIDIA H200 (141 GB) with vLLM `0.18.0+rhaiv.14` (the vLLM build that ships with Red Hat AI), PyTorch 2.10.0 and driver 580.126.20, with `vllm bench serve` running inside each pod, so no network hop is included. All setups ran with `enforce_eager` off, so torch.compile and CUDA graphs were on, and the startup logs show FlashAttention 3 for all of them. Single stream means 30 ShareGPT prompts, one at a time, at temperature 0.

| Setup | GPUs | Tokens/s, one request | vs BF16 | Mean time per request | Mean time to first token | Weights per GPU |
|---|---|---|---|---|---|---|
| BF16 (tensor parallel 2) | 2 | 49.7 | 1.00× | 3.90 s | 33 ms | 65.7 GiB |
| INT4, Red Hat's LLM Compressor build (Machete kernel) | 1 | 44.0 | 0.89× | 4.48 s | 80 ms (median 47) | 37.1 GiB |
| INT4, community AWQ build, the naive pick (Machete kernel) | 1 | 47.7 | 0.96× | 4.24 s | 34 ms | 37.9 GiB |
| Spec decode (70B + 8B draft, 5 tokens) | 2 | 62.2 | 1.25× | 3.22 s | 107 ms | 73.2 GiB |
| FP8, Red Hat's build (CutlassFP8ScaledMM kernel) | 1 | 51.1 | 1.03× | 3.77 s | 62 ms (median 41) | 67.7 GiB |

At temperature 0.7 the same runs give 49.3, 44.4 (0.90×), 47.2 (0.96×), 58.3 (1.18×) and 51.0 (1.03×) tokens/s. FP8 ran last, into the early hours of Sep 30.

- **INT4** runs at 89% of BF16's single-request speed on half the GPUs. The community AWQ build is a little faster per request (96%), and both run on vLLM's Machete kernel on Hopper; the Red Hat build is a GPTQ checkpoint in the `compressed-tensors` format, loaded through `gptq_marlin`.
- **Spec decode** runs about 1.25× faster than BF16 on the same two GPUs (1.18× at temperature 0.7). Its first token is about 3× slower than BF16's, because the draft model runs first. The draft's tokens were accepted 70.5% of the time at temperature 0 and 63.9% at 0.7, which works out to 4.5 and 4.2 tokens per 70B forward pass. Those rates are isolated per temperature: the counters were read before and after each run.
- **FP8** matches BF16 one request at a time (1.03×) on one GPU: 8-bit weights and activations in the `compressed-tensors` format ([RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8)), run on Hopper's native FP8 tensor cores through vLLM's CutlassFP8ScaledMM kernel. The weights take 67.7 GiB, so it fits on one H200 with 182K tokens of KV cache (INT4 leaves room for 283K). Its c=1 random-prompt sweep point looks cold (40.8 tokens/s, mean first token 144 ms), so the headline is the ShareGPT single-stream run.
- **Accuracy, GSM8K** (8-shot chain of thought, all 1,319 questions, `lm_eval`): BF16 94.8%, FP8 95.2%, INT4 (Red Hat) 95.1%, INT4 (AWQ) 94.8%. No loss; the differences are inside the ±0.6 point standard error.
- **Accuracy, MMLU-Pro** (5-shot, the first 20 questions of each of 14 subjects, 280 in all, standard error about ±2.8 points): BF16 66.8%, FP8 66.1%, INT4 (Red Hat) 63.6%, INT4 (AWQ) 62.9%. FP8 is inside a point of BF16. On the hardest questions INT4 scored 3 to 4 points lower on 280 questions, which is too few to call it, so the hard questions stay on BF16 until it's tested further. Spec decode wasn't evaluated: its output is the 70B's by design.
- **Recorded answers:** all five setups answered the sheep riddle with 9, put Carol's meeting on Monday (at temperature 0 and in 5 of 5 samples at 0.7), and extracted all three JSON values.

### Under load

The sweep is `vllm bench serve` at 1, 8, 16, 32 and 64 requests in flight, with random-token prompts of 512 tokens asking for 256 (`bench/<VARIANT>/c<N>.json`) and with ShareGPT prompts (`sharegpt-c<N>.json`). Every run finished with zero failed requests. What sets cost is how many output tokens a GPU serves while each user still gets a fast stream, so the number that matters is output tokens per second per GPU with the p95 time per output token at or under 50 ms (20 tokens per second per user).

| Setup | Random prompts, best under 50 ms | ShareGPT prompts, best under 50 ms |
|---|---|---|
| BF16 (2 GPUs) | 981 per GPU at 64 in flight | 867 at 64 |
| INT4, Red Hat build (1 GPU) | 977 at 32 | 916 at 32 |
| INT4, AWQ build (1 GPU) | 1,025 at 32 | see the note below |
| Spec decode (2 GPUs) | 517 at 32 | 561 at 32 |
| FP8 (1 GPU) | 1,485 at 64 (p95 38.8 ms) | 860 at 32 (64 in flight is 55 ms, just over) |

- At the 50 ms target, INT4 serves about the same output per GPU as BF16 on random prompts and a little more on chat-like prompts. **INT4's win is the replica that needs one GPU instead of two; per GPU at saturation it is roughly even.**
- INT4 hits a ceiling near 1,050 tokens/s per GPU at 64 in flight on random prompts (1,043 for the Red Hat build and 1,056 for AWQ, at a p95 of 53 to 55 ms), while BF16 keeps scaling (981 per GPU at 64, p95 36 ms). That is the finding from Red Hat's quantization study showing up: 4-bit weights with 16-bit activations (W4A16) win when a GPU is starved for memory bandwidth, one request at a time, and lose that edge under heavy batching, where the work becomes compute-bound and 8-bit weights and activations (W8A8) are the more cost-efficient choice.
- Spec decode serves about half of BF16's tokens per GPU under load (517 vs 981 at the target). It is a latency tool, and the sweep proves it.
- FP8 serves about 1.5× BF16's tokens per GPU under load on random prompts (1,485 at 64 in flight, still under the target at a p95 of 38.8 ms), and matches BF16 one request at a time. On Hopper, that makes it the everyday lane; INT4 is the lane for when 73 GB of weights won't fit.
- `nvidia-smi` showed every GPU at 100% utilization through the 64-in-flight runs (`logs/*-gpu-util-c64.csv`, all five setups), which says the GPUs were busy, not how well the busy time was used, so those files are evidence and not a metric.
- The AWQ build's ShareGPT run at 64 in flight (1,636 per GPU, p95 33 ms) was a retry after a timed-out first attempt on the same prompts with prefix caching on, and its first-token times are far below every other setup's at that load, so that point isn't used for any claim.
- 50 ms is a reasonable target for this data: BF16 and both INT4 builds sit between 36 ms (32 in flight) and 53 to 55 ms (64 in flight) on random prompts, so the target separates "still fast" from "starting to queue". A 60 ms target would let INT4's 64-in-flight point count (1,043 against BF16's 981) without changing the story: about even per GPU, on half the GPUs per replica.

### Where the numbers live

Every number above is built from the raw files in `bench/raw/2026-09-29-round2/` by `scripts/build_benchmark_file.py`, which writes `benchmark_results.json`, the file the dashboard reads. The sweeps the Under load scene plays are the same files under `bench/<VARIANT>/`, the `lm_eval` output is under `eval/`, the startup logs, versions and spec decode counters are under `logs/`, and the recorded answers (all 8 presets for all four setups, with their own timings, recorded through a port-forward with the same 1,024-token cap as live Ask) are in `quality/`. The deployments that produced them are in `kubernetes/models/`. The live path (the Ask scene streaming from a real vLLM endpoint, with the other columns falling back to their recordings) was verified on Sep 30 against the FP8 deployment; the preflight output, the browser's error log and the screenshots are under `bench/raw/2026-09-29-round2/live/`.

The afternoon run of the same day (5 requests per setup on one prompt, through a port-forward) is kept under `history` in the benchmark file and under `bench/raw/2026-09-29/`. It measured spec decode at 64.9 tokens/s, about 1.4× BF16, on that one prompt with one cold run in the average; the 30-prompt run above gives 1.25×, which is the number used everywhere. An even earlier spec decode run with `enforce_eager` on measured 40.0 tokens/s and is kept there too.

## Technical questions

**So why not just use FP8?**
On Hopper, for everyday traffic, that's the answer: FP8 matched BF16 one request at a time (51.1 vs 49.7 tokens/s), served about 1.5× BF16's tokens per GPU under load, and lost nothing we could measure (GSM8K 95.2% vs 94.8%, MMLU-Pro 66.1% vs 66.8% on 280 questions). Red Hat's published numbers for the same build say 99.9% of BF16 on OpenLLM v1. It needs a GPU with native FP8 (Hopper or newer), and its weights take 67.7 GiB, so it leaves less KV cache than INT4 (182K vs 283K tokens on an H200) and it won't fit where INT4's 37 GiB will. That's the split on the router slide: FP8 for everyday questions, INT4 where 73 GB doesn't fit, BF16 for the hardest questions, spec decode where latency matters.

**What changed for speculative decoding?**
The first spec decode run had `enforce_eager` on, which in vLLM 0.18 turns off both `torch.compile` and CUDA graphs for the 70B target and the 8B draft. It measured 40.0 tokens/s, slower than BF16. With `enforce_eager` off it measures 62.2 tokens/s on 30 ShareGPT prompts at temperature 0, about 1.25× BF16 on the same two GPUs, and 58.3 (1.18×) at temperature 0.7, where the draft's guesses get accepted less often (63.9% against 70.5%). Spec decode uses the same GPUs as BF16: it buys lower latency per request, not fewer GPUs, and the draft model takes memory away from the KV cache (218K tokens of cache vs 367K for BF16). Two more things to know before choosing it: the first token takes about 3× longer than BF16's (107 ms against 33 ms mean, inside the pod), and under load it serves about half of BF16's output tokens per GPU, because each 70B forward pass verifies one request's draft instead of decoding a token for many requests at once.

**Does this hold for an 8B model?**
The trade-offs have the same shape, but smaller models lose more when quantized. In Red Hat's quantization study, INT4 kept 97.4% of the 70B's score on the OpenLLM v2 benchmarks and 96.1% of the 8B's, so test an 8B carefully on your own prompts.

**AWQ or GPTQ?**
Both produce INT4 weights that vLLM serves with fast mixed-precision kernels. Red Hat's study found GPTQ slightly ahead on harder benchmarks, and Red Hat's published INT4 build of this model ([RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16)) is GPTQ.

**Where do these numbers come from?**
Red Hat's quantization study is Kurtic et al., ["Give Me BF16 or Give Me Death? Accuracy-Performance Trade-Offs in LLM Quantization"](https://arxiv.org/abs/2411.02355), ACL 2025, which ran more than 500,000 evaluations. The benchmark numbers in this repo come from `vllm bench serve` and `lm_eval` on NVIDIA H200s with vLLM 0.18 (see The real numbers above): 30 prompts one at a time for the headline speeds, a concurrency sweep for the load numbers, and 1,319 GSM8K plus 280 MMLU-Pro questions for accuracy.

**Why isn't INT4 faster than BF16?**
At one request at a time, decoding is limited by how fast the GPU can read the weights. BF16 across two H200s reads about 70 GB per GPU per token against 4.8 TB/s each, and INT4 on one H200 reads about 40 GB, so on paper INT4 has more headroom: by bandwidth alone it should be about 1.75× faster. In practice the Red Hat build ran at 89% of BF16's speed and the AWQ build at 96%, which works out to about 36% and 40% of one H200's peak bandwidth against about 73% per GPU for BF16. We haven't profiled where the gap goes. Red Hat's quantization study saw the same pattern on H100: at one request at a time, INT4 on one GPU was only slightly faster than BF16 on two (Kurtic et al., Table 5). The result that matters for the bill is that INT4 does this on half the GPUs, and under load it serves about the same output per GPU as BF16 up to its ceiling.

**Is speculative decoding really lossless?**
The verification step keeps the output distribution of the target model, so the quality you get is the target's quality. vLLM describes it as lossless up to the precision limits of hardware numerics. We compared spec decode's temperature-0 answers with BF16's on all eight preset questions (`quality/SPEC_DECODE/` against `quality/FP16/`). Three are identical word for word: the sheep riddle, the meeting decline, and the JSON extraction. The other five start the same and then diverge at a word choice, after which the wording differs while the substance doesn't (positions are estimated from characters): the Python function at about token 45 ("an empty list, a list with a single element" versus "empty lists, lists with a single element"), the summary at about token 103, the quick fact at about token 36, the KV cache explanation at about token 211 (the recording of the earlier wording of that prompt), and the logic puzzle at its third token ("To find out" versus "To determine"), with the same answer, Monday. Both answers in every pair come from the 70B; spec decode just isn't guaranteed to be byte-identical to plain decoding, because the two paths batch and round differently.

**Which INT4 checkpoints were these?**
The INT4 on screen is [RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16), a GPTQ checkpoint made with LLM Compressor in the `compressed-tensors` format: 4-bit weights, 16-bit activations. vLLM loads it through `gptq_marlin` and, on Hopper GPUs, runs it with the Machete kernel (the startup log says `Using MacheteLinearKernel for GPTQMarlinLinearMethod`). The naive pick was `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`, an AutoAWQ checkpoint with group size 128 and FP16 activations, loaded through `awq_marlin` and also run on Machete. LLM Compressor supports AWQ too, through its `AWQModifier`.

**Can I route per request, and with what?**
The router on screen is an example rule written for the demo, not a deployed gateway. To build one, [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) classifies each request and picks the model, and [llm-d Router](https://github.com/llm-d/llm-d-router) (formerly the llm-d inference scheduler) picks the replica within each model's pool, preferring one that already has the prompt's prefix cached. INT4 plus spec decode, a draft model on an INT4 target, is the obvious combination to add as another lane, and we haven't tested it here.

**Where's PyTorch in all this?**
vLLM is a PyTorch Foundation project and compiles its models with torch.compile, LLM Compressor calibrates in PyTorch, and the arena's networks are trained and quantized in PyTorch.

## Running with real models

Copy `.env.example` to `.env`, set `SIMULATION_MODE=false`, and point the endpoints at your vLLM deployments:

| Variable | What it is |
|---|---|
| `MODEL_FP16_ENDPOINT` | BF16 deployment, `/v1/chat/completions` URL |
| `MODEL_INT4_ENDPOINT` | INT4 deployment |
| `MODEL_SPEC_DECODE_ENDPOINT` | Speculative decoding deployment |
| `MODEL_FP8_ENDPOINT` | Optional FP8 deployment, adds a fourth column |
| `MODEL_<VARIANT>_NAME` | The `--served-model-name` for each deployment |
| `MODEL_<VARIANT>_MODE` | `live` (default) or `recorded`, a backup switch that plays that setup's preset recordings instead of calling it |
| `MODEL_INT4_CAPTURES` | Which recordings the INT4 column uses: `INT4` (community AWQ build) or `INT4_RH` (Red Hat's LLM Compressor build) |
| `PRESENTER_KEY` | Protects the Ask box and the controls. Open `/presenter?key=<value>` once on the presenter laptop |
| `MAX_INFLIGHT_PER_VARIANT` | Caps concurrent requests per deployment (default 32) |
| `GPU_HOURLY_USD` | Shows cost per million output tokens at the latency target when set, labeled as an assumption |
| `TPOT_TARGET_MS` | The tail time per output token a setup must stay under for its load-test throughput to count (default 50) |

The speculative decoding deployment that was benchmarked (the args from `kubernetes/models/isvc-spec-decode.yaml`):

```bash
vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --tensor-parallel-size 2 \
  --dtype bfloat16 --max-model-len 131072 \
  --speculative-config '{"model": "meta-llama/Llama-3.1-8B-Instruct", "num_speculative_tokens": 5}'
```

If you don't have H200s, the same comparison should work with Llama 3.1 8B as the target and Llama 3.2 1B as the draft on a single 24 GB GPU. We haven't tested that setup, and it needs a shorter context to fit, for example `--max-model-len 8192`. The absolute numbers change, and the comparison is still apples to apples.

## Deploying to OpenShift

For the talk the app runs on the presenter laptop with `.env` pointing at the vLLM endpoints (see Running with real models), and this deployment is the backup:

```bash
podman build -t quay.io/<your-org>/pytorch-quantization-demo:latest .
podman push quay.io/<your-org>/pytorch-quantization-demo:latest
oc create secret generic pytorch-quantization-demo --from-literal=PRESENTER_KEY=$(openssl rand -hex 16)
oc apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml -f kubernetes/route.yaml
```

`kubernetes/configmap.yaml` holds the same variables as `.env`; edit the endpoints and served names there first. The vLLM deployments themselves are in `kubernetes/models/` as KServe InferenceServices, one per setup, written for our OpenShift AI cluster (raw deployment mode, GPUs scheduled through Kueue; each file says what to drop elsewhere).

### GPU sizing

Not every setup needs a full H200 either. From the vLLM startup logs ("Model loading took"), the weights as loaded:

| Setup | Weights | GPUs | Fits a 71 GB MIG slice? |
|---|---|---|---|
| BF16, tensor parallel 2 | 65.7 GiB per GPU | 2 full H200s | No |
| Spec decode, tensor parallel 2 | 73.2 GiB per GPU (target plus draft) | 2 full H200s | No |
| FP8 | 67.7 GiB (72.7 GB) | 1 full H200 | No: the weights alone are more than the slice, before any KV cache |
| INT4, Red Hat's build | 37.1 GiB | 1 GPU | Yes, with about 30 GB left for KV cache |
| INT4, AWQ build | 37.9 GiB | 1 GPU | Yes |

The measurements here ran on full H200s (INT4 had 283K tokens of KV cache on one), and the manifests request whole GPUs so the startup logs match. On a MIG-partitioned cluster the INT4 files would request `nvidia.com/mig-3g.71gb` instead and leave the full GPUs to the two-GPU setups. Every file uses `--max-model-len 131072`, the model's full context and what was measured; 32768 is plenty for a demo and leaves more room for KV cache.

The deployment runs one replica on purpose. Metrics and websocket connections live in memory, so a second replica would split the presenter screen's metrics across two pods.

## Keyboard shortcuts

| Key | What it does |
|---|---|
| `1` `2` `3` | Ask, Under load, Numbers |
| `Space` | Play the load run (on Under load) |
| `/` | Jump to the question box |
| `Enter` | Send the question to every setup |
| `R` | Switch every column between live models and replay |
| `T` | Light or dark theme |
| `F` | Full screen |

To force replay mode while presenting, press `R` or open `/presenter?mode=sim`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Project layout

```
app/                 FastAPI backend (Ask streaming, replay, live vLLM client, benchmark loader, metrics)
arena/               PyTorch training and quantization for the arena policies
static/arena/        Exported policy weights the browser runs
static/js/           Presenter dashboard, arena engine
templates/           Presenter page and the /arena booth page ("/" redirects to /presenter)
benchmark_results.json, bench/, quality/   Measured data the dashboard reads
scripts/             Setup, local run, and benchmark scripts
kubernetes/          OpenShift manifests
tests/               API, Ask, simulation, metrics, quality, and arena tests
```

## Credits and license

MIT. Built with Llama: the models served in this demo are Llama 3.1 70B Instruct and Llama 3.1 8B Instruct, and Llama 3.1 is licensed under the [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE), Copyright © Meta Platforms, Inc. All Rights Reserved. The recorded answers in `quality/` are Llama 3.1 outputs. The arena's physics and game loop are adapted from [FlappyLearning](https://github.com/xviniette/FlappyLearning) by Vincent Bazia (MIT), and the Red Hat fonts are bundled under the SIL Open Font License. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Markell Rawls**, AI Developer Advocate, Red Hat
