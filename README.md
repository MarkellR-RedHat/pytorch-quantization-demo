# Not Every Question Needs Two GPUs

Everyone who has shipped a 70B model has had the same conversation with their GPU bill, and the first answer is usually to quantize it. This demo puts that decision side by side. It sends the same question to Llama 3.1 70B Instruct served by vLLM on NVIDIA H200s in three setups (BF16 on two GPUs, INT4 AWQ on one, and speculative decoding with an 8B draft model on two), streams all three answers at once, and then shows what each setup gets you, what it costs, and where a router fits.

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

**Ask** (press `1`). Type a question, or pick one of the presets, and it goes to every setup at once. The answers stream side by side with the time to first token, tokens per second, how many tokens each answer used, total time, and how many GPUs each setup uses. Answer length matters, because a setup that writes shorter answers can look faster per token while reasoning less. A line above the answers shows where a simple example router would send that question and why. With live endpoints connected these are real answers from the real deployments. Without them, a small "Replay" badge shows in the corner and each column plays back the speed that setup measured, without calling a model and without inventing a difference in the answers.

**Under load** (press `2`, then `Space`). A replay of the `vllm bench serve` load test: 1, 8, 32, then 64 requests in flight at once against each setup (synthetic random-token prompts), with output tokens per second per GPU, total output tokens per second, and median time per answer at each step. The result line compares INT4 with BF16 at the same load per GPU, and Spec Decode with BF16 on the same GPUs. It plays back whatever sweep files are in `bench/<VARIANT>/c<N>.json` (the `vllm bench serve --save-result` output), and until those exist the scene stays out of the numbered flow, so `1` and `2` go to Ask and Numbers.

**Numbers** (press `3`, or `2` until the load test exists). The money slide: GPUs, tokens per second and mean time for one request, and accuracy against BF16 for each setup (shown as pending where it hasn't been measured), what each one is best for, what to watch out for, and when a router is worth adding.

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

Measured on September 29, 2026 on NVIDIA H200 (141 GB) with vLLM `0.18.0+rhaiv.14`, the vLLM build that ships with Red Hat AI. All three setups ran with `enforce_eager` off, so torch.compile and CUDA graphs were on, and FlashAttention 3 was the attention backend. Each setup got 5 requests one at a time, on the same benchmark prompt, at temperature 0, with 256 output tokens each.

| Setup | GPUs | Tokens/s, one request | Mean time per request | Time to first token | Weights per GPU |
|---|---|---|---|---|---|
| BF16 (tensor parallel 2) | 2 | 46.9 | 5.46 s | 315 ms | 65.7 GiB |
| INT4 AWQ (Machete kernel) | 1 | 45.3 | 5.65 s | 339 ms | 37.9 GiB |
| Spec decode (70B + 8B draft, 5 tokens) | 2 | 64.9 | 4.01 s | not measured | 73.2 GiB |

- **INT4** runs at 97% of BF16's speed on half the GPUs. From 1 to 8 requests at once, it stayed within a few percent of BF16 per request (34.5 vs 35.1 tokens/s at 8).
- **Spec decode** runs about 1.4× faster than BF16 on the same GPUs. That average includes one cold first run at 49.6 tokens/s; the four warm runs averaged 68.8. Across our runs the draft's tokens were accepted 70% of the time, which works out to 4.5 tokens per 70B forward pass.
- **Quality:** all three answered the sheep riddle correctly at temperature 0, and 20 out of 20 times each at temperature 0.7. INT4 hasn't been run on a benchmark suite yet. In 5 open-ended prompts, INT4's answers weren't shorter than BF16's (764 vs 677 tokens on average).

Every number above is built from the raw files in `bench/raw/2026-09-29/` by `scripts/build_benchmark_file.py`, which writes `benchmark_results.json`, the file the dashboard reads. The deployments that produced them are in `kubernetes/models/`, and the captured answers are in `quality/`. Two caveats:
- The requests went through a port-forward from a laptop, so absolute latencies include that network hop.
- The weights and KV cache figures come from the vLLM startup logs.

An earlier run had spec decode with `enforce_eager` on and temperature 0.7, and it measured 40.0 tokens/s. It's kept under `history` in the benchmark file. Two things changed between that run and this one, so the jump from 40 to 65 isn't all down to the flag.

These are single-stream numbers, so they tell you about latency and not about how much traffic a GPU can serve. The load test (`vllm bench serve` at increasing concurrency) drops into `bench/<VARIANT>/c<N>.json`, and the Under load scene appears once those files exist.

## Technical questions

**Why not FP8?**
Llama 3.1 70B in FP8 is about 71 GB, so it fits on one H200 with room left for KV cache, and Hopper GPUs have native FP8 tensor cores. Red Hat's FP8 build of this model ([RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8)) keeps 99.9% of the BF16 score on the OpenLLM v1 benchmarks, so it's the next variant to add.

**What changed for speculative decoding?**
The first spec decode run had `enforce_eager` on, which in vLLM 0.18 turns off both `torch.compile` and CUDA graphs for the 70B target and the 8B draft. It measured 40.0 tokens/s, slower than BF16. With `enforce_eager` off, it measured 64.9 tokens/s, about 1.4× BF16 on the same two GPUs. That rerun was also at temperature 0 instead of 0.7, where the draft's guesses get accepted more often, so both changes contributed. The draft's tokens were accepted 70% of the time, with acceptance falling from 86% for the first guessed token to 56% for the fifth. Spec decode uses the same GPUs as BF16: it buys lower latency per request, not fewer GPUs, and the draft model takes memory away from the KV cache (218K tokens of cache vs 367K for BF16).

**Does this hold for an 8B model?**
The trade-offs have the same shape, but smaller models lose more when quantized. In Red Hat's quantization study, INT4 kept 97.4% of the 70B's score on the OpenLLM v2 benchmarks and 96.1% of the 8B's, so test an 8B carefully on your own prompts.

**AWQ or GPTQ?**
Both produce INT4 weights that vLLM serves with fast mixed-precision kernels. Red Hat's study found GPTQ slightly ahead on harder benchmarks, and Red Hat's published INT4 build of this model ([RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16)) is GPTQ.

**Where do these numbers come from?**
Red Hat's quantization study is Kurtic et al., ["Give Me BF16 or Give Me Death? Accuracy-Performance Trade-Offs in LLM Quantization"](https://arxiv.org/abs/2411.02355), ACL 2025, which ran more than 500,000 evaluations. The benchmark numbers in this repo come from single requests on NVIDIA H200s with vLLM 0.18 (see The real numbers above), so they measure one request at a time and not throughput under heavy load.

**Why isn't INT4 faster than BF16?**
At one request at a time, decoding is limited by how fast the GPU can read the weights. BF16 across two H200s reads about 70 GB per GPU per token against 4.8 TB/s each, and INT4 on one H200 reads about 35 to 40 GB, so on paper INT4 has more headroom. In practice it ran at 97% of BF16's speed, which is typical for 4-bit kernels at batch size 1. The result that matters for the bill is that it does that on half the GPUs.

**Is speculative decoding really lossless?**
The verification step keeps the output distribution of the target model, so the quality you get is the target's quality. vLLM describes it as lossless up to the precision limits of hardware numerics. We saw exactly that at temperature 0. On the sheep riddle, spec decode's answer matched BF16's word for word. On the code and summary prompts, the two matched for the first 100 to 150 tokens and then split on a near-tie word choice ("applications where low latency is critical" versus "latency-critical applications"). Both answers are the 70B's; they just aren't guaranteed to be byte-identical.

**Which INT4 checkpoint was this?**
`hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`, an AutoAWQ checkpoint: 4-bit weights, group size 128, FP16 activations. vLLM loads it through its `awq_marlin` path and, on Hopper GPUs, runs it with the Machete kernel. LLM Compressor writes the `compressed-tensors` format and supports AWQ through its `AWQModifier`, and Red Hat's published INT4 build of this model is a GPTQ checkpoint.

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
| `PRESENTER_KEY` | Protects the Ask box and the controls. Open `/presenter?key=<value>` once on the presenter laptop |
| `MAX_INFLIGHT_PER_VARIANT` | Caps concurrent requests per deployment (default 32) |
| `GPU_HOURLY_USD` | Shows cost per request when set, labeled as an assumption |

The speculative decoding deployment that was benchmarked:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 2 \
  --speculative-config '{"method": "draft_model", "model": "meta-llama/Llama-3.1-8B-Instruct", "num_speculative_tokens": 5}'
```

If you don't have H200s, the same setup works with Llama 3.1 8B as the target and Llama 3.2 1B as the draft on a single 24 GB GPU. The absolute numbers change, and the comparison is still apples to apples.

## Deploying to OpenShift

```bash
podman build -t quay.io/<your-org>/pytorch-quantization-demo:latest .
podman push quay.io/<your-org>/pytorch-quantization-demo:latest
oc create secret generic openshift-ai-config --from-literal=endpoint=<url> --from-literal=token=<token> \
  --from-literal=presenter_key=$(openssl rand -hex 16)
oc apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml -f kubernetes/route.yaml
```

The deployment runs one replica on purpose. Metrics, votes, and websocket connections live in memory, so a second replica would split the room in half.

## Keyboard shortcuts

| Key | What it does |
|---|---|
| `1` `2` `3` | Ask, Under load, Numbers |
| `Space` | Play the load run (on Under load) |
| `/` | Jump to the question box |
| `Enter` | Send the question to every setup |
| `T` | Light or dark theme |
| `F` | Full screen |

To force replay mode while presenting, open `/presenter?mode=sim`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Project layout

```
app/                 FastAPI backend (simulation, live vLLM client, metrics, arena relay, rate limits)
arena/               PyTorch training and quantization for the arena policies
static/arena/        Exported policy weights the browser runs
static/js/           Presenter dashboard, arena engine
templates/           Presenter and audience pages
benchmark_results.json, bench/, quality/   Measured data the dashboard reads
scripts/             Setup, local run, and benchmark scripts
kubernetes/          OpenShift manifests
tests/               API, simulation, metrics, rate limit, and arena tests
```

## Credits and license

MIT. The arena's physics and game loop are adapted from [FlappyLearning](https://github.com/xviniette/FlappyLearning) by Vincent Bazia (MIT), and the Red Hat fonts are bundled under the SIL Open Font License. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Markell Rawls**, AI Developer Advocate, Red Hat
