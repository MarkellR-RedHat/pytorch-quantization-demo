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

**Under load** (press `2`, then `Space`). A replay of the `vllm bench serve` load test: 1, 8, 32, then 64 questions in flight at once against each setup, with tokens per second per GPU, total tokens per second, and median time per answer at each step, and the result in one line at the end. It plays back whatever sweep files are in `bench/`, and says so when there aren't any yet.

**Numbers** (press `3`). The money slide: tokens per second for one request, tokens per second per GPU, and accuracy for each setup, what each one is best for, and when a router is worth adding.

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

Measured on September 29, 2026 on NVIDIA H200 (141 GB) with vLLM `0.18.0+rhaiv.14`, the vLLM build that ships with Red Hat AI. Each variant got 20 requests sent one at a time, with up to 256 output tokens.

| Variant | GPUs | Mean time per request | p95 | Tokens/s per stream | Tokens/s per GPU |
|---|---|---|---|---|---|
| BF16 (tensor parallel 2) | 2 | 4.96 s | 5.53 s | 47.3 | 23.6 |
| INT4 AWQ (`awq_marlin`) | 1 | 5.07 s | 5.84 s | 44.3 | 44.3 |
| Spec decode (70B + 8B draft, 5 tokens) | 2 | 5.54 s | 8.65 s | 40.0 | 20.0 |

The raw file is `benchmark_results.json`. The dashboard reads it directly, and `scripts/benchmark.py` produces it.

These are single-stream numbers, so they tell you about latency and not about how much traffic a GPU can serve. The load test (`vllm bench serve` at 1, 8, 32, and 64 concurrent requests) drops into `bench/<VARIANT>/c<N>.json`, and replay mode scales its timings by concurrency once those files exist.

## Technical questions

**Why not FP8?**
Llama 3.1 70B in FP8 is about 71 GB, so it fits on one H200 with room left for KV cache, and Hopper GPUs have native FP8 tensor cores. Red Hat's FP8 build of this model ([RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8)) keeps 99.9% of the BF16 score on the OpenLLM v1 benchmarks, so it's the next variant to add.

**Why was speculative decoding slow in this benchmark?**
The run used `enforce_eager`, which in vLLM 0.18 turns off both `torch.compile` and CUDA graphs. Speculative decoding runs many small forward passes (the 8B draft proposes five tokens for every 70B pass), so it loses the most when every pass pays full kernel-launch cost. Draft-model speculative decoding in vLLM supports CUDA graphs, so a rerun without `enforce_eager` is the fair comparison. Also keep in mind that spec decode uses the same GPUs as BF16. It buys lower latency per request, not fewer GPUs.

**Does this hold for an 8B model?**
The trade-offs have the same shape, but smaller models lose more when quantized. In Red Hat's quantization study, INT4 kept 97.4% of the 70B's score on the OpenLLM v2 benchmarks and 96.1% of the 8B's, so test an 8B carefully on your own prompts.

**AWQ or GPTQ?**
Both produce INT4 weights that vLLM serves with fast mixed-precision kernels. Red Hat's study found GPTQ slightly ahead on harder benchmarks, and Red Hat's published INT4 build of this model ([RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16)) is GPTQ.

**Where do these numbers come from?**
Red Hat's quantization study is Kurtic et al., ["Give Me BF16 or Give Me Death? Accuracy-Performance Trade-Offs in LLM Quantization"](https://arxiv.org/abs/2411.02355), ACL 2025, which ran more than 500,000 evaluations. The benchmark numbers in this repo come from 20 sequential requests per variant on NVIDIA H200s with vLLM 0.18, so they measure one request at a time and not throughput under heavy load.

**Why isn't INT4 faster than BF16?**
At one request at a time, decoding is limited by how fast the GPU can read the weights. BF16 across two H200s reads about 70 GB per GPU per token against 4.8 TB/s each, and INT4 on one H200 reads about 35 to 40 GB. BF16 lands at about 70% of its bandwidth ceiling and INT4 at a much lower fraction, which is typical for 4-bit kernels at batch size 1. The result that matters for the bill is that INT4 runs at 94% of BF16's speed on half the GPUs, so it gets 1.87 times the tokens per GPU.

**Is speculative decoding really lossless?**
The verification step keeps the output distribution of the target model, so the quality you get is the target's quality. vLLM describes it as lossless up to the precision limits of hardware numerics, so greedy outputs can differ in rare cases because verification runs with a different batch shape.

**Which INT4 checkpoint was this?**
vLLM reported `awq_marlin`, which is the AutoAWQ checkpoint format. LLM Compressor writes the `compressed-tensors` format and supports AWQ through its `AWQModifier`.

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
