# Quantization Showdown

Everyone who has shipped a 70B model has had the same conversation with their GPU bill, and the first answer is always to quantize it. This demo puts that decision on a big screen. It shows real numbers from Llama 3.1 70B Instruct served by vLLM on NVIDIA H200s in three ways (BF16 on two GPUs, INT4 AWQ on one, and speculative decoding with an 8B draft model), and it runs a small game where the audience throws hard prompts at birds flown by neural networks stored at different precisions, so you can watch a rounding error decide who makes it through.

Built for the Demo Theater at PyTorch Conference North America 2026 in San Jose.

## Run it on your laptop

You don't need a GPU or a cluster. Simulated mode replays the timings measured on the H200s, and everything (fonts included) is bundled, so it works with no network at all.

```bash
git clone https://github.com/MarkellR-RedHat/pytorch-quantization-demo.git
cd pytorch-quantization-demo
./scripts/setup.sh
./scripts/run-local.sh
```

Open http://localhost:8000/presenter on the big screen and http://localhost:8000 on your phone (use your laptop's IP address instead of localhost). You need Python 3.11 or newer.

## What's on the screen

The presenter dashboard is a fixed 1920 by 1080 stage that scales to whatever it's plugged into, so nothing reflows or clips on a 720p projector.

**Arena** (press `1`). Five birds fly the same course. Every one of them is flown by the same 4,673-parameter network I trained in PyTorch, and the only thing that changes between them is how its weights are stored. Red gaps are hard prompts, 112 px wide instead of 170, and people in the room add more of them from their phones.

| Bird | Weights | Hard gaps cleared* |
|---|---|---|
| BF16 | 16-bit brain float | 219 of 219 |
| FP8 | E4M3 with one scale per output row | 219 of 219 |
| INT4 AWQ | 4-bit, group size 16, activation-aware scaling | 219 of 219 |
| INT4 RTN | 4-bit, group size 16, plain round to nearest | 135 of 220 |
| Spec Decode | BF16 network plus a 4-unit draft network | same path as BF16, to the pixel |

\* Measured by `arena/train_policy.py` on 20 held-out courses. Every variant also clears every easy gap.

**Numbers** (press `2`). The H200 benchmark for each deployment, plus a "right now" panel with whatever traffic is flowing (live requests in live mode, replayed timings in simulated mode). Every number carries a label that says where it came from.

**Quality** (press `3`). The same prompt sent to every variant. These are marked "illustrative" until real captures from the deployments are dropped into `quality/`, and then the dashboard shows those instead with no code change.

**Phones.** People scan the QR code on the Arena sidebar, back a bird, throw hard prompts at the course, and send requests to the real models. Nothing anyone types is sent anywhere, because every prompt is fixed on the server.

## How the arena works, and how to check it

I wanted the game to hold up if someone who works on quantization kernels looks at it for more than ten seconds, so nothing in it is scripted.

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

These are single-stream numbers, so they tell you about latency and not about how much traffic a GPU can serve. The load test (`vllm bench serve` at 1, 8, 32, and 64 concurrent requests) drops into `bench/<VARIANT>/c<N>.json`, and the Numbers scene draws the under-load chart as soon as those files exist.

## Questions people ask

**Why isn't INT4 faster than BF16?** At one request at a time, decoding is limited by how fast the GPU can read the weights. BF16 across two H200s reads about 70 GB per GPU per token against 4.8 TB/s each, and INT4 on one H200 reads about 35 to 40 GB. BF16 lands at about 70% of its bandwidth ceiling and INT4 at a much lower fraction, which is typical for 4-bit kernels at batch size 1. The result that matters for the bill is that INT4 runs at 94% of BF16's speed on half the GPUs, so it gets 1.87 times the tokens per GPU.

**Why is speculative decoding the slowest?** The benchmark deployment ran with `enforce_eager=true`, which turns off torch.compile and CUDA graphs for both the 70B target and the 8B draft. The draft also runs at the same tensor parallel size as the target in vLLM 0.18, so it pays for cross-GPU communication on every step. A rerun without `enforce_eager`, along with the acceptance rate from vLLM's `spec_decode` metrics, is the next measurement.

**Is speculative decoding really lossless?** The verification step keeps the output distribution of the target model, so the quality you get is the target's quality. vLLM describes it as lossless up to the precision limits of hardware numerics, so greedy outputs can differ in rare cases because verification runs with a different batch shape. The arena bird is bit-exact because it's a tiny network doing float64 math in your browser.

**Why not FP8?** FP8 on Hopper is the format Red Hat's own study ([Kurtic et al., 2024](https://arxiv.org/abs/2411.02355)) found effectively lossless, and a 70B model in FP8 fits on one H200. It's the FP8 bird in the arena, and the dashboard picks up a fourth column automatically when an FP8 endpoint or benchmark entry exists.

**Are the quality answers real model output?** Only when the dashboard says "captured". Until the temperature 0 captures from each deployment are saved to `quality/<VARIANT>/<scenario>.json`, the Quality scene uses illustrative text and labels it that way. INT4 on a 70B model keeps around 99% of BF16's accuracy on standard benchmarks, so a single riddle is an anecdote, and I treat it as one.

**Which INT4 checkpoint was this?** vLLM reported `awq_marlin`, which is the AutoAWQ checkpoint format. LLM Compressor writes the `compressed-tensors` format and supports AWQ through its `AWQModifier`, and Red Hat also publishes a GPTQ-based W4A16 build of Llama 3.1 70B.

**Where's PyTorch in all this?** vLLM is a PyTorch Foundation project and compiles its models with torch.compile, LLM Compressor calibrates in PyTorch, and the arena's networks are trained and quantized in PyTorch.

## Running with real models

Copy `.env.example` to `.env`, set `SIMULATION_MODE=false`, and point the endpoints at your vLLM deployments:

| Variable | What it is |
|---|---|
| `MODEL_FP16_ENDPOINT` | BF16 deployment, `/v1/chat/completions` URL |
| `MODEL_INT4_ENDPOINT` | INT4 deployment |
| `MODEL_SPEC_DECODE_ENDPOINT` | Speculative decoding deployment |
| `MODEL_FP8_ENDPOINT` | Optional FP8 deployment, adds a fourth column |
| `MODEL_<VARIANT>_NAME` | The `--served-model-name` for each deployment |
| `PRESENTER_KEY` | Protects start, stop, and reset. Open `/presenter?key=<value>` once on the presenter laptop |
| `PUBLIC_URL` | The URL phones should open, encoded in the QR code |
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
| `1` `2` `3` | Arena, Numbers, Quality |
| `Space` | Fly your own bird in the arena |
| `H` | Throw a hard prompt |
| `P` | Pause the arena |
| `R` | New course |
| `T` | Light or dark theme |
| `F` | Full screen |
| `Ctrl` `Shift` `S` | Start or stop background traffic |

To switch to simulated mode while presenting, open `/presenter?mode=sim`.

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
static/js/           Arena engine, presenter dashboard, audience phone view
templates/           Presenter and audience pages
benchmark_results.json, bench/, quality/   Measured data the dashboard reads
scripts/             Setup, local run, and benchmark scripts
kubernetes/          OpenShift manifests
tests/               API, simulation, metrics, rate limit, and arena tests
```

## Credits and license

MIT. The arena's physics and game loop are adapted from [FlappyLearning](https://github.com/xviniette/FlappyLearning) by Vincent Bazia (MIT), and the Red Hat fonts are bundled under the SIL Open Font License. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Markell Rawls**, AI Developer Advocate, Red Hat
