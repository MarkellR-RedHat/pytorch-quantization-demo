# PyTorch Conference NA 2026: Not Every Question Needs the Whole GPU

**Session:** Demo Theater, 10 minutes, San Jose.
**Repo:** https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## The story

Two models on vLLM on NVIDIA H200s, each at three precisions with speculative decoding as an add-on on BF16 (four columns in the demo). Llama 3.1 70B Instruct: BF16 on two GPUs, FP8 on one, INT4 (Red Hat's validated build: W4A16, GPTQ via AutoGPTQ) on one, and speculative decoding with an 8B draft model on two. Qwen3.8-27B: BF16 on one GPU, FP8 and INT4 (Red Hat's build: W4A16, GPTQ via LLM Compressor) on 71 GB MIG slices, and speculative decoding with the model's own MTP head on the same GPU. A question from the room goes to all four setups of the track on screen at once (`Q` switches tracks) and the answers stream side by side with their timing and device; Under load shows what each device serves at a latency target; Numbers is the money slide and says when a router earns its keep. The Llama lanes: FP8 for everyday questions on Hopper, INT4 where 73 GB of weights won't fit, BF16 for the hardest questions until the others are tested on them, and spec decode as a switch you turn on in the BF16 lane for anyone waiting on the answer (measured on the BF16 target). The Qwen lanes: FP8 for everyday traffic (two replicas per H200), INT4 for long contexts on a slice, BF16 with its MTP head on for anyone waiting on the answer, plain BF16 for batch work where nobody waits. The numbers behind every line are in the README's two results tables.

## The talk flow

The speaker script lives in the notes of `slides.html` (press `N` while presenting). The demo slide calls for a question from the room: press `1` for Ask, type it, `Enter`. Then `2` for Under load and `Space` to play the load run (about 30 seconds), and finish on `3`, Numbers.

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
| Python function | Second largest number in a list, with edge cases | BF16 + spec decode, long answers with someone waiting |
| Explain KV cache | The KV cache in a transformer LLM, for a new engineer, in about 300 words | BF16 + spec decode (the long-answer, latency case) |

The exact wording lives in `app/quality.py` (`PROMPTS`), and `scripts/capture_presets.py` records every setup's answer to each one at temperature 0 with the same 1,024-token cap as live Ask.

**If a live request fails.** When a setup errors, sends no first token within 8 seconds, or goes quiet for 10 seconds mid-answer, that column plays its recorded answer to the preset instead. Any live text it already streamed stays above a red dashed line, the line says "Recorded <date> · live request failed after N tokens", and the column is badged RECORDED, so a recording is never passed off as live. A typed question has no recording, so the column says the live request failed. Press `R` to switch every column between the live models and replay in one keystroke.

**Under load** (press `2`, then `Space`). A replay of the `vllm bench serve` load test: 1, 8, 16, 32, then 64 requests in flight at once against each setup (synthetic random-token prompts), with output tokens per second per GPU, total output tokens per second, the tail time per output token (TPOT, p95 when the run recorded it, otherwise vLLM's default p99), and median time per answer at each step. What sets cost is how many tokens a GPU serves while users still get a fast stream, so each card marks a latency target on its TPOT chart (`TPOT_TARGET_MS`, default 50 ms, which is 20 tokens per second per user) and ends with its best output tokens per GPU under that target. The result line compares INT4 and Spec Decode with BF16 on that number, and the Numbers screen's cost line uses it too. It plays the sweep files under `bench/raw/2026-09-29-r2/sweeps/` on the Llama track and `bench/raw/2026-09-30-qwen-r1/sweeps/` on the Qwen track.

**Numbers** (press `3`). The money slide: GPUs, tokens per second and mean time for one request, and accuracy for each setup (GSM8K on the card, MMLU-Pro with its sample size in the footnote), what each one is best for, what to watch out for, and when a router is worth adding.

**The arena** (http://localhost:8000/arena) is a booth game, not part of the talk; see [ARENA.md](ARENA.md).

## Keys

| Key | What it does |
|---|---|
| `1` `2` `3` | Ask, Under load, Numbers |
| `Space` | Play the load run (on Under load) |
| `/` | Jump to the question box |
| `Enter` | Send the question to every setup |
| `R` | Switch every column between live models and replay |
| `Q` | Switch tracks (Llama 3.1 70B, Qwen3.8-27B) |
| `T` | Light or dark theme |
| `F` | Full screen |

## The plan

Every column on both tracks plays its recording. The recordings are the answers each setup gave in the measured runs, captured at temperature 0, and they play at the speed that setup measured; each column carries a Recorded pill and the corner badge says so. Nothing is run or recorded again: the numbers and the recordings in the repo are final, and the talk needs no GPU, no cluster and no VPN. The app runs on whichever laptop presents.

| Track | Columns | Badge |
|---|---|---|
| Llama 3.1 70B | BF16, FP8, INT4 (`quality/INT4_RH/`), BF16 + spec decode, all recorded | "Llama 70B · Recorded: BF16, FP8, INT4 (GPTQ via AutoGPTQ), BF16 + spec decode" |
| Qwen3.8-27B | BF16, FP8, INT4, BF16 + MTP, all recorded | "Qwen 27B · Recorded: BF16, FP8, INT4 (GPTQ via LLM Compressor), BF16 + MTP" |

A preset button is offered only when every recorded column has an answer for it. The Llama track offers seven: its "Explain KV cache" prompt was reworded after the run and has no Llama recording. The Qwen track offers all eight. A typed question has no recording, so each column says "Not live today. Recorded answers cover the preset questions." The Llama spec decode column's sheep riddle recording is the afternoon capture of the same answer (the evening one was a cold first request), paced at the benchmark speed and labeled so.

If one OpenAI-compatible endpoint serving one of the Llama setups exists on the day, it can go in one column; everything else stays recorded and the talk works the same without it.

### Setup, once

`make setup`, then this `.env`, then `make serve`, and open `http://localhost:8000/presenter?key=<value>`:

```bash
SIMULATION_MODE=false
PRESENTER_KEY=<a long random string>
TRACK=llama
MODEL_BF16_MODE=recorded
MODEL_SPEC_DECODE_MODE=recorded
MODEL_FP8_MODE=recorded
MODEL_INT4_MODE=recorded
MODEL_INT4_CAPTURES=INT4_RH
QWEN_MODEL_BF16_MODE=recorded
QWEN_MODEL_FP8_MODE=recorded
QWEN_MODEL_INT4_MODE=recorded
QWEN_MODEL_SPEC_DECODE_MODE=recorded
```

To put one endpoint in one column, set that column's mode to live and add its two lines, for example FP8:

```bash
MODEL_FP8_MODE=live
MODEL_FP8_ENDPOINT=<the URL, ending /v1/chat/completions>
MODEL_FP8_NAME=<its served model name>
```

### An hour before the talk

- `make serve`, then `python scripts/preflight.py` from the repo root with the same `.env`. With every column recorded it checks that every preset on screen has a recording on both tracks and prints the badge each track will show. It ends `PASS: 0 failed, 0 warnings, 18 checks` and exits 1 on any FAIL. With one live column it also checks that endpoint lists the served name and answers, and sends 3 warm-up requests.
- The badges match the table above (with a live column, that column moves to "Live:").
- `http://localhost:8000/presenter` full screen (`F`), notifications off, other apps closed.

## If something goes wrong

**The live column stops responding:** nothing to do for the presets. After 8 seconds with no first token, or 10 seconds of silence mid-answer, that column plays its recorded answer under a red line that says "Recorded <date> · live request failed", and its badge says RECORDED.

**The laptop app breaks:** open the static replay, https://markellr-redhat.github.io/pytorch-quantization-demo/demo/, in the same browser. It plays the same recordings and the same measured runs on both tracks with no server.

**Everything fails:** play the backup video from the USB drive, narrate over it, and move to Q&A.

## Equipment

- Laptop with the presenter dashboard open
- HDMI adapter, and a backup one
- Backup laptop with the same setup
- USB drive with the backup video
