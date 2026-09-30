# PyTorch Conference NA 2026: Not Every Question Needs Two GPUs

**Session:** Demo Theater, Tuesday October 20, 4:10 PM PDT, 10 minutes, San Jose.
**Repo:** https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## The story

Llama 3.1 70B Instruct runs on vLLM in four setups on NVIDIA H200s: BF16 on two GPUs, FP8 on one, INT4 (Red Hat's validated W4A16 build) on one, and speculative decoding with an 8B draft model on two. A question from the room goes to all four at once and the answers stream side by side with their timing and GPU count; Under load shows what each GPU serves at a latency target; Numbers is the money slide and says when a router earns its keep. The lanes: FP8 for everyday questions on Hopper, INT4 where 73 GB of weights won't fit, BF16 for the hardest questions until the others are tested on them, spec decode where latency matters. The numbers behind every line are in the README's results table.

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
| Python function | Second largest number in a list, with edge cases | Spec Decode, long answers with someone waiting |
| Explain KV cache | The KV cache in a transformer LLM, for a new engineer, in about 300 words | Spec Decode (the long-answer, latency case) |

The exact wording lives in `app/quality.py` (`PROMPTS`), and `scripts/capture_presets.py` records every setup's answer to each one at temperature 0 with the same 1,024-token cap as live Ask.

**If a live request fails.** When a setup errors, sends no first token within 8 seconds, or goes quiet for 10 seconds mid-answer, that column plays its recorded answer to the preset instead. Any live text it already streamed stays above a red dashed line, the line says "Recorded <date> · live request failed after N tokens", and the column is badged RECORDED, so a recording is never passed off as live. A typed question has no recording, so the column says the live request failed. Press `R` to switch every column between the live models and replay in one keystroke.

**Under load** (press `2`, then `Space`). A replay of the `vllm bench serve` load test: 1, 8, 32, then 64 requests in flight at once against each setup (synthetic random-token prompts), with output tokens per second per GPU, total output tokens per second, the tail time per output token (TPOT, p95 when the run recorded it, otherwise vLLM's default p99), and median time per answer at each step. What sets cost is how many tokens a GPU serves while users still get a fast stream, so each card marks a latency target on its TPOT chart (`TPOT_TARGET_MS`, default 50 ms, which is 20 tokens per second per user) and ends with its best output tokens per GPU under that target. The result line compares INT4 and Spec Decode with BF16 on that number, and the Numbers screen's cost line uses it too. It plays the sweep files under `bench/raw/2026-09-29-r2/sweeps/`.

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
| `Q` | Switch tracks (Llama 3.1 70B, Qwen3.8-27B) once both have data |
| `T` | Light or dark theme |
| `F` | Full screen |

## The plan: five H200s

| Setup | On the day |
|---|---|
| BF16, tensor parallel 2 | 2x H200, live |
| Speculative decoding, tensor parallel 2 | 2x H200, live |
| FP8 | 1x H200, live |
| INT4 | not deployed: the column plays the Oct 19 recordings (`MODEL_INT4_MODE=recorded`), labeled "Recorded <date>", and the corner badge says so |

FP8 needs the full GPU; INT4, if it were live, would fit a 71 GB MIG slice (untested), which would leave the full GPUs for other people's work. The sizing table is in the README.

### Oct 19: final testing and fresh recordings

Re-record every preset from the day's deployments, so the answers the fallback and the INT4 column play are warm and from the exact models on stage. INT4 borrows the fifth GPU first:

1. Deploy BF16, spec decode and INT4 with `kubernetes/models/isvc-bf16-tp2.yaml`, `isvc-spec-decode.yaml` and `isvc-int4-rh.yaml` (`oc apply -n <namespace>`). Wait until all three are Ready, then warm each up (`python scripts/preflight.py` sends the warm-ups).
2. With the port-forwards up, record INT4, which writes straight into `quality/INT4_RH/`:
   ```bash
   python3 scripts/capture_presets.py INT4_RH http://localhost:18002/v1/chat/completions benchmark-int4-rh
   ```
3. Delete the INT4 InferenceService, deploy FP8 with `isvc-fp8.yaml` on the freed GPU, wait for Ready, warm it up.
4. Record the three live setups:
   ```bash
   python3 scripts/capture_presets.py BF16        http://localhost:18001/v1/chat/completions benchmark-bf16
   python3 scripts/capture_presets.py SPEC_DECODE http://localhost:18003/v1/chat/completions benchmark-spec
   python3 scripts/capture_presets.py FP8         http://localhost:18004/v1/chat/completions benchmark-fp8
   ```
5. `python scripts/preflight.py` with the day-of `.env` (INT4 passes on its recordings alone), then commit `quality/`.

About 5 minutes per setup. If the day runs short, the Sep 29 recordings already in `quality/` are acceptable, all except "Explain KV cache": that prompt was reworded on Sep 30 (to say "in a transformer LLM", because every setup had explained a generic key-value store) and has no recording until this step, so preflight fails it on purpose. The Sep 29 evening recording of the sheep riddle on spec decode was a cold first request, so `quality/` plays the afternoon capture of the same answer (paced at the benchmark speed, labeled so) until this step replaces it. In replay, and for a recorded-by-plan setup, a preset button is only offered when every setup on screen has a recording for it.

### The day before

- BF16, spec decode and FP8 deployed and answering (`/v1/models` on each endpoint)
- The app runs on the presenter laptop, the one plugged into the projector: `make setup` once, then this `.env` (port-forwards on 18001, 18003 and 18004, or the VPN routes in their place), then `make serve`:
  ```bash
  SIMULATION_MODE=false
  PRESENTER_KEY=<a long random string>
  MODEL_BF16_ENDPOINT=http://localhost:18001/v1/chat/completions
  MODEL_BF16_NAME=benchmark-bf16
  MODEL_SPEC_DECODE_ENDPOINT=http://localhost:18003/v1/chat/completions
  MODEL_SPEC_DECODE_NAME=benchmark-spec
  MODEL_FP8_ENDPOINT=http://localhost:18004/v1/chat/completions
  MODEL_FP8_NAME=benchmark-fp8
  MODEL_INT4_MODE=recorded
  MODEL_INT4_CAPTURES=INT4_RH
  ```
- The laptop opened `http://localhost:8000/presenter?key=<value>` once, so the Ask box works
- One typed question and all eight presets answered live
- The backup, the same app behind an OpenShift Route (README, Deploy), is up with the same values
- `/presenter?mode=sim` tested as the fallback
- Backup video recorded and on a USB drive

### 30 minutes before

- `python scripts/preflight.py` from the repo root with the same `.env` as the app. For each setup it checks that `/v1/models` lists the served name and a 1-token completion answers, sends 3 warm-up requests so the first live answer isn't cold, and confirms every preset has a recording. It prints what the corner badge will say and one PASS/FAIL table, and exits 1 on any FAIL. It needs no cluster login.
- Port-forwards (or VPN) up, app started, `http://localhost:8000/presenter` full screen (`F`)
- The badge in the top right says "Live: BF16, FP8, Spec Decode · Recorded: INT4 (Red Hat W4A16)"
- Notifications off, other apps closed

## If something goes wrong

**One model stops responding:** nothing to do for the presets. After 8 seconds with no first token, or 10 seconds of silence mid-answer, that column plays its recorded answer under a red line that says "Recorded <date> · live request failed", and its badge says RECORDED. Mention it when it happens.

**All the models stop responding:** press `R` (or open `/presenter?mode=sim`). Numbers doesn't change, because it shows the measured benchmark. In Ask, a "Replay" badge appears and each column plays its recording. `R` again goes back to live.

**A setup's GPUs get pulled before the talk:** set `MODEL_<VARIANT>_MODE=recorded` for it and restart the app. That column never calls its endpoint, plays its recordings labeled "Recorded <date>", and the badge says which setups are live and which are recorded.

**The laptop app breaks:** open the backup Route (`https://<route>/presenter?key=<value>`) in the same browser; same code, same recordings. If the venue network is bad, stay on the laptop: fonts and everything else are bundled, and `R` plays the recordings.

**Nobody calls out a question:** use the eight preset buttons. Sheep riddle and Logic puzzle go to BF16, the four everyday ones to FP8, Python function and Explain KV cache to Spec Decode.

**Everything fails:** play the backup video from the USB drive, narrate over it, and move to Q&A.

## Equipment

- Laptop with the presenter dashboard open
- HDMI adapter, and a backup one
- Backup laptop with the full setup
- USB drive with the backup video
