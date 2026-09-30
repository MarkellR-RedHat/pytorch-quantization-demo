# PyTorch Conference NA 2026: Not Every Question Needs Two GPUs

**Conference:** PyTorch Conference North America 2026, San Jose  
**Session:** Demo Theater, Tuesday October 20, 4:10 PM PDT  
**Format:** 10-minute live demo  
**Repo:** https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## What This Demo Does

Llama 3.1 70B Instruct runs on vLLM in three setups on NVIDIA H200s: BF16 on two GPUs, INT4 (Red Hat's LLM Compressor build) on one, and speculative decoding with an 8B draft model on two. During the demo Markell types a question from the room, it goes to all three at once, and the answers stream side by side with their timing and GPU count. The Numbers scene then shows what each setup gets you and when a router earns its keep.

What the audience sees, backed by data: one request at a time, INT4 runs at 89% of BF16's speed on half the GPUs and serves about the same output per GPU under load, and speculative decoding runs about 1.25× faster than BF16 on the same two GPUs (with a first token about 3× slower, and about half of BF16's tokens per GPU under load). GSM8K shows no loss for INT4 on 1,319 questions; on MMLU-Pro it scored 3 to 4 points lower on 280 questions, which is too few to call it, so the hard questions stay on BF16 until it's tested further.

## How It Works Under the Hood

The backend is FastAPI (`app/`). The presenter page (`/presenter`) has three scenes. **Ask** streams one question to every setup through the vLLM endpoints (or replays the measured speeds when they aren't connected), **Under load** replays the `vllm bench serve` load test, and **Numbers** shows the benchmark, accuracy, what each setup is best for, and the router guidance. The Quantization Arena game lives at `/arena` for the booth.

The full method, the numbers, and the technical questions are in the README.

## The Talk Flow (10 Minutes)

<!-- TALK FLOW: owned by the slides session; keep in sync with the speaker notes in slides.html -->

The talk flow and the full speaker script live in the speaker notes of `slides.html` (press `N` while presenting). The demo slide calls for a question from the room: press `1` for Ask, type the question, and press `Enter`. Then finish on Numbers, the money slide: press `2` while the load test hasn't been recorded (Under load stays out of the numbered flow until `bench/<VARIANT>/c<N>.json` exists). Once it has, press `2` for Under load and `Space` to play the load run, then `3` for Numbers.

## Presenter Shortcuts

| Key | What it does |
|---|---|
| `1` `2` `3` | Ask, Under load, Numbers (`1` `2` are Ask and Numbers until the load test is recorded) |
| `Space` | Play the load run (on Under load) |
| `/` | Jump to the question box |
| `Enter` | Send the question to every setup |
| `R` | Switch every column between live models and replay |
| `T` | Light or dark theme |
| `F` | Full screen |

To force replay mode while presenting, press `R` or open `/presenter?mode=sim`.

## The Plan

### GPUs per setup

- BF16 Llama 3.1 70B: 2x H200 (tensor parallel)
- INT4 Llama 3.1 70B, Red Hat's LLM Compressor build (`RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16`): 1x H200
- Speculative decoding (Llama 3.1 70B BF16 target + Llama 3.1 8B draft, both tensor parallel 2): 2x H200

### Pre-Demo Checklist (Day Before)

- All model variants deployed and answering (`/v1/models` on each vLLM endpoint)
- Demo app deployed behind an OpenShift Route with `PRESENTER_KEY` set
- Presenter laptop opened `/presenter?key=<value>` once, so the Ask box works
- One typed question and all eight presets answered live on the demo laptop
- `/presenter?mode=sim` tested as the fallback
- Backup video recorded and on a USB drive

### Oct 19: final testing and fresh recordings

Once all five setups are up (BF16 on 2 H200s, spec decode on 2, Red Hat's INT4 on 1) and warm, re-record every preset from the day's deployments, so the answers the fallback plays are warm and from the exact models on stage. From a folder with the port-forwards up:

```bash
python3 scripts/capture_presets.py FP16        http://localhost:18001/v1/chat/completions benchmark-bf16
python3 scripts/capture_presets.py INT4_RH     http://localhost:18002/v1/chat/completions benchmark-int4-rh
python3 scripts/capture_presets.py SPEC_DECODE http://localhost:18003/v1/chat/completions benchmark-spec
```

About 5 minutes per setup. It writes `results-2/quality/<VARIANT>/`; copy those folders over `quality/<VARIANT>/` in the repo, run `python scripts/preflight.py`, then commit the new `quality/` files. Until this is done, the "Explain KV cache" preset has no usable recording: its prompt was reworded on Sep 30 (to say "in a transformer LLM", because every setup had explained a generic key-value store), and preflight fails that preset on purpose. The round-2 recording of the sheep riddle on spec decode was also a cold first request (first token 1.1 s), which this step replaces.

### 30 Minutes Before

- Run the preflight from the repo root, with the same `.env` as the app:
  ```bash
  python scripts/preflight.py
  ```
  For each setup it checks that `/v1/models` lists the served name, that a 1-token completion answers, sends 3 warm-up requests so the first live answer isn't cold, and confirms all eight presets have a recording to fall back on. It ends with what the corner badge will say and one PASS/FAIL table, and exits 1 if anything failed. It uses only the endpoints in `.env`, so it needs no cluster login.
- Open the presenter dashboard full screen (`F`)
- Check the badge in the top right says Live models
- Silence notifications and close other apps

## Backup Plans

**If one model stops responding:** nothing to do for the presets. After 8 seconds with no first token, or 10 seconds of silence mid-answer, that column plays its recorded answer under a red line that says "Recorded <date> · live request failed", and its badge says RECORDED. Mention it when it happens.

**If all the models stop responding:** press `R` (or open `/presenter?mode=sim`). The Numbers scene doesn't change, because it shows the measured benchmark. In Ask, a "Replay" badge appears and each column plays its recorded answer at the speed that setup measured. Press `R` again to go back to live.

**If a setup's GPUs get pulled before the talk:** set `MODEL_<VARIANT>_MODE=recorded` for it (for example `MODEL_SPEC_DECODE_MODE=recorded`) and restart the app. That column never calls its endpoint, plays its recorded preset answers labeled "Recorded <date>", and the corner badge says which setups are live and which are recorded.

**If the venue network is bad:** run the app on the demo laptop with `./scripts/run-local.sh`. Fonts and everything else are bundled, so the screen renders correctly with no network.

**If nobody calls out a question:** use the eight preset buttons under the question box. Sheep riddle and Logic puzzle go to BF16, the four everyday ones go to INT4, and Python function and Explain KV cache go to Spec Decode. Explain KV cache is the long-answer, latency case, where Spec Decode's speed shows most.

**If everything fails:** play the backup video from the USB drive, narrate over it, and move to Q&A.

## Equipment

- Laptop with presenter dashboard open
- HDMI adapter (and a backup one)
- Backup laptop with full setup
- USB drive with backup video

## Running It Yourself

Everything works locally with no GPUs or external services:

```bash
git clone https://github.com/MarkellR-RedHat/pytorch-quantization-demo.git
cd pytorch-quantization-demo
./scripts/setup.sh
./scripts/run-local.sh
```

Open http://localhost:8000/presenter for the presenter dashboard.

## Author

**Markell Rawls**  
AI Developer Advocate, Red Hat  
mrawls@redhat.com
