# PyTorch Conference NA 2026: Not Every Question Needs Two GPUs

**Conference:** PyTorch Conference North America 2026, San Jose  
**Session:** Demo Theater, Tuesday October 20, 4:10 PM PDT  
**Format:** 10-minute live demo  
**Repo:** https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## What This Demo Does

Llama 3.1 70B Instruct runs on vLLM in four setups on NVIDIA H200s: BF16 on two GPUs, FP8 on one, INT4 (Red Hat's LLM Compressor build) on one, and speculative decoding with an 8B draft model on two. During the demo Markell types a question from the room, it goes to all three at once, and the answers stream side by side with their timing and GPU count. The Numbers scene then shows what each setup gets you and when a router earns its keep.

What the audience sees, backed by data: one request at a time, INT4 runs at 89% of BF16's speed on half the GPUs and serves about the same output per GPU under load, and speculative decoding runs about 1.25× faster than BF16 on the same two GPUs (with a first token about 3× slower, and about half of BF16's tokens per GPU under load). FP8 matches BF16 one request at a time on one GPU and serves about 1.5× BF16's tokens per GPU under load, with no measurable accuracy loss, so on Hopper it's the everyday lane and INT4 is the lane for when 73 GB won't fit. GSM8K shows no loss for INT4 on 1,319 questions; on MMLU-Pro it scored 3 to 4 points lower on 280 questions, which is too few to call it, so the hard questions stay on BF16 until it's tested further.

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

Five H200s on Oct 20, so three setups are live and one is recorded by plan:

| Setup | GPUs on the day | Weights as loaded | Fits a 71 GB MIG slice? |
|---|---|---|---|
| BF16 Llama 3.1 70B, tensor parallel 2 | 2x H200, live | 65.7 GiB per GPU | No |
| Speculative decoding (BF16 70B target + Llama 3.1 8B draft, both tensor parallel 2) | 2x H200, live | 73.2 GiB per GPU | No |
| FP8 Llama 3.1 70B, Red Hat's build (`RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8`) | 1x H200, live | 67.7 GiB (72.7 GB) | No, the weights alone are more than the slice |
| INT4 Llama 3.1 70B, Red Hat's LLM Compressor build (`RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w4a16`) | none: recorded by plan | 37.1 GiB | Yes, with about 30 GB left for KV cache |

INT4's column plays the recordings made on Oct 19 (`MODEL_INT4_MODE=recorded`), labeled "Recorded <date>", and the corner badge says so. If INT4 were live it wouldn't need a full H200: a 71 GB MIG slice would do, and the full GPUs would stay free for other people's work. FP8 needs the full GPU.

### Pre-Demo Checklist (Day Before)

- All model variants deployed and answering (`/v1/models` on each vLLM endpoint)
- The demo app runs on the presenter laptop, the same laptop that's plugged into the projector. From the repo root: `./scripts/setup.sh` once, then this `.env` (port-forwards on 18001, 18003 and 18004, or the VPN routes in their place):
  ```bash
  SIMULATION_MODE=false
  PRESENTER_KEY=<a long random string>
  MODEL_FP16_ENDPOINT=http://localhost:18001/v1/chat/completions
  MODEL_FP16_NAME=benchmark-bf16
  MODEL_SPEC_DECODE_ENDPOINT=http://localhost:18003/v1/chat/completions
  MODEL_SPEC_DECODE_NAME=benchmark-spec
  MODEL_FP8_ENDPOINT=http://localhost:18004/v1/chat/completions
  MODEL_FP8_NAME=benchmark-fp8
  # INT4 is recorded by plan on the day: no endpoint, the column plays quality/INT4_RH/
  MODEL_INT4_MODE=recorded
  MODEL_INT4_CAPTURES=INT4_RH
  ```
  Start it with `source venv/bin/activate && python -m uvicorn app.main:app --port 8000`.
- The laptop opened `http://localhost:8000/presenter?key=<value>` once, so the Ask box works
- One typed question and all eight presets answered live on the laptop
- The backup, the same app behind an OpenShift Route (see Deploying to OpenShift in the README), is deployed with the same `.env` values and answers `/presenter`
- `/presenter?mode=sim` tested as the fallback
- Backup video recorded and on a USB drive

### Oct 19: final testing and fresh recordings

Re-record every preset from the day's deployments, so the answers the fallback and the INT4 column play are warm and from the exact models on stage. INT4 isn't live on the day, so it borrows the fifth GPU first, in this order:

1. Deploy BF16 (2 GPUs), spec decode (2) and INT4_RH (1) with `kubernetes/models/isvc-bf16-tp2.yaml`, `isvc-spec-decode.yaml` and `isvc-int4-rh.yaml`. Wait until all three are Ready, then send each a couple of warm-up requests (`python scripts/preflight.py` does that).
2. With the port-forwards up, record INT4_RH:
   ```bash
   python3 scripts/capture_presets.py INT4_RH     http://localhost:18002/v1/chat/completions benchmark-int4-rh
   ```
3. Delete the INT4_RH InferenceService, deploy FP8 with `kubernetes/models/isvc-fp8.yaml` (Red Hat's FP8 checkpoint on one GPU, the same pattern the Sep 30 run used) on the freed GPU, wait for Ready and warm it up.
4. Record the three live setups:
   ```bash
   python3 scripts/capture_presets.py FP16        http://localhost:18001/v1/chat/completions benchmark-bf16
   python3 scripts/capture_presets.py SPEC_DECODE http://localhost:18003/v1/chat/completions benchmark-spec
   python3 scripts/capture_presets.py FP8         http://localhost:18004/v1/chat/completions benchmark-fp8
   ```

About 5 minutes per setup. It writes `results-2/quality/<VARIANT>/`; copy those folders over `quality/<VARIANT>/` in the repo, run `python scripts/preflight.py` with the day-of `.env` (it passes INT4 on its recordings alone, since that column is recorded by plan), then commit the new `quality/` files. If Oct 19 runs short, the Sep 29 INT4_RH recordings already in `quality/INT4_RH/` are acceptable for the INT4 column, all except "Explain KV cache". Until this is done, the "Explain KV cache" preset has no usable recording: its prompt was reworded on Sep 30 (to say "in a transformer LLM", because every setup had explained a generic key-value store), and preflight fails that preset on purpose. The round-2 recording of the sheep riddle on spec decode was also a cold first request (first token 1.1 s), which this step replaces.

### 30 Minutes Before

- Run the preflight from the repo root, with the same `.env` as the app:
  ```bash
  python scripts/preflight.py
  ```
  For each setup it checks that `/v1/models` lists the served name, that a 1-token completion answers, sends 3 warm-up requests so the first live answer isn't cold, and confirms all eight presets have a recording to fall back on. It ends with what the corner badge will say and one PASS/FAIL table, and exits 1 if anything failed. It uses only the endpoints in `.env`, so it needs no cluster login.
- Start the port-forwards (or connect the VPN), start the app on the laptop, and open `http://localhost:8000/presenter` full screen (`F`)
- Check the badge in the top right says "Live: BF16, FP8, Spec Decode · Recorded: INT4"
- Silence notifications and close other apps

## Backup Plans

**If one model stops responding:** nothing to do for the presets. After 8 seconds with no first token, or 10 seconds of silence mid-answer, that column plays its recorded answer under a red line that says "Recorded <date> · live request failed", and its badge says RECORDED. Mention it when it happens.

**If all the models stop responding:** press `R` (or open `/presenter?mode=sim`). The Numbers scene doesn't change, because it shows the measured benchmark. In Ask, a "Replay" badge appears and each column plays its recorded answer at the speed that setup measured. Press `R` again to go back to live.

**If a setup's GPUs get pulled before the talk:** set `MODEL_<VARIANT>_MODE=recorded` for it (for example `MODEL_SPEC_DECODE_MODE=recorded`) and restart the app. That column never calls its endpoint, plays its recorded preset answers labeled "Recorded <date>", and the corner badge says which setups are live and which are recorded.

**If the laptop app breaks:** open the backup, the OpenShift Route (`https://<route>/presenter?key=<value>`), in the same browser. It runs the same code with the same recordings. If the venue network is bad, stay on the laptop: fonts and everything else are bundled, so the screen renders with no network, and `R` plays the recordings.

**If nobody calls out a question:** use the eight preset buttons under the question box. Sheep riddle and Logic puzzle go to BF16, the four everyday ones go to FP8 (INT4 when FP8 isn't deployed), and Python function and Explain KV cache go to Spec Decode. Explain KV cache is the long-answer, latency case, where Spec Decode's speed shows most.

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
