# PyTorch Conference NA 2026: Not Every Question Needs Two GPUs

**Conference:** PyTorch Conference North America 2026, San Jose  
**Session:** Demo Theater, Tuesday October 20, 4:10 PM PDT  
**Format:** 10-minute live demo  
**Repo:** https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## What This Demo Does

Llama 3.1 70B Instruct runs on vLLM in three setups on NVIDIA H200s: BF16 on two GPUs, INT4 AWQ on one, and speculative decoding with an 8B draft model on two. During the demo Markell types a question from the room, it goes to all three at once, and the answers stream side by side with their timing and GPU count. The Numbers scene then shows what each setup gets you and when a router earns its keep.

What the audience sees, backed by data: one request at a time, INT4 runs at 97% of BF16's speed on half the GPUs, and speculative decoding runs about 1.4× faster than BF16 on the same two GPUs. All three answered the sheep riddle correctly 20 out of 20 times.

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
| `T` | Light or dark theme |
| `F` | Full screen |

To force replay mode while presenting, open `/presenter?mode=sim`.

## The Plan

### GPUs per setup

- BF16 Llama 3.1 70B: 2x H200 (tensor parallel)
- INT4 Llama 3.1 70B: 1x H200
- Speculative decoding (Llama 3.1 70B BF16 target + Llama 3.1 8B draft, both tensor parallel 2): 2x H200

### Pre-Demo Checklist (Day Before)

- All model variants deployed and answering (`/v1/models` on each vLLM endpoint)
- Demo app deployed behind an OpenShift Route with `PRESENTER_KEY` set
- Presenter laptop opened `/presenter?key=<value>` once, so the Ask box works
- One typed question and all three presets answered live on the demo laptop
- `/presenter?mode=sim` tested as the fallback
- Backup video recorded and on a USB drive

### 30 Minutes Before

- Open the presenter dashboard full screen (`F`)
- Check the badge in the top right says Live models
- Ask one warm-up question so every model has served a request
- Silence notifications and close other apps

## Backup Plans

**If the models stop responding:** open `/presenter?mode=sim` in the same tab. The Numbers scene doesn't change, because it shows the measured benchmark. In Ask, a "Replay" badge appears and each column plays back the speed that setup measured, and preset questions still show a full answer.

**If the venue network is bad:** run the app on the demo laptop with `./scripts/run-local.sh`. Fonts and everything else are bundled, so the screen renders correctly with no network.

**If nobody calls out a question:** use the preset buttons under the question box.

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
