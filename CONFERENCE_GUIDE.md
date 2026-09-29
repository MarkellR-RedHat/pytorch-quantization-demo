# PyTorch Conference NA 2026: Quantization Showdown

**Conference:** PyTorch Conference North America 2026, San Jose  
**Session:** Demo Theater, Tuesday October 20, 4:10 PM PDT  
**Format:** 10-minute live demo  
**Repo:** https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## What This Demo Does

Llama 3.1 70B Instruct runs on vLLM in three setups on NVIDIA H200s: BF16 on two GPUs, INT4 AWQ on one, and speculative decoding with an 8B draft model on two. During the demo Markell types a question from the room, it goes to all three at once, and the answers stream side by side with their timing and GPU count. The Numbers scene then shows what each setup gets you and when a router earns its keep.

What the audience sees, backed by data: INT4 runs at 94% of BF16's speed on half the GPUs, which works out to 1.9 times the tokens per GPU. Speculative decoding runs on the same two GPUs as BF16, so what it buys is lower latency per request, and this benchmark ran it with CUDA graphs off.

## How It Works Under the Hood

The backend is FastAPI (`app/`). The presenter page (`/presenter`) has three scenes. **Ask** streams one question to every setup through the vLLM endpoints (or replays the measured speeds when they aren't connected), **Under load** replays the `vllm bench serve` load test, and **Numbers** shows the benchmark, accuracy, what each setup is best for, and the router guidance. The Quantization Arena game lives at `/arena` for the booth.

The full method, the numbers, and the technical questions are in the README.

## The Talk Flow (10 Minutes)

<!-- TALK FLOW: owned by the slides session; keep in sync with the speaker notes in slides.html -->

The talk flow and the full speaker script live in the speaker notes of `slides.html` (press `N` while presenting). The demo slide calls for a question from the room: press `1` for Ask, type the question, and press `Enter`. Then press `2` for Under load and `Space` to play the load run, and finish on `3`, Numbers, the money slide.

## Presenter Shortcuts

| Key | What it does |
|---|---|
| `1` `2` `3` | Ask, Under load, Numbers |
| `Space` | Play the load run (on Under load) |
| `/` | Jump to the question box |
| `Enter` | Send the question to every setup |
| `T` | Light or dark theme |
| `F` | Full screen |

To force replay mode while presenting, open `/presenter?mode=sim`.

## The Plan

### GPU Booking

| Dates | GPUs | Purpose |
|-------|------|---------|
| Sep 29-30 | 5x H200 Full | Test run, end-to-end validation, record backup video |
| Oct 18-21 | 5x H200 Full | Pre-deploy (Oct 18), final testing (Oct 19), live conference (Oct 20-21) |

GPU breakdown per variant:
- FP16 Llama 70B: 2x H200 (tensor parallel)
- INT4 Llama 70B: 1x H200
- Speculative Decode (Llama 3.1 70B BF16 target + Llama 3.1 8B draft, both tensor parallel 2): 2x H200

### Timeline

1. **Sep 29-30:** Test run on H200s. Deploy all three model variants, run through the full demo flow, record backup video.
2. **Oct 18:** Pre-deploy the demo app and models to OpenShift AI.
3. **Oct 19:** Final testing on production URL. Upload slides to Sessionize (deadline).
4. **Oct 20:** Demo day. 4:10 PM PDT, Demo Theater. Be set up and tested 30 minutes before.
5. **Oct 21:** Conference day 2, booth availability.

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

Open http://localhost:8000/presenter for the presenter dashboard and http://localhost:8000 for the audience view.

## Contacts

- **Event logistics:** Juliana Furlow (jsweek@redhat.com)
- **vLLM / llm-d:** Sasa
- **Repo:** https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## Author

**Markell Rawls**  
AI Developer Advocate, Red Hat  
mrawls@redhat.com
