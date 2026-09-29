# PyTorch Conference NA 2026: Quantization Showdown

**Conference:** PyTorch Conference North America 2026, San Jose  
**Session:** Demo Theater, Tuesday October 20, 4:10 PM PDT  
**Format:** 10-minute live demo  
**Repo:** https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## What This Demo Does

Llama 3.1 70B Instruct runs on vLLM in three deployments on NVIDIA H200s (BF16 on two GPUs, INT4 AWQ on one, and speculative decoding with an 8B draft model), and the dashboard shows the numbers measured on them. Next to that runs the Quantization Arena, where five birds are flown by the same small network I trained in PyTorch, each one storing its weights at a different precision. People in the room scan a QR code, back a bird, and throw hard prompts at the course from their phones, and those land on the big screen as narrow red gaps.

What the audience sees, backed by data: INT4 runs at 94% of BF16's speed on half the GPUs, which works out to 1.87 times the tokens per GPU. In the arena, plain 4-bit rounding fails on hard gaps while activation-aware scaling (the idea behind AWQ) clears every one, and the speculative decoding bird flies exactly the BF16 path.

## How It Works Under the Hood

The backend is FastAPI (`app/`). In live mode it sends fixed prompts to the three vLLM endpoints. In simulated mode it replays latencies sampled from the H200 benchmark, sleeping for the real time so concurrency builds the same way. Websockets push metrics to every screen twice a second.

The presenter page (`/presenter`) has three scenes. **Arena** runs the game in the browser from weights exported by `arena/train_policy.py`, **Numbers** shows the benchmark plus live or simulated traffic, and **Quality** shows the same prompt answered by every variant (labeled illustrative until real captures are saved to `quality/`). The audience page (`/`) is where phones back a bird, throw hard prompts, and send requests to the real models.

The full method, the numbers, and the answers to the questions experts ask are in the README.

## The Talk Flow (10 Minutes)

<!-- TALK FLOW: owned by the slides session; keep in sync with the speaker notes in slides.html -->

The talk flow and speaker script live in the speaker notes of `slides.html`.

## Presenter Shortcuts

| Key | What it does |
|---|---|
| `1` `2` `3` | Arena, Numbers, Quality scenes |
| `Space` | Fly your own bird in the arena |
| `H` | Throw a hard prompt onto the course |
| `P` | Pause the arena |
| `R` | New course |
| `T` | Light or dark theme |
| `F` | Full screen |
| `Ctrl` `Shift` `S` | Start or stop background traffic |
| `Ctrl` `Shift` `Q` | Jump to the Quality scene |

`Ctrl+Shift+S` no longer switches to simulated mode. To switch while presenting, open `/presenter?mode=sim`.

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
3. **Oct 19:** Final testing on production URL. Generate QR code. Upload slides to Sessionize (deadline).
4. **Oct 20:** Demo day. 4:10 PM PDT, Demo Theater. Be set up and tested 30 minutes before.
5. **Oct 21:** Conference day 2, booth availability.

### Pre-Demo Checklist (Day Before)

- All model variants deployed and answering (`/health` on the demo app, `/v1/models` on each vLLM endpoint)
- Demo app deployed behind an OpenShift Route with `PRESENTER_KEY` and `PUBLIC_URL` set
- Presenter laptop opened `/presenter?key=<value>` once, so the control buttons work
- QR code on the Arena sidebar scanned from a phone on cellular data, not venue wifi
- Presenter dashboard checked full screen on the demo laptop in both themes
- `/presenter?mode=sim` tested as the fallback
- Quality captures saved to `quality/` (otherwise the Quality scene says illustrative)
- Backup video recorded and on a USB drive

### 30 Minutes Before

- Open the presenter dashboard full screen (`F`)
- Check the mode badge in the top right says what you expect (Live models or Simulated)
- Press Reset, then Start traffic, and watch the Numbers scene fill in
- Throw one hard prompt from your phone and watch it land in the arena
- Have the QR code visible on the Arena scene
- Silence notifications and close other apps

## Backup Plans

**If the models stop responding:** open `/presenter?mode=sim` in the same tab. The Numbers scene keeps showing the H200 benchmark, the right-now panel switches to replayed timings with a "Simulated" label, and the arena doesn't depend on the models at all, so it keeps running exactly as before.

**If the venue network is bad:** run the app on the demo laptop with `./scripts/run-local.sh`. Fonts and everything else are bundled, so the presenter screen renders correctly with no network. Phones need to reach the laptop, so this only covers the big screen.

**If the room is small:** press `H` to throw hard prompts yourself and `Space` to fly a bird. The arena tells the whole quantization story with nobody on their phones.

**If everything fails:** play the backup video from the USB drive, narrate over it, and move to Q&A.

## Equipment

- Laptop with presenter dashboard open
- HDMI adapter (and a backup one)
- Phone or tablet with QR code displayed
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
