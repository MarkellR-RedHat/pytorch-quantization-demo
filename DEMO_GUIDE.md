# Demo Execution Guide

Step-by-step guide for presenting "Quantization Showdown: PyTorch Inference Optimization" at PyTorch Conference 2026.

**Session:** Demo Theater, Tuesday Oct 20 at 4:10 PM PDT (10 minutes)

## Pre-Demo Checklist

### Day Before (Oct 19)

- [ ] All model variants deployed and answering on OpenShift AI
- [ ] Demo app deployed with `PRESENTER_KEY` and `PUBLIC_URL` set
- [ ] Presenter laptop opened `/presenter?key=<value>` once
- [ ] QR code scanned from a phone on cellular data
- [ ] Presenter dashboard checked full screen on the demo laptop
- [ ] `/presenter?mode=sim` tested as the fallback
- [ ] Quality captures saved to `quality/`
- [ ] Backup video recorded

### 30 Minutes Before

- [ ] Presenter dashboard open full screen: `{your-url}/presenter`
- [ ] Mode badge shows what you expect
- [ ] Reset, then Start traffic, and the Numbers scene fills in
- [ ] One hard prompt thrown from a phone lands in the arena
- [ ] Notifications silenced, other apps closed

## Demo Flow (10 Minutes)

<!-- TALK FLOW: owned by the slides session; keep in sync with the speaker notes in slides.html -->

| Time | Slide | What happens |
|---|---|---|
| 0:00 | 1 Title | Intro the cast (Priya, Marcus, Dana), and tell people to keep their phones handy |
| 0:35 | 2 Meet Priya | Llama 3.1 70B at BF16 fills an H200, 2 GPUs per replica, and Dana's "quick question" about 1,460 GPU-hours a month |
| 1:20 | 3 The obvious fix | Size chart: BF16 141 GB, FP8 71 GB, INT4 about 38 GB, so INT4 fits on one GPU |
| 1:55 | 4 The ticket | Marcus's ticket, a show of hands ("who hoped the quality was fine?"), and Priya sets up a showdown |
| 2:35 | 5 The showdown | The three contestants, the spec-decode token animation, and a hand vote on who's fastest |
| 3:35 | 6 Let's find out | Switch to the dashboard. Arena (1) with the QR on the sidebar, phones join, then Numbers (2) and Quality (3) |
| 6:05 | 7 Plot twist | Benchmark charts, with a payoff for each group of voters |
| 6:55 | 8 Sheep test | The room shouts an answer, then one click runs the sheep off and reveals the three answers |
| 7:50 | 9 What Priya shipped | Routing: chat to INT4, deep analysis to BF16, low-traffic latency work to spec decode. Marcus closes the ticket |
| 8:35 | 10 Lightning round | Six questions on one slide: FP8, why spec decode lost, 20 requests, 8B models, AWQ vs GPTQ/torchao, routing |
| 9:25 | 11 Close | Everything in one place: repo QR, sim mode on a laptop, the stack, the booth |

Slide keys: arrows or clicker to move, N for speaker notes, B for blackout, F for fullscreen. On slide 8, the first click is the reveal.

The full speaker script is embedded at the bottom of slides.html (press N while presenting).

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

## Backup Plans

**If the models stop responding:** open `/presenter?mode=sim` in the same tab. The Numbers scene keeps showing the H200 benchmark, the right-now panel switches to replayed timings with a "Simulated" label, and the arena doesn't depend on the models at all, so it keeps running exactly as before.

**If the venue network is bad:** run the app on the demo laptop with `./scripts/run-local.sh`. Fonts and everything else are bundled, so the presenter screen renders correctly with no network. Phones need to reach the laptop, so this only covers the big screen.

**If the room is small:** press `H` to throw hard prompts yourself and `Space` to fly a bird. The arena tells the whole quantization story with nobody on their phones.

**If everything fails:** play the backup video from the USB drive, narrate over it, and move to Q&A.

## Equipment

- Laptop with presenter dashboard open
- HDMI adapter
- Phone/tablet with QR code displayed
- Backup laptop with full setup
- USB drive with backup video
