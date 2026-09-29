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
