# Demo Execution Guide

Step-by-step guide for presenting "Quantization Showdown: PyTorch Inference Optimization" at PyTorch Conference 2026.

**Session:** Demo Theater, Tuesday Oct 20 at 4:10 PM PDT (10 minutes)

## Pre-Demo Checklist

### Day Before (Oct 19)

- [ ] All model variants deployed and answering on OpenShift AI
- [ ] Demo app deployed with `PRESENTER_KEY` set
- [ ] Presenter laptop opened `/presenter?key=<value>` once
- [ ] One typed question and all three presets answered live
- [ ] `/presenter?mode=sim` tested as the fallback
- [ ] Backup video recorded

### 30 Minutes Before

- [ ] Presenter dashboard open full screen: `{your-url}/presenter`
- [ ] Badge in the top right says Live models
- [ ] One warm-up question asked
- [ ] Notifications silenced, other apps closed

## Demo Flow (10 Minutes)

<!-- TALK FLOW: owned by the slides session; keep in sync with the speaker notes in slides.html -->

The talk flow and the full speaker script live in the speaker notes of `slides.html` (press `N` while presenting). The demo slide calls for a question from the room: press `1` for Ask, type the question, and press `Enter`, then press `2` for Numbers, the money slide.

## Presenter Shortcuts

| Key | What it does |
|---|---|
| `1` `2` | Ask, Numbers |
| `/` | Jump to the question box |
| `Enter` | Send the question to every setup |
| `T` | Light or dark theme |
| `F` | Full screen |

To force replay mode while presenting, open `/presenter?mode=sim`.

## Backup Plans

**If the models stop responding:** open `/presenter?mode=sim` in the same tab. The Numbers scene doesn't change, because it shows the measured benchmark. In Ask, a "Replay" badge appears and each column plays back the speed that setup measured, and preset questions still show a full answer.

**If the venue network is bad:** run the app on the demo laptop with `./scripts/run-local.sh`. Fonts and everything else are bundled, so the screen renders correctly with no network.

**If nobody calls out a question:** use the preset buttons under the question box.

**If everything fails:** play the backup video from the USB drive, narrate over it, and move to Q&A.

## Equipment

- Laptop with presenter dashboard open
- HDMI adapter
- Backup laptop with full setup
- USB drive with backup video
