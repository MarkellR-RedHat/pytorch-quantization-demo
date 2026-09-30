# Demo Execution Guide

Step-by-step guide for presenting "Not Every Question Needs Two GPUs" at PyTorch Conference 2026.

**Session:** Demo Theater, Tuesday Oct 20 at 4:10 PM PDT (10 minutes)

## Pre-Demo Checklist

### Day Before (Oct 19)

- [ ] All model variants deployed and answering on OpenShift AI
- [ ] Demo app running on the presenter laptop: `./scripts/setup.sh`, a `.env` with the endpoints and `PRESENTER_KEY`, then `source venv/bin/activate && python -m uvicorn app.main:app --port 8000`
- [ ] Laptop opened `http://localhost:8000/presenter?key=<value>` once
- [ ] One typed question and all eight presets answered live
- [ ] Backup deployed behind an OpenShift Route with the same `.env` values
- [ ] `/presenter?mode=sim` tested as the fallback
- [ ] Backup video recorded

### 30 Minutes Before

- [ ] `python scripts/preflight.py` passes
- [ ] Port-forwards (or VPN) up, app started on the laptop, `http://localhost:8000/presenter` open full screen
- [ ] Badge in the top right says Live models
- [ ] One warm-up question asked
- [ ] Notifications silenced, other apps closed

## Demo Flow (10 Minutes)

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

## Backup Plans

**If one model stops responding:** nothing to do for the presets. After 8 seconds with no first token, or 10 seconds of silence mid-answer, that column plays its recorded answer under a red line that says "Recorded <date> · live request failed", and its badge says RECORDED. Mention it when it happens.

**If all the models stop responding:** press `R` (or open `/presenter?mode=sim`). The Numbers scene doesn't change, because it shows the measured benchmark. In Ask, a "Replay" badge appears and each column plays its recorded answer at the speed that setup measured. Press `R` again to go back to live.

**If the laptop app breaks:** open the backup, the OpenShift Route (`https://<route>/presenter?key=<value>`), in the same browser. It runs the same code with the same recordings. If the venue network is bad, stay on the laptop: fonts and everything else are bundled, and `R` plays the recordings.

**If nobody calls out a question:** use the eight preset buttons under the question box. Sheep riddle and Logic puzzle go to BF16, the four everyday ones go to INT4, and Python function and Explain KV cache go to Spec Decode. Explain KV cache is the long-answer, latency case, where Spec Decode's speed shows most.

**If everything fails:** play the backup video from the USB drive, narrate over it, and move to Q&A.

## Equipment

- Laptop with presenter dashboard open
- HDMI adapter
- Backup laptop with full setup
- USB drive with backup video
