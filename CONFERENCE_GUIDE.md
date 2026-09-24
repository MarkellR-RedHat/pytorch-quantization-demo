# PyTorch Conference NA 2026: Quantization Showdown

**Conference:** PyTorch Conference North America 2026, San Jose  
**Session:** Demo Theater, Tuesday October 20, 4:10 PM PDT  
**Format:** 10-minute live demo  
**Repo:** https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## What This Demo Does

Three versions of the same model (Llama 70B) running side by side on vLLM, and the audience gets to beat on all three of them in real time. Everyone in the room scans a QR code on their phone, picks a variant, and starts firing requests. A presenter dashboard on the big screen shows what's happening to latency, GPU memory, throughput, and cost as traffic flows in.

The three variants are FP16 (full precision, the quality baseline), INT4 (quantized down to 4 bits for speed), and Speculative Decode (pairs a small draft model with the full model to get INT4-like speed without losing FP16-level quality). The whole point is showing that the "speed vs quality" trade-off everyone accepts as unavoidable actually has a third option most teams haven't tried yet.

The demo runs on FastAPI with WebSocket connections pushing metric updates to the presenter view twice a second. There's a simulation mode built in that generates realistic synthetic data, so the whole thing works locally with zero GPU access. That's also the backup plan if anything goes wrong day-of.

## How It Works Under the Hood

The backend sits in `app/` and does a few things. `main.py` is the FastAPI app with routes for inference requests and a quality comparison endpoint. `simulation.py` handles the synthetic data path with pre-built scenarios covering reasoning tasks, code generation, and summarization. `metrics.py` aggregates everything coming in from the audience and computes per-variant stats. `websocket.py` manages the real-time connections to the presenter dashboard.

The frontend has two views. The audience interface (`templates/index.html` + `static/js/audience.js`) is mobile-optimized with three big buttons, one per variant. The presenter dashboard (`templates/presenter.html` + `static/js/presenter.js`) is a 3-column metrics grid with hero stats for latency and GPU memory, plus a quality comparison panel you can toggle with `Ctrl+Shift+Q`. Dark theme, optimized for projector screens.

For the live version, each variant points at a separate vLLM endpoint running on OpenShift AI. FP16 needs 2x H200 GPUs with tensor parallelism, INT4 fits on a single H200, and Speculative Decode uses 2x H200 (one for the target model at INT8, one shared with the Llama 8B draft model). Quantization is handled by LLM Compressor, and speculative decoding is native to vLLM with just a `--speculative-model` flag.

## The Talk Flow (10 Minutes)

**[0:00 to 1:30] The Story.** Open with the scenario everyone's lived through. Your team ships Llama 70B at FP16, everything's great for a week, then traffic doubles and your GPU bill hits $47K in a month. So you quantize to INT4, costs drop, latency drops, everyone's happy. Until a customer files a support ticket because the model stopped giving detailed analysis on complex prompts. You didn't lose speed, you lost trust.

**[1:30 to 4:30] The Trade-off is Real.** Walk through the live metrics on the presenter dashboard. FP16 sits around 95ms per request, INT4 is at 45ms, more than 2x faster. FP16 eats 40GB of GPU memory, INT4 only needs 10GB. Then toggle the quality comparison panel (`Ctrl+Shift+Q`) and show the same reasoning prompt answered by both. FP16 nails it. INT4 misses the trick question. That's the support ticket.

**[4:30 to 7:30] The Solve.** Introduce speculative decoding. Instead of just compressing the model, you pair it with a small draft model that speculates tokens ahead of time. The full model just verifies instead of generating from scratch. Point at the Speculative Decode column on the dashboard: 55ms latency (approaching INT4 speed), and when you toggle quality comparison again, the output matches FP16. No quality loss. The draft model costs almost nothing to run.

**[7:30 to 8:30] The Stack.** Quick callout: all three variants running simultaneously on vLLM, quantization done with LLM Compressor, speculative decoding is native to vLLM with just a flag, no custom inference code required.

**[8:30 to 9:30] Audience Pile-On.** Show the QR code, invite people to scan and start hammering the models. If people join, point at the metrics spiking on screen. If the room is small or nobody bites, that's fine because you already delivered the core demo.

**[9:30 to 10:00] Close.** "The quantization trade-off is real, but it's not the only option anymore." CTA to visit the booth to try it on their own models, and plug Sasa's vLLM meetup if there is one.

## Presenter Shortcuts

| Shortcut | What it does |
|----------|-------------|
| `Ctrl+Shift+S` | Toggle simulation mode on/off |
| `Ctrl+Shift+Q` | Toggle quality comparison panel |

## The Plan

### GPU Booking

| Dates | GPUs | Purpose |
|-------|------|---------|
| Sep 29-30 | 5x H200 Full | Test run, end-to-end validation, record backup video |
| Oct 18-21 | 5x H200 Full | Pre-deploy (Oct 18), final testing (Oct 19), live conference (Oct 20-21) |

GPU breakdown per variant:
- FP16 Llama 70B: 2x H200 (tensor parallel)
- INT4 Llama 70B: 1x H200
- Speculative Decode (Llama 70B INT8 + Llama 8B draft): 2x H200

### Timeline

1. **Sep 29-30:** Test run on H200s. Deploy all three model variants, run through the full demo flow, record backup video.
2. **Oct 18:** Pre-deploy the demo app and models to OpenShift AI.
3. **Oct 19:** Final testing on production URL. Generate QR code. Upload slides to Sessionize (deadline).
4. **Oct 20:** Demo day. 4:10 PM PDT, Demo Theater. Be set up and tested 30 minutes before.
5. **Oct 21:** Conference day 2, booth availability.

### Pre-Demo Checklist (Day Before)

- All 3 model variants deployed and responding
- Demo app deployed with production URL
- QR code generated and printed/on a device
- Presenter dashboard tested on demo laptop
- Audience interface tested on a phone
- Simulation mode tested as fallback
- Backup video recorded and on a USB drive
- Quality comparison scenarios all loading

### 30 Minutes Before

- Open presenter dashboard full screen
- Verify all 3 model endpoints responding (`/health`)
- Test WebSocket connections (metrics should be updating)
- Reset demo metrics to zero
- Test simulation toggle
- Have QR code ready on a separate device
- Silence notifications, close other apps

## Backup Plans

**If models don't respond:** Toggle to simulation mode with `Ctrl+Shift+S`. The simulation uses realistic baseline metrics pulled from actual H200 benchmarks, so the demo narrative works exactly the same. Keep presenting, nobody in the audience will know the difference.

**If the audience is small:** Skip the pile-on section entirely. The core demo (story, trade-off, solve) stands on its own without audience participation. The pile-on is a bonus, not a requirement.

**If complete failure (app down, projector issues, etc.):** Play the backup video from the USB drive. Narrate over it. Turn the remaining time into Q&A. Having the backup video recorded during the Sep 29-30 test run is specifically for this scenario.

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

Open http://localhost:8000 for the audience view and http://localhost:8000/presenter for the presenter dashboard.

## Contacts

- **Event logistics:** Juliana Furlow (jsweek@redhat.com)
- **vLLM / llm-d:** Sasa
- **Repo:** https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## Author

**Markell Rawls**  
Technical Marketing Engineer, Red Hat  
mrawls@redhat.com
