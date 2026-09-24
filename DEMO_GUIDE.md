# Demo Execution Guide

Step-by-step guide for presenting "Quantization Showdown: PyTorch Inference Optimization" at PyTorch Conference 2026.

**Session:** Demo Theater, Tuesday Oct 20 at 4:10 PM PDT (10 minutes)

## Pre-Demo Checklist

### Day Before (Oct 19)

- [ ] All 3 model variants deployed and responding on OpenShift AI
- [ ] Demo app deployed with production URL
- [ ] QR code generated with public URL
- [ ] Presenter dashboard tested on demo laptop
- [ ] Audience interface tested on mobile device
- [ ] Simulation mode tested and ready as fallback
- [ ] Backup video recorded
- [ ] Quality comparison scenarios all loading correctly

### 30 Minutes Before

- [ ] Open presenter dashboard in full screen: `{your-url}/presenter`
- [ ] Verify all 3 model endpoints responding (`/health`)
- [ ] Test WebSocket connections (metrics updating)
- [ ] Reset demo metrics to zero
- [ ] Test simulation toggle (Ctrl+Shift+S)
- [ ] Have QR code ready on separate device
- [ ] Silence notifications, close other apps

## Demo Flow (10 Minutes)

### [0:00-1:30] The Story

> "So your team just shipped Llama 70B to production. Full precision, FP16, everyone's proud. First week goes great. Then traffic doubles. Then triples. Your P95 latency goes through the roof. Your GPU bill hits $47K in one month. Your manager walks over with a screenshot of the invoice and asks 'what happened?'"
>
> "So you do what everyone does. You quantize. INT4, ship it, latency drops, costs drop, everyone's happy again. Until a customer opens a support ticket: 'Your model used to give me detailed analysis, now it just gives me bullet points.' You check the logs. The INT4 model is struggling on complex reasoning. You didn't lose speed. You lost trust."
>
> "That's the trade-off everyone tells you is unavoidable. Speed or quality. Pick one."

*Point to the dashboard, already live with background traffic.*

> "Let's see if that's actually true."

### [1:30-4:30] The Trade-off is Real

Walk through FP16 vs INT4 under load. Point to specific numbers:

**Latency:**
> "FP16 is sitting at around 95ms per request. INT4? 45ms. More than 2x faster."

**GPU Memory:**
> "FP16 is using 40GB of GPU memory. INT4 only needs 10GB. That's your $47K bill right there."

**Cost:**
> "Cost per request on FP16 is 3x higher than INT4."

Then toggle quality comparison (Ctrl+Shift+Q):

> "But look at this. Same reasoning prompt sent to both. FP16 gets it right. INT4 misses the trick question entirely. That's the support ticket."

### [4:30-7:30] The Solve

> "After the support ticket incident, your team tries something different. Instead of just compressing the model, you pair it with a small draft model that speculates tokens ahead of time. The full model just has to verify, not generate from scratch."

Point to the Speculative Decode column:

> "Look at the latency. 55ms. That's approaching INT4 speed."

Toggle quality comparison again:

> "And the output? Matches FP16. Same reasoning, same depth, same accuracy. You didn't pick speed or quality. You got both."

> "The draft model costs almost nothing to run. Your target model does less work per token. GPU bill drops. Customers stop complaining."

### [7:30-8:30] The Stack

> "All three variants running simultaneously on vLLM. Quantization done with LLM Compressor. Speculative decoding is native to vLLM, just a flag. No custom inference code."

### [8:30-9:30] Audience Pile-On

> "Want to stress test it yourself? Scan this QR code and start sending requests. Let's see which one breaks first."

Show QR code. If people join, point at metrics spiking. If not, you already delivered the demo.

### [9:30-10:00] Close

> "The quantization trade-off is real. But it's not the only option anymore. Come to the booth if you want to try this on your own models."

> "And if you want to go deeper on vLLM and llm-d, Sasa is running a meetup at [TIME] in [LOCATION]."

## Presenter Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+Shift+S | Toggle simulation mode |
| Ctrl+Shift+Q | Toggle quality comparison panel |

## Backup Plans

### If models don't respond
Switch to simulation mode (Ctrl+Shift+S). Continue narration. Simulation uses realistic baseline metrics.

### If audience is small
Skip the pile-on section. Focus on the story and quality comparison.

### If complete failure
Play backup video. Narrate over it. Turn into Q&A.

## Equipment

- Laptop with presenter dashboard open
- HDMI adapter
- Phone/tablet with QR code displayed
- Backup laptop with full setup
- USB drive with backup video
