# Project Summary: PyTorch Quantization Demo

## What Was Built

An interactive demo for PyTorch Conference 2026 (Demo Theater, Oct 20) that compares three PyTorch inference optimization strategies in real-time: FP16 (quality baseline), INT4 (speed play), and Speculative Decoding (best of both).

## Repository

**GitHub**: https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## The Three Variants

| Variant | Role | Latency | GPU Memory | Cost | Quality |
|---------|------|---------|------------|------|---------|
| **FP16** | Quality baseline | ~95ms | ~40GB | $0.0015/1k tokens | Full quality |
| **INT4** | Speed champion | ~45ms | ~10GB | $0.0005/1k tokens | Degrades on complex reasoning |
| **Spec Decode** | Best of both | ~55ms | ~25GB | $0.0007/1k tokens | Matches FP16 |

Speculative decoding pairs a small draft model (Llama 8B) with the target model (Llama 70B INT8). The draft model speculates tokens, the target model verifies. Result: near-INT4 speed with FP16-level quality.

## Key Components

### Backend (FastAPI)

**Files**: `app/main.py`, `app/config.py`, `app/models.py`, `app/openshift.py`, `app/simulation.py`, `app/metrics.py`, `app/websocket.py`

- RESTful API for inference requests across 3 variants
- WebSocket for real-time metric updates (2x/sec)
- Quality comparison endpoint with pre-scripted scenarios (reasoning, code gen, summarization)
- Simulation mode for failsafe demo execution
- OpenShift AI integration for vLLM model serving

### Frontend

**Audience Interface** (`templates/index.html`, `static/js/audience.js`):
- Mobile-optimized, 3 model buttons, rapid-fire tap to send
- QR code access for easy mobile connection

**Presenter Dashboard** (`templates/presenter.html`, `static/js/presenter.js`):
- 3-column metrics grid with hero stats (latency + GPU memory)
- Quality comparison panel (Ctrl+Shift+Q) showing side-by-side outputs
- Simulation toggle (Ctrl+Shift+S)
- Dark theme optimized for projection

### Infrastructure

- Docker (Red Hat UBI, non-root, health checks)
- Kubernetes/OpenShift manifests (deployment, service, route, configmap, secrets)
- CI/CD via GitHub Actions

## Demo Flow (10 minutes)

1. **The Story** (0:00-1:30): "The 3 AM GPU bill" — team ships FP16, costs spike, they quantize to INT4, quality drops
2. **The Trade-off** (1:30-4:30): Live FP16 vs INT4 metrics + quality comparison
3. **The Solve** (4:30-7:30): Speculative decode — INT4 speed, FP16 quality
4. **The Stack** (7:30-8:30): vLLM + LLM Compressor
5. **Audience Pile-On** (8:30-9:30): QR code, optional stress test
6. **Close** (9:30-10:00): Booth plug, vLLM meetup

## GPU Requirements

| Dates | GPUs | Purpose |
|-------|------|---------|
| Sep 29-30 | 5x H200 Full | Test run + backup video recording |
| Oct 18-21 | 5x H200 Full | Setup + live conference |

- FP16 Llama 70B: 2x H200 (tensor parallel)
- INT4 Llama 70B: 1x H200
- Spec Decode (Llama 70B INT8 + Llama 8B draft): 2x H200

## Current Status

**Complete**:
- Backend with 3-variant architecture
- Presenter dashboard with quality comparison
- Audience interface
- Simulation mode with realistic baselines
- WebSocket real-time updates
- Docker/Kubernetes deployment configs
- Test suite
- CI/CD pipelines
- Story-driven demo guide

**Pending**:
- Deploy real models to OpenShift AI (test run Sep 29-30)
- Production deployment URL
- QR code with final public URL
- Record backup demo video (Sep 29-30)
- Upload slides to Sessionize (due Oct 19)

## Timeline

1. **Sep 29-30**: Test run on H200s, deploy models, record backup video
2. **Oct 18**: Pre-deploy to OpenShift
3. **Oct 19**: Final testing, slide upload deadline
4. **Oct 20**: Demo day — 4:10 PM PDT, Demo Theater
5. **Oct 21**: Conference day 2

## Contacts

- **Event**: Juliana Furlow (jsweek@redhat.com)
- **vLLM/llm-d**: Sasa
- **GitHub**: https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## License

MIT License
