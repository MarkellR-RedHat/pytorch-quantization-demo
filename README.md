# Quantization Showdown: PyTorch Inference Optimization

This is a live, interactive demo that lets an audience stress test three different quantization variants of the same model side by side. Everyone in the room scans a QR code, picks a variant (FP16, INT4, or Speculative Decode), and starts hammering the model with requests. A presenter dashboard shows the real-time metrics updating as traffic flows in, so you can actually see the latency, GPU memory, throughput, cost, and quality trade-offs happening right in front of you.

Originally presented at **PyTorch Conference North America 2026** in San Jose.

## Try It Yourself

You don't need GPUs, a cluster, or any external services to run this. Simulation mode generates realistic synthetic metrics so you can see exactly how the demo works locally.

```bash
git clone https://github.com/MarkellR-RedHat/pytorch-quantization-demo.git
cd pytorch-quantization-demo
./scripts/setup.sh
./scripts/run-local.sh
```

That's it. Open http://localhost:8000 for the audience view and http://localhost:8000/presenter for the presenter dashboard.

The presenter dashboard has a few keyboard shortcuts: `Ctrl+Shift+S` toggles simulation mode on and off, and `Ctrl+Shift+Q` opens the quality comparison panel where you can see the same prompt answered by all three variants.

## What's Being Compared

The demo runs three variants of Llama simultaneously on vLLM:

**FP16 (Half Precision)** is the quality baseline. Full model weights at 16-bit floating point. Best output quality, but it uses the most GPU memory and has the highest latency.

**INT4 (4-bit Quantized)** is the speed play. Quantized down to 4 bits using LLM Compressor, so it runs faster and uses way less memory. The trade-off is that output quality drops on complex reasoning tasks.

**Speculative Decode** is the interesting one. It pairs a small draft model (like Llama 8B) with the full-size target model. The draft model generates candidate tokens quickly, and the target model verifies them in parallel. You end up with latency close to INT4 but output quality that matches FP16.

## Running With Real Models

When you're ready to connect to actual model endpoints instead of simulation mode, update your `.env` file:

```bash
cp .env.example .env
```

Then fill in your three vLLM model endpoints:

| Variable | What it points to |
|----------|------------------|
| `MODEL_FP16_ENDPOINT` | vLLM serving Llama 70B at FP16 |
| `MODEL_INT4_ENDPOINT` | vLLM serving Llama 70B quantized to INT4 (via LLM Compressor) |
| `MODEL_SPEC_DECODE_ENDPOINT` | vLLM serving Llama 70B with a draft model for speculative decoding |

Set `SIMULATION_MODE=false` and start the app:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For the speculative decode endpoint, vLLM handles this natively with the `--speculative-model` flag:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --speculative-model meta-llama/Llama-3.3-8B-Instruct \
  --num-speculative-tokens 5
```

## Smaller GPU Option

If you don't have access to H200s or large GPUs, you can run this with **Llama 3.3 8B** instead of 70B. The demo works exactly the same way, you just swap the model endpoints in your `.env` file. Llama 8B fits comfortably on a single consumer GPU (A10, L4, T4, or even a 3090).

For speculative decoding with the 8B model, use something like TinyLlama 1.1B or Llama 3.2 1B as the draft model. Check the `.env.example` file for example endpoint configs.

The audience experience and presenter dashboard are identical regardless of model size. The absolute numbers change (smaller model, lower latency across the board) but the relative trade-offs between variants are the same, which is the whole point.

## Deploying to OpenShift

```bash
podman build -t pytorch-quantization-demo .
podman tag pytorch-quantization-demo quay.io/your-org/pytorch-quantization-demo:latest
podman push quay.io/your-org/pytorch-quantization-demo:latest
oc apply -f kubernetes/configmap.yaml
oc apply -f kubernetes/deployment.yaml
oc apply -f kubernetes/service.yaml
```

## Project Structure

```
pytorch-quantization-demo/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Settings and model variant config
│   ├── models.py            # Pydantic data models
│   ├── simulation.py        # Simulation mode with quality comparisons
│   ├── metrics.py           # Real-time metrics aggregation
│   ├── openshift.py         # OpenShift AI client
│   └── websocket.py         # WebSocket connection manager
├── static/
│   ├── css/style.css
│   └── js/
│       ├── audience.js      # Mobile audience interface
│       └── presenter.js     # Presenter dashboard
├── templates/
│   ├── index.html           # Audience view
│   └── presenter.html       # Presenter view
├── kubernetes/              # OpenShift/K8s manifests
├── scripts/
│   ├── setup.sh             # One-time environment setup
│   ├── run-local.sh         # Start in simulation mode
│   └── test.sh              # Run tests with coverage
├── tests/                   # Test suite
├── Dockerfile
├── .env.example
└── requirements.txt
```

## API Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/` | GET | Audience interface (mobile-optimized) |
| `/presenter` | GET | Presenter dashboard with metrics grid |
| `/request` | POST | Submit an inference request |
| `/quality/{scenario}` | GET | Quality comparison for a given scenario |
| `/health` | GET | Health check |
| `/ws` | WS | WebSocket for live metric updates |

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENSHIFT_AI_ENDPOINT` | OpenShift AI API endpoint | Yes (or use simulation mode) |
| `OPENSHIFT_AI_TOKEN` | Authentication token | Yes (or use simulation mode) |
| `MODEL_FP16_ENDPOINT` | FP16 model serving endpoint | Yes (or use simulation mode) |
| `MODEL_INT4_ENDPOINT` | INT4 model serving endpoint | Yes (or use simulation mode) |
| `MODEL_SPEC_DECODE_ENDPOINT` | Speculative decode model endpoint | Yes (or use simulation mode) |
| `SIMULATION_MODE` | Run with synthetic data (true/false) | No (defaults to false) |
| `PORT` | Server port | No (defaults to 8000) |

## Tests

```bash
source venv/bin/activate
pytest tests/
```

## Docs

- [Demo Execution Guide](DEMO_GUIDE.md) for the full presenter script with timing
- [Development Guide](DEVELOPMENT.md) for local development setup
- [Contributing](CONTRIBUTING.md)

## License

MIT

## Author

**Markell Rawls**
Technical Marketing Engineer, Red Hat
mrawls@redhat.com
