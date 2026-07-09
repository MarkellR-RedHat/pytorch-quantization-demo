# Project Summary: PyTorch Quantization Demo

## What Was Built

A complete, production-ready interactive demo application for PyTorch Conference 2026 that allows audience members to stress test different quantization levels of PyTorch models in real-time.

## Repository

**GitHub**: https://github.com/MarkellR-RedHat/pytorch-quantization-demo

## Key Components

### 1. Backend (FastAPI)

**Files**: `app/main.py`, `app/config.py`, `app/models.py`, `app/openshift.py`, `app/simulation.py`, `app/metrics.py`, `app/websocket.py`

Features:
- RESTful API endpoints for inference requests
- WebSocket support for real-time metric updates
- OpenShift AI integration for model serving
- Simulation mode for failsafe demo execution
- Comprehensive metrics collection and aggregation
- Support for 4 quantization variants (FP32, FP16, INT8, INT4)

### 2. Frontend (HTML/CSS/JavaScript)

**Audience Interface** (`templates/index.html`, `static/js/audience.js`):
- Mobile-optimized model selection interface
- Large "Send Request" button for easy tapping
- Real-time stats display (your requests, total requests, participants)
- QR code access for easy mobile connection

**Presenter Dashboard** (`templates/presenter.html`, `static/js/presenter.js`):
- 4-way metrics comparison grid
- Real-time updates via WebSocket
- Simulation mode toggle (Ctrl+Shift+S)
- Demo reset functionality
- Professional dark theme optimized for presentation

**Styling** (`static/css/style.css`):
- Red Hat brand colors
- Mobile-responsive design
- Clean, modern UI
- Separate themes for audience and presenter views

### 3. Infrastructure

**Docker** (`Dockerfile`):
- Multi-stage build
- Red Hat UBI base image
- Non-root user for security
- Health checks included

**Kubernetes/OpenShift** (`kubernetes/`):
- Deployment manifest with proper resource limits
- Service definition
- Route configuration for external access
- ConfigMap for model endpoints
- Secret template for credentials

### 4. Testing

**Test Suite** (`tests/`):
- Simulation mode tests
- Metrics collection tests
- Async/await support
- Code coverage reporting

### 5. Documentation

- **README.md**: Project overview, quick start, API docs
- **DEVELOPMENT.md**: Complete development guide, local setup, troubleshooting
- **DEMO_GUIDE.md**: Step-by-step presenter guide with full script and timing
- **CONTRIBUTING.md**: Contribution guidelines
- **LICENSE**: MIT License

### 6. Automation

**Helper Scripts** (`scripts/`):
- `setup.sh`: Automated environment setup
- `run-local.sh`: Quick start for local development
- `test.sh`: Run tests with coverage

**CI/CD** (`.github/workflows/`):
- Continuous integration (test, lint, build)
- Container image publishing to Quay.io
- Multi-version Python testing (3.9, 3.10, 3.11)

## Features

### Core Functionality

1. **Multi-Model Serving**: Simultaneously serve 4 quantization variants
2. **Real-Time Metrics**: Live updates of latency, throughput, cost, GPU memory
3. **Interactive Audience Participation**: QR code access, mobile-optimized
4. **Simulation Mode**: Failsafe backup with synthetic data
5. **WebSocket Communication**: Real-time bidirectional updates
6. **Presenter Controls**: Hidden shortcuts for demo management

### Production-Ready Features

1. **Health Checks**: `/health` endpoint for monitoring
2. **Structured Logging**: Configurable log levels
3. **Error Handling**: Comprehensive exception handling
4. **Resource Limits**: Kubernetes resource requests/limits
5. **Security**: Non-root container, secret management
6. **Scalability**: Horizontal scaling support with replicas

### Demo-Specific Features

1. **QR Code Generation**: Automatic QR code for audience access
2. **Metrics Aggregation**: Rolling windows, P95 latency, cost tracking
3. **Visual Comparison**: Side-by-side metrics for all 4 variants
4. **Simulation Toggle**: Seamless failover with Ctrl+Shift+S
5. **Reset Functionality**: Clear all metrics and start fresh

## Technology Stack

- **Backend**: Python 3.9+, FastAPI, Uvicorn
- **Frontend**: Vanilla JavaScript, WebSocket API
- **Deployment**: Docker/Podman, Kubernetes/OpenShift
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **CI/CD**: GitHub Actions
- **Model Serving**: OpenShift AI with vLLM

## File Structure

```
pytorch-quantization-demo/
├── .github/
│   └── workflows/           # CI/CD pipelines
├── app/                     # Backend application
│   ├── __init__.py
│   ├── main.py             # FastAPI app
│   ├── config.py           # Configuration
│   ├── models.py           # Data models
│   ├── openshift.py        # OpenShift AI client
│   ├── simulation.py       # Simulation mode
│   ├── metrics.py          # Metrics collection
│   └── websocket.py        # WebSocket manager
├── static/
│   ├── css/
│   │   └── style.css       # Styling
│   └── js/
│       ├── audience.js     # Audience interface
│       └── presenter.js    # Presenter dashboard
├── templates/
│   ├── index.html          # Audience view
│   └── presenter.html      # Presenter view
├── kubernetes/              # K8s manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── route.yaml
│   ├── configmap.yaml
│   └── secret-template.yaml
├── scripts/                 # Helper scripts
│   ├── setup.sh
│   ├── run-local.sh
│   └── test.sh
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_simulation.py
│   └── test_metrics.py
├── .env.example            # Environment template
├── .gitignore
├── CONTRIBUTING.md
├── DEMO_GUIDE.md           # Presenter guide
├── DEVELOPMENT.md          # Developer guide
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## How It Works

### Audience Flow

1. Scan QR code on mobile device
2. Opens audience interface
3. Select model variant (FP32, FP16, INT8, INT4)
4. Tap "Send Request" button repeatedly
5. See personal request count and total stats
6. WebSocket provides real-time updates

### Presenter Flow

1. Open presenter dashboard
2. Display on main screen with 4-way metrics grid
3. Show QR code for audience to scan
4. Monitor participant count
5. Narrate metrics changes as requests flood in
6. Compare performance, cost, and quality across variants
7. Toggle simulation mode if needed (Ctrl+Shift+S)

### Technical Flow

1. User taps button → POST /request
2. Backend routes to OpenShift AI model endpoint (or simulation)
3. Metrics recorded (latency, throughput, cost)
4. Aggregated metrics broadcast via WebSocket
5. All connected clients update in real-time
6. Presenter dashboard shows 4-way comparison

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENSHIFT_AI_ENDPOINT` | OpenShift AI API endpoint | Yes |
| `OPENSHIFT_AI_TOKEN` | Authentication token | Yes |
| `MODEL_FP32_ENDPOINT` | FP32 model endpoint | Yes |
| `MODEL_FP16_ENDPOINT` | FP16 model endpoint | Yes |
| `MODEL_INT8_ENDPOINT` | INT8 model endpoint | Yes |
| `MODEL_INT4_ENDPOINT` | INT4 model endpoint | Yes |
| `SIMULATION_MODE` | Enable simulation (true/false) | No |
| `PORT` | Server port (default: 8000) | No |

## Deployment Options

### Local Development

```bash
./scripts/setup.sh      # One-time setup
./scripts/run-local.sh  # Run with simulation mode
```

### OpenShift Deployment

```bash
oc apply -f kubernetes/configmap.yaml
oc apply -f kubernetes/secret-template.yaml
oc apply -f kubernetes/deployment.yaml
oc apply -f kubernetes/service.yaml
oc apply -f kubernetes/route.yaml
```

### Container Image

```bash
podman build -t pytorch-quantization-demo:latest .
podman run -p 8000:8000 --env-file .env pytorch-quantization-demo:latest
```

## Testing

```bash
./scripts/test.sh           # Run all tests with coverage
pytest tests/               # Run tests directly
pytest tests/ --cov=app     # With coverage
```

## Current Status

✅ **Complete and Ready**:
- Full backend implementation
- Interactive frontend (audience + presenter)
- Simulation mode working
- Metrics collection and aggregation
- WebSocket real-time updates
- Docker/Kubernetes deployment
- Comprehensive documentation
- Test suite
- CI/CD pipelines
- Helper scripts

⏳ **Pending Configuration**:
- Real OpenShift AI model endpoints (will be configured closer to event)
- H200 GPU reservations (October 19-21, 2026)
- Production deployment URL
- QR code with final public URL

📋 **Before Demo**:
- Deploy 4 quantization variants to OpenShift AI
- Reserve H200 GPUs
- Deploy demo app to OpenShift
- Test end-to-end with real models
- Generate QR code with public URL
- Record backup demo video

## Next Steps

1. **Test with Real Models** (August 2026):
   - Deploy Llama 3.3 in 4 quantization levels to OpenShift AI
   - Get model endpoint URLs
   - Update ConfigMap with real endpoints
   - Test with actual inference

2. **Production Deployment** (September 2026):
   - Deploy to OpenShift cluster
   - Configure DNS/route
   - Generate production QR code
   - Load test with target audience size

3. **Pre-Demo Preparation** (October 2026):
   - Reserve H200 GPUs
   - Final testing on October 19
   - Record backup video
   - Prepare simulation mode data

4. **Demo Day** (October 20, 2026):
   - Health checks 30 min before
   - Display QR code
   - Execute demo following DEMO_GUIDE.md
   - Monitor metrics live

## Support

- **Issues**: https://github.com/MarkellR-RedHat/pytorch-quantization-demo/issues
- **Email**: mrawls@redhat.com
- **Conference Contact**: Juliana Furlow (jsweek@redhat.com)

## License

MIT License - See LICENSE file

---

**Built by**: Markell Rawls  
**For**: PyTorch Conference 2026  
**Date Created**: July 9, 2026  
**GitHub**: https://github.com/MarkellR-RedHat/pytorch-quantization-demo
