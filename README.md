# PyTorch Quantization Demo

Interactive demo for PyTorch Conference 2026 showing real-time comparison of model quantization levels (FP32, FP16, INT8, INT4) running on OpenShift AI.

## Overview

This demo allows audience members to stress test different quantization variants of the same PyTorch model by sending requests from their mobile devices. Real-time metrics show performance, cost, and quality trade-offs.

## Features

- Interactive audience participation via QR code
- Real-time metrics dashboard for 4 quantization variants
- Simulation mode for failsafe demonstration
- WebSocket-based live updates
- Mobile-optimized interface
- Presenter controls for demo management

## Architecture

- **Backend**: FastAPI with WebSocket support
- **Frontend**: Vanilla JavaScript with WebSocket client
- **Model Serving**: OpenShift AI with vLLM backend
- **Monitoring**: Real-time metrics aggregation

## Quick Start

### Prerequisites

- Python 3.9+
- OpenShift AI access with deployed models
- Environment variables configured

### Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create `.env` file:

```env
# OpenShift AI Configuration
OPENSHIFT_AI_ENDPOINT=https://your-openshift-ai-endpoint
OPENSHIFT_AI_TOKEN=your-api-token

# Model Endpoints
MODEL_FP32_ENDPOINT=https://...
MODEL_FP16_ENDPOINT=https://...
MODEL_INT8_ENDPOINT=https://...
MODEL_INT4_ENDPOINT=https://...

# Demo Configuration
SIMULATION_MODE=false
PORT=8000
```

### Running Locally

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` for the audience interface.
Visit `http://localhost:8000/presenter` for presenter controls.

### Simulation Mode

For testing or failsafe demo execution:

```bash
# Via environment variable
SIMULATION_MODE=true python -m uvicorn app.main:app

# Via URL parameter (presenter only)
http://localhost:8000/presenter?mode=sim

# Via keyboard shortcut
Ctrl+Shift+S (on presenter view)
```

## Deployment to OpenShift

```bash
# Build container
podman build -t pytorch-quantization-demo .

# Tag for registry
podman tag pytorch-quantization-demo quay.io/your-org/pytorch-quantization-demo:latest

# Push to registry
podman push quay.io/your-org/pytorch-quantization-demo:latest

# Deploy to OpenShift
oc apply -f kubernetes/deployment.yaml
```

## Project Structure

```
pytorch-quantization-demo/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── websocket.py         # WebSocket handler
│   ├── metrics.py           # Metrics aggregation
│   ├── simulation.py        # Simulation mode logic
│   └── openshift.py         # OpenShift AI client
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── audience.js      # Audience interface
│   │   └── presenter.js     # Presenter dashboard
│   └── images/
├── templates/
│   ├── index.html           # Audience view
│   └── presenter.html       # Presenter view
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── route.yaml
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## API Endpoints

### Public Endpoints
- `GET /` - Audience interface
- `GET /qr` - QR code for mobile access
- `POST /request` - Submit inference request
- `WS /ws` - WebSocket for live updates

### Presenter Endpoints
- `GET /presenter` - Presenter dashboard
- `POST /simulation/toggle` - Toggle simulation mode
- `POST /demo/reset` - Reset demo state
- `GET /metrics/export` - Export metrics data

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENSHIFT_AI_ENDPOINT` | OpenShift AI API endpoint | Yes |
| `OPENSHIFT_AI_TOKEN` | Authentication token | Yes |
| `MODEL_FP32_ENDPOINT` | FP32 model serving endpoint | Yes |
| `MODEL_FP16_ENDPOINT` | FP16 model serving endpoint | Yes |
| `MODEL_INT8_ENDPOINT` | INT8 model serving endpoint | Yes |
| `MODEL_INT4_ENDPOINT` | INT4 model serving endpoint | Yes |
| `SIMULATION_MODE` | Enable simulation mode (true/false) | No |
| `PORT` | Server port (default: 8000) | No |

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/
```

## License

Copyright (c) 2026 Red Hat, Inc.

## Contact

Markell Rawls - mrawls@redhat.com
