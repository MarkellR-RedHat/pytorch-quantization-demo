# Development Guide

Complete guide for developing and testing the PyTorch Quantization Demo locally.

## Prerequisites

- Python 3.9 or higher
- Virtual environment tool (venv)
- Git
- Docker/Podman (for containerization)
- Access to OpenShift AI cluster (for production mode)

## Local Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/MarkellR-RedHat/pytorch-quantization-demo.git
cd pytorch-quantization-demo
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` file with your configuration:

```env
# For local development, use simulation mode
SIMULATION_MODE=true

# These can be dummy values in simulation mode
OPENSHIFT_AI_ENDPOINT=https://dummy-endpoint.com
OPENSHIFT_AI_TOKEN=dummy-token
MODEL_FP32_ENDPOINT=https://dummy.com
MODEL_FP16_ENDPOINT=https://dummy.com
MODEL_INT8_ENDPOINT=https://dummy.com
MODEL_INT4_ENDPOINT=https://dummy.com

PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO
```

### 5. Run Application

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Application will be available at:
- Audience interface: http://localhost:8000
- Presenter dashboard: http://localhost:8000/presenter

## Development Workflow

### Running in Simulation Mode

Simulation mode is perfect for local development and testing without needing real model endpoints:

```bash
# Via environment variable
SIMULATION_MODE=true python -m uvicorn app.main:app --reload

# Or set in .env file
SIMULATION_MODE=true
```

Features in simulation mode:
- Synthetic metrics generation
- Realistic latency simulation
- No external dependencies
- Perfect for testing UI/UX

### Testing with Real Models

To test with actual OpenShift AI models:

1. Get OpenShift AI credentials and model endpoints
2. Update `.env` with real values
3. Set `SIMULATION_MODE=false`
4. Run the application

```bash
SIMULATION_MODE=false python -m uvicorn app.main:app --reload
```

### Presenter Controls

When running the presenter dashboard:

- **Ctrl+Shift+S**: Toggle simulation mode on/off
- **URL parameter**: `http://localhost:8000/presenter?mode=sim` to enable simulation
- **Reset button**: Clear all metrics and reset demo state

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-asyncio pytest-cov
```

### Run All Tests

```bash
pytest tests/
```

### Run with Coverage

```bash
pytest --cov=app --cov-report=html tests/
```

View coverage report:
```bash
open htmlcov/index.html
```

### Run Specific Test File

```bash
pytest tests/test_simulation.py -v
```

## Building Container Image

### Using Docker

```bash
docker build -t pytorch-quantization-demo:latest .
docker run -p 8000:8000 --env-file .env pytorch-quantization-demo:latest
```

### Using Podman

```bash
podman build -t pytorch-quantization-demo:latest .
podman run -p 8000:8000 --env-file .env pytorch-quantization-demo:latest
```

### Push to Registry

```bash
# Tag for your registry
podman tag pytorch-quantization-demo:latest quay.io/your-org/pytorch-quantization-demo:latest

# Login to registry
podman login quay.io

# Push
podman push quay.io/your-org/pytorch-quantization-demo:latest
```

## Deploying to OpenShift

### 1. Create Namespace/Project

```bash
oc new-project pytorch-demo
```

### 2. Create Secrets

```bash
oc create secret generic openshift-ai-config \
  --from-literal=endpoint=https://your-openshift-ai-endpoint.com \
  --from-literal=token=your-api-token
```

### 3. Update ConfigMap

Edit `kubernetes/configmap.yaml` with your actual model endpoints, then apply:

```bash
oc apply -f kubernetes/configmap.yaml
```

### 4. Deploy Application

```bash
oc apply -f kubernetes/deployment.yaml
oc apply -f kubernetes/service.yaml
oc apply -f kubernetes/route.yaml
```

### 5. Get Route URL

```bash
oc get route pytorch-quantization-demo
```

## Project Structure Explained

```
pytorch-quantization-demo/
├── app/                      # Backend application
│   ├── main.py              # FastAPI app with all endpoints
│   ├── config.py            # Configuration and settings
│   ├── models.py            # Pydantic data models
│   ├── openshift.py         # OpenShift AI client
│   ├── simulation.py        # Simulation mode logic
│   ├── metrics.py           # Metrics collection and aggregation
│   └── websocket.py         # WebSocket connection manager
├── static/                  # Frontend assets
│   ├── css/style.css        # Styling for both views
│   ├── js/audience.js       # Audience interface logic
│   └── js/presenter.js      # Presenter dashboard logic
├── templates/               # HTML templates
│   ├── index.html           # Audience view
│   └── presenter.html       # Presenter dashboard
├── kubernetes/              # OpenShift/K8s manifests
├── tests/                   # Test suite
└── [config files]           # Docker, requirements, etc.
```

## Common Development Tasks

### Adding New Metrics

1. Update `app/models.py` to add metric field to `MetricsSnapshot`
2. Update `app/metrics.py` to collect and calculate new metric
3. Update `templates/presenter.html` to display new metric
4. Update `static/js/presenter.js` to update new metric value

### Modifying UI

- Audience interface: Edit `templates/index.html`, `static/js/audience.js`, and `static/css/style.css`
- Presenter dashboard: Edit `templates/presenter.html`, `static/js/presenter.js`, and `static/css/style.css`

### Adding API Endpoints

Add new endpoints to `app/main.py`:

```python
@app.get("/your-endpoint")
async def your_endpoint():
    return {"message": "Hello"}
```

## Troubleshooting

### WebSocket Connection Issues

If WebSocket connections fail:
- Check firewall settings
- Ensure correct protocol (ws:// for http, wss:// for https)
- Check browser console for errors

### Simulation Mode Not Working

- Verify `SIMULATION_MODE=true` in environment
- Check logs for simulation mode activation message
- Try toggling with Ctrl+Shift+S in presenter view

### Model Endpoint Errors

- Verify endpoints are reachable
- Check authentication token is valid
- Enable debug logging: `LOG_LEVEL=DEBUG`

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn app.main:app --port 8001
```

## Performance Optimization

### For Large Audience

- Increase replica count in `kubernetes/deployment.yaml`
- Use Redis for shared state across replicas
- Enable horizontal pod autoscaling

### For Better Metrics

- Adjust `window_size` in `MetricsCollector` initialization
- Modify `maxlen` values in deque collections
- Increase WebSocket broadcast frequency for more responsive updates

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Support

For issues or questions:
- GitHub Issues: https://github.com/MarkellR-RedHat/pytorch-quantization-demo/issues
- Email: mrawls@redhat.com

## License

MIT License - see [LICENSE](LICENSE) file.
