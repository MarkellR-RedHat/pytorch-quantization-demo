"""Main FastAPI application"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import qrcode
from io import BytesIO
import base64

from app.config import settings
from app.models import InferenceRequest, InferenceResponse, DemoState, MetricsSnapshot
from app.openshift import openshift_client
from app.simulation import simulator
from app.metrics import metrics_collector
from app.websocket import connection_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Demo state
demo_state = DemoState(
    is_running=False,
    simulation_mode=settings.simulation_mode,
    participant_count=0,
    total_requests=0
)


# Background task for periodic metrics broadcast
async def metrics_broadcast_task():
    """Periodically broadcast metrics to all connected clients"""
    while True:
        try:
            if demo_state.is_running:
                metrics = metrics_collector.get_all_snapshots()
                metrics_dict = {k: v.model_dump() for k, v in metrics.items()}
                await connection_manager.broadcast_metrics(metrics_dict)

                # Update demo state
                demo_state.participant_count = connection_manager.get_connection_count()
                demo_state.metrics = metrics

            await asyncio.sleep(1)  # Update every second
        except Exception as e:
            logger.error(f"Error in metrics broadcast task: {e}")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting PyTorch Quantization Demo...")
    logger.info(f"Simulation mode: {simulator.is_enabled()}")

    # Start background tasks
    task = asyncio.create_task(metrics_broadcast_task())

    yield

    # Shutdown
    logger.info("Shutting down...")
    task.cancel()


# Create FastAPI app
app = FastAPI(
    title="PyTorch Quantization Demo",
    description="Interactive demo comparing PyTorch model quantization levels",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def audience_view(request: Request):
    """Serve the audience interface"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "simulation_mode": simulator.is_enabled()}
    )


@app.get("/presenter", response_class=HTMLResponse)
async def presenter_view(request: Request, mode: str = None):
    """Serve the presenter dashboard"""
    # Check for simulation mode toggle via URL parameter
    if mode == "sim":
        simulator.enable()
        demo_state.simulation_mode = True
        logger.info("Simulation mode enabled via URL parameter")

    return templates.TemplateResponse(
        "presenter.html",
        {
            "request": request,
            "simulation_mode": simulator.is_enabled(),
            "demo_state": demo_state.model_dump()
        }
    )


@app.get("/qr")
async def get_qr_code():
    """Generate QR code for audience to scan"""
    # In production, this would be the actual public URL
    url = f"http://localhost:{settings.port}/"

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return JSONResponse({
        "qr_code": f"data:image/png;base64,{img_str}",
        "url": url
    })


@app.post("/request", response_model=InferenceResponse)
async def submit_inference_request(request: InferenceRequest):
    """Submit an inference request to a model variant"""
    try:
        metrics_collector.increment_active(request.model_type)

        # Use simulation mode if enabled
        if simulator.is_enabled():
            response_text, latency_ms, tokens_per_sec, cost = await simulator.simulate_request(
                request.model_type,
                request.prompt
            )
        else:
            response_text, latency_ms, tokens_per_sec, cost = await openshift_client.send_request(
                request.model_type,
                request.prompt
            )

        # Record metrics
        metrics_collector.record_request(
            request.model_type,
            latency_ms,
            tokens_per_sec,
            cost
        )
        metrics_collector.decrement_active(request.model_type)

        # Update total requests
        demo_state.total_requests += 1

        return InferenceResponse(
            model_type=request.model_type,
            response_text=response_text,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_sec,
            cost_per_request=cost,
            timestamp=datetime.now()
        )

    except Exception as e:
        metrics_collector.decrement_active(request.model_type)
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await connection_manager.connect(websocket)

    try:
        while True:
            # Keep connection alive and receive any client messages
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message: {data}")

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


@app.websocket("/ws/presenter")
async def presenter_websocket(websocket: WebSocket):
    """WebSocket endpoint for presenter dashboard"""
    await connection_manager.connect(websocket, is_presenter=True)

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received presenter message: {data}")

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Presenter WebSocket error: {e}")
        connection_manager.disconnect(websocket)


@app.post("/simulation/toggle")
async def toggle_simulation():
    """Toggle simulation mode (presenter only)"""
    if simulator.is_enabled():
        simulator.disable()
        demo_state.simulation_mode = False
        message = "Simulation mode disabled"
    else:
        simulator.enable()
        demo_state.simulation_mode = True
        message = "Simulation mode enabled"

    logger.info(message)
    await connection_manager.notify_presenter("simulation_toggled", {
        "enabled": simulator.is_enabled(),
        "message": message
    })

    return JSONResponse({"message": message, "enabled": simulator.is_enabled()})


@app.post("/demo/start")
async def start_demo():
    """Start the demo"""
    demo_state.is_running = True
    demo_state.start_time = datetime.now()
    demo_state.total_requests = 0
    metrics_collector.reset()

    logger.info("Demo started")
    await connection_manager.broadcast_state(demo_state.model_dump())

    return JSONResponse({"message": "Demo started", "state": demo_state.model_dump()})


@app.post("/demo/stop")
async def stop_demo():
    """Stop the demo"""
    demo_state.is_running = False

    logger.info("Demo stopped")
    await connection_manager.broadcast_state(demo_state.model_dump())

    return JSONResponse({"message": "Demo stopped", "state": demo_state.model_dump()})


@app.post("/demo/reset")
async def reset_demo():
    """Reset demo state and metrics"""
    demo_state.is_running = False
    demo_state.total_requests = 0
    demo_state.start_time = None
    demo_state.participant_count = 0
    metrics_collector.reset()

    logger.info("Demo reset")
    await connection_manager.broadcast_state(demo_state.model_dump())

    return JSONResponse({"message": "Demo reset", "state": demo_state.model_dump()})


@app.get("/metrics")
async def get_metrics():
    """Get current metrics for all model variants"""
    metrics = metrics_collector.get_all_snapshots()
    return {k: v.model_dump() for k, v in metrics.items()}


@app.get("/state")
async def get_state():
    """Get current demo state"""
    demo_state.participant_count = connection_manager.get_connection_count()
    return demo_state.model_dump()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "simulation_mode": simulator.is_enabled(),
        "active_connections": connection_manager.get_connection_count(),
        "total_requests": demo_state.total_requests
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
