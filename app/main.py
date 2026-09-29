"""Main FastAPI application"""

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.models import InferenceRequest, InferenceResponse, DemoState, MetricsSnapshot
from app.openshift import openshift_client
from app.simulation import simulator
from app.metrics import metrics_collector
from app.websocket import connection_manager

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

demo_state = DemoState(
    is_running=False,
    simulation_mode=settings.simulation_mode,
    participant_count=0,
    total_requests=0
)

if settings.simulation_mode:
    simulator.enable()

auto_traffic_handle: asyncio.Task | None = None

TRAFFIC_PROMPTS = [
    "Explain the trade-offs of model quantization for production LLM deployments.",
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
    "Write a Python function that returns the second largest number in a list.",
    "Compare microservices vs monolithic architecture for high-traffic web apps.",
    "Summarize zero trust security in three bullet points.",
]

TRAFFIC_VARIANTS = ["FP16", "INT4", "SPEC_DECODE"]
VARIANT_WEIGHTS = [0.3, 0.45, 0.25]


async def auto_traffic_task():
    """Self-driving traffic generator — fills the dashboard without audience participation."""
    ramp_seconds = 8
    start = asyncio.get_event_loop().time()

    while demo_state.is_running and simulator.is_enabled():
        elapsed = asyncio.get_event_loop().time() - start
        ramp = min(elapsed / ramp_seconds, 1.0)
        delay = 0.6 - (ramp * 0.4)

        variant = random.choices(TRAFFIC_VARIANTS, weights=VARIANT_WEIGHTS, k=1)[0]
        prompt = random.choice(TRAFFIC_PROMPTS)

        _, latency_ms, tokens_per_sec, cost = await simulator.simulate_request(variant, prompt)
        metrics_collector.record_request(variant, latency_ms, tokens_per_sec, cost)
        demo_state.total_requests += 1

        await asyncio.sleep(delay)


async def metrics_broadcast_task():
    while True:
        try:
            metrics = metrics_collector.get_all_snapshots()
            metrics_dict = {k: v.model_dump() for k, v in metrics.items()}
            await connection_manager.broadcast_metrics(metrics_dict)

            demo_state.participant_count = connection_manager.get_connection_count()
            demo_state.total_requests = sum(metrics_collector.request_counts.values())
            demo_state.metrics = metrics

            state_dict = demo_state.model_dump()
            if state_dict.get('start_time'):
                state_dict['start_time'] = state_dict['start_time'].isoformat()
            await connection_manager.broadcast_state(state_dict)

            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error in metrics broadcast task: {e}")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PyTorch Quantization Demo...")
    logger.info(f"Simulation mode: {simulator.is_enabled()}")
    task = asyncio.create_task(metrics_broadcast_task())
    yield
    logger.info("Shutting down...")
    task.cancel()


app = FastAPI(
    title="PyTorch Quantization Demo",
    description="Interactive demo comparing PyTorch model quantization and speculative decoding",
    version="2.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def audience_view(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"simulation_mode": simulator.is_enabled()}
    )


@app.get("/presenter", response_class=HTMLResponse)
async def presenter_view(request: Request, mode: str = None):
    if mode == "sim":
        simulator.enable()
        demo_state.simulation_mode = True
        logger.info("Simulation mode enabled via URL parameter")

    return templates.TemplateResponse(
        request=request,
        name="presenter.html",
        context={
            "simulation_mode": simulator.is_enabled(),
            "demo_state": demo_state.model_dump()
        }
    )


@app.post("/request", response_model=InferenceResponse)
async def submit_inference_request(request: InferenceRequest):
    try:
        metrics_collector.increment_active(request.model_type)

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

        metrics_collector.record_request(
            request.model_type,
            latency_ms,
            tokens_per_sec,
            cost
        )
        metrics_collector.decrement_active(request.model_type)

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


@app.get("/quality/{scenario}")
async def get_quality_comparison(scenario: str = "complex_reasoning"):
    """Get pre-scripted quality comparison for presenter dashboard"""
    return simulator.get_quality_comparison(scenario)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message: {data}")
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


@app.websocket("/ws/presenter")
async def presenter_websocket(websocket: WebSocket):
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
    global auto_traffic_handle

    demo_state.is_running = True
    demo_state.start_time = datetime.now()
    demo_state.total_requests = 0
    metrics_collector.reset()

    if simulator.is_enabled():
        if auto_traffic_handle and not auto_traffic_handle.done():
            auto_traffic_handle.cancel()
        auto_traffic_handle = asyncio.create_task(auto_traffic_task())
        logger.info("Demo started with auto-traffic")
    else:
        logger.info("Demo started (live mode)")

    state_dict = demo_state.model_dump()
    if state_dict.get('start_time'):
        state_dict['start_time'] = state_dict['start_time'].isoformat()
    await connection_manager.broadcast_state(state_dict)

    return JSONResponse({"message": "Demo started", "auto_traffic": simulator.is_enabled()})


@app.post("/demo/stop")
async def stop_demo():
    global auto_traffic_handle

    demo_state.is_running = False
    if auto_traffic_handle and not auto_traffic_handle.done():
        auto_traffic_handle.cancel()
        auto_traffic_handle = None

    logger.info("Demo stopped")
    await connection_manager.broadcast_state(demo_state.model_dump())
    return JSONResponse({"message": "Demo stopped"})


@app.post("/demo/reset")
async def reset_demo():
    global auto_traffic_handle

    demo_state.is_running = False
    demo_state.total_requests = 0
    demo_state.start_time = None
    demo_state.participant_count = 0

    if auto_traffic_handle and not auto_traffic_handle.done():
        auto_traffic_handle.cancel()
        auto_traffic_handle = None

    metrics_collector.reset()

    logger.info("Demo reset")
    await connection_manager.broadcast_state(demo_state.model_dump())
    return JSONResponse({"message": "Demo reset"})


@app.get("/metrics")
async def get_metrics():
    metrics = metrics_collector.get_all_snapshots()
    return {k: v.model_dump() for k, v in metrics.items()}


@app.get("/state")
async def get_state():
    demo_state.participant_count = connection_manager.get_connection_count()
    total = sum(metrics_collector.request_counts.values())
    demo_state.total_requests = total

    state_dict = demo_state.model_dump()
    if state_dict.get('start_time'):
        state_dict['start_time'] = state_dict['start_time'].isoformat()
    return state_dict


@app.get("/health")
async def health_check():
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
