
import asyncio
import logging
import random
import secrets
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import tracks
from app.ask import ask_stream, preset_list, preset_prompt
from app.benchmark import benchmark_data  # noqa: F401 - tests read the Llama data through main
from app.config import ALL_VARIANTS, REPO_ROOT, benchmark_label, build_note, settings, variant_label
from app.metrics import metrics_collector
from app.models import (
    AskRequest,
    DemoState,
    InferenceResponse,
    MetricsSnapshot,
)
from app.openshift import VariantNotConfigured, openshift_client
from app.quality import SCENARIOS, get_comparison
from app.simulation import simulator
from app.websocket import ConnectionManager

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Fixed prompts for background traffic; free text only comes from the presenter's Ask box.
PROMPTS = {
    "chat": "Explain the trade-offs of model quantization for production LLM deployments.",
    "reasoning": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
    "code": "Write a Python function that returns the second largest number in a list.",
    "summary": "Summarize zero trust security in three bullet points.",
}
SIM_MAX_INFLIGHT = 1000

demo_state = DemoState(
    is_running=False,
    simulation_mode=settings.simulation_mode,
    total_requests=0,
)
if settings.simulation_mode:
    simulator.enable()

connection_manager = ConnectionManager()

live_slots = {k: asyncio.Semaphore(settings.max_inflight_per_variant) for k in ALL_VARIANTS}
sim_slots = {k: asyncio.Semaphore(SIM_MAX_INFLIGHT) for k in ALL_VARIANTS}
auto_traffic_tasks: set[asyncio.Task] = set()


# helpers

def bench():
    return tracks.active().benchmark


def active_variants() -> list[str]:
    if simulator.is_enabled():
        fp8 = bench().has("FP8")
    else:
        fp8 = bool(settings.endpoint_for("FP8")) or settings.mode_for("FP8") == "recorded"
    return [k for k in ALL_VARIANTS if k != "FP8" or fp8]


def bench_label(key: str) -> str:
    """The benchmark's own label for a setup, which for INT4 follows the checkpoint that was measured."""
    if key == "INT4" and not tracks.active().labels.get(key) \
            and "RedHatAI" in str(bench().variant(key).get("checkpoint") or ""):
        return "INT4 (Red Hat W4A16)"
    return benchmark_label(key)


def current_mode() -> str:
    return "simulated" if simulator.is_enabled() else "live"


def state_payload() -> dict:
    demo_state.total_requests = metrics_collector.total_requests()
    demo_state.simulation_mode = simulator.is_enabled()
    return demo_state.model_dump(mode="json")


def snapshots() -> dict[str, MetricsSnapshot]:
    variants = {}
    for key in active_variants():
        in_flight = metrics_collector.active_requests[key]
        if simulator.is_enabled():
            basis = bench().latency_model(key, max(1, in_flight))[2] if bench().has(key) \
                else "no benchmark data"
        else:
            basis = f"live · {in_flight} in flight"
        variants[key] = {
            "label": variant_label(key),
            "source": current_mode(),
            "basis": basis,
            "gpus": bench().gpus(key),
            "weights_gib_per_gpu": bench().weights_gib_per_gpu(key),
            "include_cost": settings.gpu_hourly_usd > 0,
        }
    return metrics_collector.get_all_snapshots(variants)


def metrics_payload() -> dict:
    return {k: v.model_dump(mode="json") for k, v in snapshots().items()}


def cost_per_request(key: str, latency_ms: float, concurrency: int) -> float | None:
    """GPU-hours this request occupied, shared across concurrent requests (continuous batching)."""
    if settings.gpu_hourly_usd <= 0:
        return None
    gpu_hours = bench().gpus(key) * (latency_ms / 1000) / 3600
    return gpu_hours * settings.gpu_hourly_usd / max(1, concurrency)


def key_matches(key: str | None) -> bool:
    if not key:
        return False
    return secrets.compare_digest(key.encode(), settings.presenter_key.encode())


def require_presenter(request: Request):
    if not settings.presenter_key:
        return
    key = request.headers.get("x-presenter-key") or request.cookies.get("presenter_key")
    if not key:
        raise HTTPException(401, detail="presenter key required")
    if not key_matches(key):
        raise HTTPException(403, detail="invalid presenter key")


def is_https(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", "") if settings.trust_proxy else ""
    return request.url.scheme == "https" or proto == "https"


# inference

async def run_inference(model_type: str, prompt_id: str) -> InferenceResponse:
    if model_type not in active_variants():
        raise HTTPException(400, detail=f"variant {model_type} is not enabled")
    slots = (sim_slots if simulator.is_enabled() else live_slots)[model_type]
    if slots.locked():
        raise HTTPException(503, detail="busy")

    async with slots:
        concurrency = metrics_collector.active_requests[model_type] + 1
        metrics_collector.increment_active(model_type)
        try:
            if simulator.is_enabled():
                source = "simulated"
                text, latency_ms, tps, tokens = await simulator.simulate_request(
                    model_type, PROMPTS[prompt_id], concurrency
                )
            else:
                source = "live"
                text, latency_ms, tps, tokens = await openshift_client.send_request(
                    model_type, PROMPTS[prompt_id]
                )
        except VariantNotConfigured as e:
            metrics_collector.record_error(model_type)
            raise HTTPException(503, detail=str(e)) from e
        except (httpx.HTTPError, KeyError, ValueError) as e:
            metrics_collector.record_error(model_type)
            logger.error(f"Inference failed for {model_type}: {e!r}")
            raise HTTPException(502, detail="upstream error") from e
        finally:
            metrics_collector.decrement_active(model_type)

    cost = cost_per_request(model_type, latency_ms, concurrency)
    metrics_collector.record_request(model_type, latency_ms, tps, cost, tokens)
    return InferenceResponse(
        model_type=model_type,
        source=source,
        response_text=text,
        latency_ms=latency_ms,
        tokens_per_second=tps,
        completion_tokens=tokens,
        cost_per_request=cost,
        timestamp=datetime.now(),
    )


async def _auto_request(key: str):
    try:
        await run_inference(key, random.choice(list(PROMPTS)))
    except HTTPException:
        pass


async def auto_traffic_loop(key: str):
    """Poisson arrivals per variant, independent of how long each request takes."""
    while demo_state.is_running and settings.auto_traffic_rps > 0:
        await asyncio.sleep(random.expovariate(settings.auto_traffic_rps))
        if demo_state.is_running and key in active_variants():
            task = asyncio.create_task(_auto_request(key))
            auto_traffic_tasks.add(task)
            task.add_done_callback(auto_traffic_tasks.discard)


def start_auto_traffic():
    stop_auto_traffic()
    if not settings.auto_traffic or settings.auto_traffic_rps <= 0:
        return
    for key in ALL_VARIANTS:
        task = asyncio.create_task(auto_traffic_loop(key))
        auto_traffic_tasks.add(task)
        task.add_done_callback(auto_traffic_tasks.discard)


def stop_auto_traffic():
    for task in list(auto_traffic_tasks):
        task.cancel()
    auto_traffic_tasks.clear()


async def metrics_broadcast_task():
    while True:
        try:
            await connection_manager.broadcast({"type": "metrics_update", "data": metrics_payload()})
            await connection_manager.broadcast({"type": "state_update", "data": state_payload()})
        except Exception as e:  # keep the loop alive no matter what one tick does
            logger.error(f"Error in metrics broadcast task: {e!r}")
        await asyncio.sleep(0.5)


# app

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting PyTorch Quantization Demo ({current_mode()} mode)")
    if not settings.presenter_key:
        logger.warning(
            "PRESENTER_KEY is not set: anyone who can reach this app can start, stop and reset the demo"
        )
    if not simulator.is_enabled():
        await openshift_client.start()
    task = asyncio.create_task(metrics_broadcast_task())
    yield
    logger.info("Shutting down...")
    task.cancel()
    stop_auto_traffic()
    await openshift_client.close()


app = FastAPI(
    title="PyTorch Quantization Demo",
    description="Interactive demo comparing quantization and speculative decoding on vLLM",
    version="2.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=REPO_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=REPO_ROOT / "templates")


@app.get("/")
async def root():
    return RedirectResponse("/presenter", 307)


@app.get("/presenter", response_class=HTMLResponse)
async def presenter_view(
    request: Request, key: str | None = None, mode: str | None = None, track: str | None = None
):
    if settings.presenter_key:
        if key is not None:
            if not key_matches(key):
                raise HTTPException(403, detail="invalid presenter key")
            response = RedirectResponse("/presenter" + ("?mode=sim" if mode == "sim" else ""), 303)
            response.set_cookie(
                "presenter_key",
                key,
                max_age=12 * 3600,
                httponly=True,
                samesite="strict",
                secure=is_https(request),
            )
            return response
        require_presenter(request)

    if mode == "sim":
        simulator.enable()
        logger.info("Simulation mode enabled via URL parameter")
    if track:
        try:
            tracks.select(track)
        except (KeyError, tracks.TrackPending) as e:
            logger.warning(f"?track={track} ignored: {e}")

    return templates.TemplateResponse(
        request=request,
        name="presenter.html",
        context={
            "simulation_mode": simulator.is_enabled(), "demo_state": state_payload(), "title": tracks.TITLE,
        },
    )


@app.get("/arena", response_class=HTMLResponse)
async def arena_view(request: Request):
    return templates.TemplateResponse(request=request, name="arena.html", context={})


@app.post("/ask/{variant}", dependencies=[Depends(require_presenter)])
async def ask(variant: str, body: AskRequest):
    if variant not in active_variants():
        raise HTTPException(404, detail="unknown variant")
    prompt = preset_prompt(body.preset) or body.prompt.strip()
    if not prompt:
        raise HTTPException(422, detail="empty question")
    stream = ask_stream(variant, prompt, body.preset, not simulator.is_enabled(), bench())
    headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream, media_type="application/x-ndjson", headers=headers)


@app.get("/quality/{scenario}")
async def get_quality_comparison(scenario: str):
    if scenario not in SCENARIOS:
        raise HTTPException(404, detail=f"unknown scenario, use one of {', '.join(SCENARIOS)}")
    return get_comparison(scenario, active_variants())


@app.get("/api/config")
async def get_config(request: Request):
    return {
        "mode": current_mode(),
        "presenter_protected": bool(settings.presenter_key),
        "variants": [
            {
                "key": k,
                "label": variant_label(k),
                "build": build_note(k),
                "mode": "recorded" if simulator.is_enabled() else settings.mode_for(k),
                "gpus": bench().gpus(k),
                "weights_gib_per_gpu": bench().weights_gib_per_gpu(k),
                "device": tracks.active().device(k),
            }
            for k in active_variants()
        ],
        "track": tracks.describe(),
        "gpu_hourly_usd": settings.gpu_hourly_usd,
        "auto_traffic": settings.auto_traffic and settings.auto_traffic_rps > 0,
        "benchmark": bench().meta(bench_label),
        "presets": preset_list(active_variants(), simulator.is_enabled()),
    }


@app.websocket("/ws/presenter")
async def presenter_websocket(websocket: WebSocket):
    if settings.presenter_key:
        key = websocket.cookies.get("presenter_key") or websocket.headers.get("x-presenter-key")
        if not key_matches(key):
            await websocket.close(code=1008)
            return
    await connection_manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive()  # messages from the screen are ignored
            if message["type"] == "websocket.disconnect":
                break
    finally:
        connection_manager.disconnect(websocket)


@app.post("/track/{key}", dependencies=[Depends(require_presenter)])
async def switch_track(key: str):
    if key not in tracks.TRACKS:
        raise HTTPException(404, detail="unknown track")
    try:
        track = tracks.select(key)
    except tracks.TrackPending as e:
        raise HTTPException(409, detail=str(e)) from e
    logger.info(f"Track switched to {track.key}")
    await connection_manager.broadcast({"type": "track_switched", "data": tracks.describe(track)})
    return JSONResponse({"track": tracks.describe(track)})


@app.post("/simulation/toggle", dependencies=[Depends(require_presenter)])
async def toggle_simulation():
    if simulator.is_enabled():
        simulator.disable()
        await openshift_client.start()
    else:
        simulator.enable()
    demo_state.simulation_mode = simulator.is_enabled()
    message = f"Simulation mode {'enabled' if simulator.is_enabled() else 'disabled'}"
    logger.info(message)
    if demo_state.is_running:
        start_auto_traffic()
    await connection_manager.broadcast(
        {"type": "simulation_toggled", "data": {"enabled": simulator.is_enabled(), "message": message}}
    )
    await connection_manager.broadcast({"type": "state_update", "data": state_payload()})
    return JSONResponse({"message": message, "enabled": simulator.is_enabled()})


@app.post("/demo/start", dependencies=[Depends(require_presenter)])
async def start_demo():
    demo_state.is_running = True
    demo_state.start_time = datetime.now()
    metrics_collector.reset()
    start_auto_traffic()
    logger.info(f"Demo started ({current_mode()} mode, auto-traffic {'on' if auto_traffic_tasks else 'off'})")
    await connection_manager.broadcast({"type": "state_update", "data": state_payload()})
    return JSONResponse(
        {"message": "Demo started", "auto_traffic": bool(auto_traffic_tasks), "mode": current_mode()}
    )


@app.post("/demo/stop", dependencies=[Depends(require_presenter)])
async def stop_demo():
    demo_state.is_running = False
    stop_auto_traffic()
    logger.info("Demo stopped")
    await connection_manager.broadcast({"type": "state_update", "data": state_payload()})
    return JSONResponse({"message": "Demo stopped"})


@app.post("/demo/reset", dependencies=[Depends(require_presenter)])
async def reset_demo():
    demo_state.is_running = False
    demo_state.start_time = None
    stop_auto_traffic()
    metrics_collector.reset()
    logger.info("Demo reset")
    await connection_manager.broadcast({"type": "state_update", "data": state_payload()})
    return JSONResponse({"message": "Demo reset"})


@app.get("/metrics")
async def get_metrics():
    return metrics_payload()


@app.get("/state")
async def get_state():
    return state_payload()


@app.get("/health")
async def health_check():
    return {"ok": True, "mode": current_mode()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, log_level=settings.log_level.lower())
