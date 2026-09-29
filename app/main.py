"""Main FastAPI application"""

import asyncio
import io
import json
import logging
import math
import random
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
import segno
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.arena import HARD_PROMPT_LABELS, MAX_STATE_BYTES, ArenaRelay, ArenaVotes, handle_for
from app.ask import ask_stream, preset_prompt
from app.benchmark import benchmark_data
from app.config import ALL_VARIANTS, REPO_ROOT, settings, variant_label
from app.metrics import metrics_collector
from app.models import (
    AskRequest,
    BackRequest,
    DemoState,
    HardPromptRequest,
    InferenceRequest,
    InferenceResponse,
    MetricsSnapshot,
)
from app.openshift import VariantNotConfigured, openshift_client
from app.quality import SCENARIOS, get_comparison
from app.ratelimit import RateLimiter
from app.simulation import simulator
from app.websocket import ConnectionManager

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Fixed server-side prompts. Phones pick an id; free text never reaches a model or the big screen.
PROMPTS = {
    "chat": "Explain the trade-offs of model quantization for production LLM deployments.",
    "reasoning": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
    "code": "Write a Python function that returns the second largest number in a list.",
    "summary": "Summarize zero trust security in three bullet points.",
}
SIM_MAX_INFLIGHT = 1000
AID_RE = re.compile(r"[0-9a-f]{8}")

demo_state = DemoState(
    is_running=False,
    simulation_mode=settings.simulation_mode,
    participant_count=0,
    total_requests=0,
)
if settings.simulation_mode:
    simulator.enable()

connection_manager = ConnectionManager(max_connections=settings.max_connections)
arena_votes = ArenaVotes()
arena_relay = ArenaRelay(connection_manager)

request_limiter = RateLimiter(rate=4, burst=8)
ip_limiter = RateLimiter(rate=50, burst=100)  # ceiling for a whole venue NAT
vote_limiter = RateLimiter(rate=2, burst=4)
hard_client_limiter = RateLimiter(rate=1 / 4, burst=1)
hard_global_limiter = RateLimiter(rate=1.0, burst=1)

live_slots = {k: asyncio.Semaphore(settings.max_inflight_per_variant) for k in ALL_VARIANTS}
sim_slots = {k: asyncio.Semaphore(SIM_MAX_INFLIGHT) for k in ALL_VARIANTS}
auto_traffic_tasks: set[asyncio.Task] = set()


# ---------------------------------------------------------------- helpers

def active_variants() -> list[str]:
    fp8 = benchmark_data.has("FP8") if simulator.is_enabled() else bool(settings.model_fp8_endpoint)
    return [k for k in ALL_VARIANTS if k != "FP8" or fp8]


def current_mode() -> str:
    return "simulated" if simulator.is_enabled() else "live"


def state_payload() -> dict:
    demo_state.participant_count = connection_manager.get_connection_count()
    demo_state.total_requests = metrics_collector.total_requests()
    demo_state.simulation_mode = simulator.is_enabled()
    return demo_state.model_dump(mode="json")


def snapshots() -> dict[str, MetricsSnapshot]:
    variants = {}
    for key in active_variants():
        in_flight = metrics_collector.active_requests[key]
        if simulator.is_enabled():
            basis = benchmark_data.latency_model(key, max(1, in_flight))[2] if benchmark_data.has(key) \
                else "no benchmark data"
        else:
            basis = f"live · {in_flight} in flight"
        variants[key] = {
            "label": variant_label(key),
            "source": current_mode(),
            "basis": basis,
            "gpus": benchmark_data.gpus(key),
            "weights_gb": benchmark_data.weights_gb(key),
            "include_cost": settings.gpu_hourly_usd > 0,
        }
    return metrics_collector.get_all_snapshots(variants)


def metrics_payload() -> dict:
    return {k: v.model_dump(mode="json") for k, v in snapshots().items()}


def cost_per_request(key: str, latency_ms: float, concurrency: int) -> float | None:
    """GPU-hours this request occupied, shared across concurrent requests (continuous batching)."""
    if settings.gpu_hourly_usd <= 0:
        return None
    gpu_hours = benchmark_data.gpus(key) * (latency_ms / 1000) / 3600
    return gpu_hours * settings.gpu_hourly_usd / max(1, concurrency)


def client_ip(request: Request) -> str:
    if settings.trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def client_key(request: Request) -> str:
    """Anonymous cookie when the client has one, so phones sharing venue NAT get separate buckets."""
    aid = request.cookies.get("aid", "")
    return f"aid:{aid}" if AID_RE.fullmatch(aid) else f"ip:{client_ip(request)}"


def limit(limiter: RateLimiter, key: str, detail: str = "rate limited"):
    retry = limiter.take(key)
    if retry:
        raise HTTPException(429, detail=detail, headers={"Retry-After": str(max(1, math.ceil(retry)))})


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


# ---------------------------------------------------------------- inference

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
            await connection_manager.broadcast_metrics(metrics_payload())
            await connection_manager.broadcast_state(state_payload())
        except Exception as e:  # keep the loop alive no matter what one tick does
            logger.error(f"Error in metrics broadcast task: {e!r}")
        await asyncio.sleep(0.5)


# ---------------------------------------------------------------- app

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


@app.middleware("http")
async def anonymous_id(request: Request, call_next):
    aid = request.cookies.get("aid", "")
    issued = None
    if not AID_RE.fullmatch(aid):
        aid = issued = secrets.token_hex(4)
    request.state.aid = aid
    response = await call_next(request)
    if issued:
        response.set_cookie("aid", issued, max_age=7 * 86400, httponly=True, samesite="lax")
    return response


@app.get("/", response_class=HTMLResponse)
async def audience_view(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"simulation_mode": simulator.is_enabled()},
    )


@app.get("/presenter", response_class=HTMLResponse)
async def presenter_view(request: Request, key: str | None = None, mode: str | None = None):
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

    return templates.TemplateResponse(
        request=request,
        name="presenter.html",
        context={"simulation_mode": simulator.is_enabled(), "demo_state": state_payload()},
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
    stream = ask_stream(variant, prompt, body.preset, not simulator.is_enabled(), benchmark_data)
    headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream, media_type="application/x-ndjson", headers=headers)


@app.post("/request", response_model=InferenceResponse)
async def submit_inference_request(body: InferenceRequest, request: Request):
    limit(ip_limiter, client_ip(request))
    limit(request_limiter, client_key(request))
    return await run_inference(body.model_type, body.prompt_id)


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
                "gpus": benchmark_data.gpus(k),
                "weights_gb": benchmark_data.weights_gb(k),
            }
            for k in active_variants()
        ],
        "public_url": settings.public_url or str(request.base_url),
        "gpu_hourly_usd": settings.gpu_hourly_usd,
        "auto_traffic": settings.auto_traffic and settings.auto_traffic_rps > 0,
        "benchmark": benchmark_data.meta(variant_label),
    }


@app.get("/api/me")
async def get_me(request: Request):
    return {"handle": handle_for(request.state.aid)}


@app.get("/qr.svg")
async def qr_code(request: Request):
    url = settings.public_url or str(request.base_url)
    buffer = io.BytesIO()
    qr = segno.make(url, error="m")
    qr.save(buffer, kind="svg", dark="#151515", light=None, border=2, scale=8, xmldecl=False)
    return Response(buffer.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.post("/arena/hard-prompt", status_code=202)
async def arena_hard_prompt(body: HardPromptRequest, request: Request):
    client = client_key(request)
    limit(hard_client_limiter, client, detail="one hard prompt every 4 seconds")
    retry = hard_global_limiter.take("global")
    if retry:
        hard_client_limiter.refund(client)
        raise HTTPException(429, detail="queue full", headers={"Retry-After": str(max(1, math.ceil(retry)))})
    event = {"kind": body.kind, "label": HARD_PROMPT_LABELS[body.kind], "from": handle_for(request.state.aid)}
    await connection_manager.broadcast({"type": "arena_hard_prompt", "data": event}, target="presenter")
    return event


@app.post("/arena/back")
async def arena_back(body: BackRequest, request: Request):
    limit(ip_limiter, client_ip(request))
    limit(vote_limiter, client_key(request))
    arena_votes.back(request.state.aid, body.variant)
    summary = arena_votes.summary()
    await connection_manager.broadcast({"type": "arena_votes", "data": summary})
    return summary


@app.get("/arena/votes")
async def get_arena_votes():
    return arena_votes.summary()


@app.get("/arena/state")
async def get_arena_state():
    return {"data": arena_relay.latest}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not await connection_manager.connect(websocket):
        return
    try:
        await connection_manager.send(websocket, {"type": "arena_votes", "data": arena_votes.summary()})
        if arena_relay.latest is not None:
            await connection_manager.send(websocket, {"type": "arena_state", "data": arena_relay.latest})
        while True:
            message = await websocket.receive()  # audience messages are ignored
            if message["type"] == "websocket.disconnect":
                break
    finally:
        connection_manager.disconnect(websocket)


@app.websocket("/ws/presenter")
async def presenter_websocket(websocket: WebSocket):
    if settings.presenter_key:
        key = websocket.cookies.get("presenter_key") or websocket.headers.get("x-presenter-key")
        if not key_matches(key):
            await websocket.close(code=1008)
            return
    await connection_manager.connect(websocket, is_presenter=True)
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if not text or len(text.encode()) > MAX_STATE_BYTES:
                continue
            try:
                payload = json.loads(text)
            except ValueError:
                continue
            if (
                isinstance(payload, dict)
                and payload.get("type") == "arena_state"
                and isinstance(payload.get("data"), dict)
            ):
                await arena_relay.submit(payload["data"])
    finally:
        connection_manager.disconnect(websocket)


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
    await connection_manager.notify_presenter(
        "simulation_toggled", {"enabled": simulator.is_enabled(), "message": message}
    )
    await connection_manager.broadcast_state(state_payload())
    return JSONResponse({"message": message, "enabled": simulator.is_enabled()})


@app.post("/demo/start", dependencies=[Depends(require_presenter)])
async def start_demo():
    demo_state.is_running = True
    demo_state.start_time = datetime.now()
    metrics_collector.reset()
    start_auto_traffic()
    logger.info(f"Demo started ({current_mode()} mode, auto-traffic {'on' if auto_traffic_tasks else 'off'})")
    await connection_manager.broadcast_state(state_payload())
    return JSONResponse(
        {"message": "Demo started", "auto_traffic": bool(auto_traffic_tasks), "mode": current_mode()}
    )


@app.post("/demo/stop", dependencies=[Depends(require_presenter)])
async def stop_demo():
    demo_state.is_running = False
    stop_auto_traffic()
    logger.info("Demo stopped")
    await connection_manager.broadcast_state(state_payload())
    return JSONResponse({"message": "Demo stopped"})


@app.post("/demo/reset", dependencies=[Depends(require_presenter)])
async def reset_demo():
    demo_state.is_running = False
    demo_state.start_time = None
    stop_auto_traffic()
    metrics_collector.reset()
    arena_votes.reset()
    arena_relay.reset()
    request_limiter.reset()
    hard_client_limiter.reset()
    hard_global_limiter.reset()
    logger.info("Demo reset")
    await connection_manager.broadcast_state(state_payload())
    await connection_manager.broadcast({"type": "arena_votes", "data": arena_votes.summary()})
    return JSONResponse({"message": "Demo reset"})


@app.get("/metrics")
async def get_metrics():
    return metrics_payload()


@app.get("/state")
async def get_state():
    return state_payload()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": current_mode(),
        "simulation_mode": simulator.is_enabled(),
        "active_connections": connection_manager.get_connection_count(),
        "total_requests": metrics_collector.total_requests(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, log_level=settings.log_level.lower())
