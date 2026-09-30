# Development guide

## Setup

You need Python 3.11 or newer. Node is optional, and the arena parity tests use it when it's installed.

```bash
./scripts/setup.sh          # creates venv/ and installs requirements.txt
source venv/bin/activate
pip install -r requirements-dev.txt
./scripts/run-local.sh      # simulated mode on http://localhost:8000
```

No `.env` file is needed for simulated mode. Every setting has a default, and `.env.example` lists all of them with a comment each.

## How the pieces fit

The backend is one FastAPI process (`app/main.py`) that keeps all state in memory, which is why the OpenShift deployment runs a single replica.

- `app/benchmark.py` loads `benchmark_results.json` and any `bench/<VARIANT>/c<N>.json` sweeps from `vllm bench serve`. In simulated mode, each request's latency is sampled from a lognormal fitted to the measured mean and p95, and it's scaled by the in-flight concurrency once sweep files exist.
- `app/simulation.py` replays those timings, sleeping for the real latency so concurrency builds up the way it would against real GPUs.
- `app/openshift.py` is the live client for vLLM's OpenAI-compatible chat completions API. It runs at temperature 0 and computes tokens per second from `completion_tokens`.
- `app/ask.py` streams the Ask scene: one question to every variant, with time to first token, tokens per second, and total time measured per variant. In replay mode it plays back the measured speed, using captured outputs from `quality/<VARIANT>/<scenario>.json` for the presets when they exist and the baseline's text otherwise.

The booth arena (`/arena`) runs entirely in the browser. `static/js/arena-core.js` is the physics and the network forward pass (no DOM, so Node can run it in tests), `static/js/arena.js` is the game loop and renderer, and `static/arena/policy.json` holds the weights exported by `arena/train_policy.py`.

## Changing the arena

The physics constants live in two places, `arena/train_policy.py` and the world block of `static/arena/policy.json`, which the browser reads. After any change to the physics, the features, or the training:

```bash
pip install -r arena/requirements.txt
python arena/train_policy.py
pytest tests/test_arena.py
```

The training script prints the evaluation for every variant. `tests/test_arena.py` fails if the browser and Python disagree on a single frame, or if the numbers quoted in the README and on screen stop matching the exported evaluation, so update the caption and README when the numbers move.

## Working on the dashboard

The presenter page is a fixed 1920 by 1080 stage scaled to the window, so check layout changes at full HD and at 1280 by 720, in both themes (`T`). Colors are defined once in `static/css/base.css`. The four variant colors were checked for color-blind separation and contrast, so keep them unless you re-run that check.

## Tests and lint

```bash
pytest
ruff check app tests scripts
```

CI runs both on Python 3.11 and 3.12, builds the container, and checks `/health`.

## Troubleshooting

**The Ask box says to open /presenter?key=….** `PRESENTER_KEY` is set. Open `/presenter?key=<value>` once and the browser keeps a cookie for 12 hours.

**Live requests return 503.** The in-flight cap for that variant is full. Raise `MAX_INFLIGHT_PER_VARIANT` if the GPUs have headroom.

**Port 8000 is taken.** `lsof -i :8000` shows what's using it, or run on another port with `python -m uvicorn app.main:app --port 8001`.
