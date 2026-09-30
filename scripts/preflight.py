#!/usr/bin/env python3
"""Preflight for the talk: run it 30 minutes before, from the repo root, with the same .env as the app.

    python scripts/preflight.py

For every setup the app would show, it checks the endpoint (/v1/models lists the served name, and a
1-token completion answers), sends 3 warm-up requests so the first live answer on stage isn't cold,
and confirms every preset has a recording to fall back on. A setup switched to recorded
(MODEL_<VARIANT>_MODE=recorded) skips the endpoint checks and passes only if every preset is recorded.
It ends with what the corner badge will say and one PASS/FAIL table, and exits 1 if anything failed.
It needs no cluster credentials, only the endpoints in .env.
"""

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ask import PRESETS  # noqa: E402
from app.config import ALL_VARIANTS, settings, variant_label  # noqa: E402
from app.quality import recorded_answer  # noqa: E402

WARMUPS = 3
WARMUP_PROMPT = "Say hello in five words."


def configured_variants() -> list[str]:
    """The setups the dashboard shows: the base three, plus FP8 when it has an endpoint or is recorded."""
    fp8 = bool(settings.model_fp8_endpoint) or settings.mode_for("FP8") == "recorded"
    return [k for k in ALL_VARIANTS if k != "FP8" or fp8]


def short(e: Exception) -> str:
    """One line per failure, so the table stays readable."""
    if isinstance(e, httpx.HTTPStatusError):
        return f"HTTP {e.response.status_code} from {e.request.url}"
    return f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"[:120]


def models_url(endpoint: str) -> str:
    return endpoint.rsplit("/chat/completions", 1)[0] + "/models"


def check_recordings(key: str) -> tuple[str, str]:
    captures = settings.captures_for(key)
    missing = [label for scenario, label in PRESETS.values() if recorded_answer(captures, scenario) is None]
    if missing:
        return "FAIL", f"no recording in quality/{captures}/ for: {', '.join(missing)}"
    return "PASS", f"all {len(PRESETS)} presets recorded in quality/{captures}/"


def check_build(key: str) -> tuple[str, str] | None:
    """Warn when the INT4 endpoint and the INT4 recordings are different builds."""
    if key != "INT4":
        return None
    name = settings.served_name_for(key).lower()
    captures = settings.captures_for(key)
    looks_rh = "rh" in name.replace("-", " ").replace("_", " ").split()
    if looks_rh != (captures == "INT4_RH"):
        return "WARN", f"served name {name!r} but recordings come from quality/{captures}/"
    return "PASS", f"served name {name!r} matches quality/{captures}/"


def check_live(client: httpx.Client, key: str) -> list[tuple[str, str, str]]:
    rows = []
    endpoint = settings.endpoint_for(key)
    name = settings.served_name_for(key)
    if not endpoint:
        return [("endpoint", "FAIL", f"MODEL_{key}_ENDPOINT is empty")]
    try:
        served = [m["id"] for m in client.get(models_url(endpoint)).raise_for_status().json()["data"]]
        ok = name in served
        rows.append(("models list", "PASS" if ok else "FAIL",
                     f"serves {name!r}" if ok else f"{name!r} not in {served}"))
    except Exception as e:  # noqa: BLE001 - every failure is reported, none stops the run
        return [("models list", "FAIL", short(e))]
    body = {"model": name, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1, "temperature": 0}
    try:
        start = time.perf_counter()
        client.post(endpoint, json=body).raise_for_status()
        rows.append(("1-token completion", "PASS", f"{(time.perf_counter() - start) * 1000:.0f} ms"))
    except Exception as e:  # noqa: BLE001
        return rows + [("1-token completion", "FAIL", short(e))]
    times = []
    for _ in range(WARMUPS):
        warm = {**body, "messages": [{"role": "user", "content": WARMUP_PROMPT}], "max_tokens": 16}
        try:
            start = time.perf_counter()
            client.post(endpoint, json=warm).raise_for_status()
            times.append((time.perf_counter() - start) * 1000)
        except Exception as e:  # noqa: BLE001
            return rows + [("warm-up", "FAIL", short(e))]
    rows.append(("warm-up", "PASS", f"{WARMUPS} requests, " + ", ".join(f"{t:.0f} ms" for t in times)))
    return rows


def badge_text() -> str:
    if settings.simulation_mode:
        return "Replay"
    keys = configured_variants()
    recorded = [variant_label(k) for k in keys if settings.mode_for(k) == "recorded"]
    if not recorded:
        return "Live models"
    live = [variant_label(k) for k in keys if settings.mode_for(k) != "recorded"]
    return (f"Live: {', '.join(live)} · " if live else "") + f"Recorded: {', '.join(recorded)}"


def run(client: httpx.Client) -> list[tuple[str, str, str, str]]:
    rows = []
    for key in configured_variants():
        label = variant_label(key)
        if settings.mode_for(key) == "recorded":
            rows.append((label, "mode", "PASS", "recorded by plan, endpoint not checked"))
        else:
            rows += [(label, check, result, detail) for check, result, detail in check_live(client, key)]
        rows.append((label, "recordings", *check_recordings(key)))
        if build := check_build(key):
            rows.append((label, "INT4 build", *build))
    badge = badge_text()
    rows.append(("app", "corner badge", "FAIL" if badge == "Replay" else "PASS",
                 f'"{badge}"' + (" (SIMULATION_MODE is on)" if badge == "Replay" else "")))
    return rows


def main() -> int:
    token = settings.openshift_ai_token
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(timeout=30.0, headers=headers) as client:
        rows = run(client)
    widths = [max(len(r[i]) for r in rows + [("Setup", "Check", "Result", "Detail")]) for i in range(3)]
    print(f"{'Setup':{widths[0]}}  {'Check':{widths[1]}}  {'Result':{widths[2]}}  Detail")
    for setup, check, result, detail in rows:
        print(f"{setup:{widths[0]}}  {check:{widths[1]}}  {result:{widths[2]}}  {detail}")
    failed = [r for r in rows if r[2] == "FAIL"]
    print(f"\n{'FAIL' if failed else 'PASS'}: {len(failed)} failed, "
          f"{sum(r[2] == 'WARN' for r in rows)} warnings, {len(rows)} checks")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
