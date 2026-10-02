#!/usr/bin/env python3
"""Build demo/, the replay-only copy of the presenter that GitHub Pages serves and that opens from file://.

    python scripts/build_static_demo.py            # writes demo/
    python scripts/build_static_demo.py --check    # exits 1 if demo/ differs from a fresh build

Nothing is rendered twice: demo/index.html is the presenter template with its asset paths made relative,
and demo/data/<track>.js holds, per track, the app's own /api/config answer (replay mode, that track
selected) and, per setup and preset, the event script the server's replay stream would have sent (the
text chunks with their delays, the timing basis, the labels). static/js/presenter.js plays those scripts
when window.DEMO_STATIC is set and talks to the server otherwise. tests/test_static_demo.py rebuilds and
compares, so demo/ can't drift from the app.
"""

import asyncio
import base64
import json
import random
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

from app import ask, main, tracks  # noqa: E402
from app.config import settings  # noqa: E402

DEMO = ROOT / "demo"


def event_script(variant: str, preset: str) -> list[dict]:
    """The replay stream for one column and preset, as a list of events, each carrying the delay the
    server would have waited before sending it. The pacing jitter is seeded so a rebuild is identical."""
    pending = 0.0
    events = []

    async def record_sleep(seconds: float) -> None:
        nonlocal pending
        pending += seconds

    async def collect() -> None:
        nonlocal pending
        original = ask._sleep
        ask._sleep = record_sleep
        try:
            async for chunk in ask.replay_stream(variant, preset, main.bench()):
                for line in chunk.decode().splitlines():
                    if line.strip():
                        ev = json.loads(line)
                        if pending:
                            ev["delay_ms"] = round(pending * 1000)
                            pending = 0.0
                        events.append(ev)
        finally:
            ask._sleep = original

    random.seed(f"{tracks.active().key}/{variant}/{preset}")
    asyncio.run(collect())
    return events


def track_data(client: TestClient, key: str) -> dict:
    tracks.select(key)
    config = client.get("/api/config").json()
    recordings = {
        v["key"]: {p["key"]: event_script(v["key"], p["key"]) for p in config["presets"]}
        for v in config["variants"]
    }
    return {"config": config, "recordings": recordings}


FONTS = {"Red Hat Display": "RedHatDisplay-VF.woff2", "Red Hat Text": "RedHatText-VF.woff2",
         "Red Hat Mono": "RedHatMono-VF.woff2"}


def fonts_css() -> str:
    """The three Red Hat fonts as data URIs: a page opened from file:// can't load a font file from disk
    (browsers treat it as cross-origin), and nothing may come from the network."""
    rules = []
    for family, name in FONTS.items():
        data = base64.b64encode((ROOT / "static" / "fonts" / name).read_bytes()).decode()
        rules.append(f"@font-face {{ font-family: '{family}'; font-weight: 300 900; font-style: normal; "
                     f"src: url('data:font/woff2;base64,{data}') format('woff2'); }}")
    head = "/* built by scripts/build_static_demo.py from static/fonts/, SIL Open Font License */\n"
    return head + "\n".join(rules) + "\n"


def page(track_keys: list[str]) -> str:
    env = Environment(loader=FileSystemLoader(str(ROOT / "templates")), autoescape=True)
    html = env.get_template("presenter.html").render(title=tracks.TITLE)
    html = html.replace('"/static/', '"../static/')
    # the font preloads point at files; the static copy carries the fonts inline in fonts.css instead
    html = re.sub(r'\s*<link rel="preload"[^>]*as="font"[^>]*>', "", html)
    presenter_css = '<link rel="stylesheet" href="../static/css/presenter.css">'
    html = html.replace(presenter_css, presenter_css + '\n    <link rel="stylesheet" href="fonts.css">')
    boot = (
        "<script>\n"
        "    // the static copy: the track from ?track=, llama otherwise; Q switches tracks\n"
        "    window.DEMO_STATIC = { track: new URLSearchParams(location.search).get('track') || 'llama' };\n"
        "</script>\n"
        + "".join(f'<script src="data/{key}.js"></script>\n' for key in track_keys)
    )
    marker = '<script src="../static/js/presenter.js"></script>'
    assert marker in html, "the presenter template no longer loads presenter.js as this expects"
    return html.replace(marker, boot + marker)


def build(out: Path) -> dict[str, str]:
    """Every file of the static demo as {relative path: content}."""
    files = {}
    with TestClient(main.app) as client:
        main.simulator.enable()
        start = tracks.active().key
        keys = [t.key for t in tracks.TRACKS.values() if t.status == "ready"]
        try:
            for key in keys:
                data = track_data(client, key)
                body = json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
                files[f"data/{key}.js"] = (
                    "// built by scripts/build_static_demo.py from the app's data; do not edit\n"
                    "window.DEMO_DATA = window.DEMO_DATA || {};\n"
                    f"window.DEMO_DATA[{json.dumps(key)}] = {body};\n"
                )
        finally:
            tracks.select(start)
    files["index.html"] = page(keys)
    files["fonts.css"] = fonts_css()
    for rel, content in files.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return files


def stale() -> list[str]:
    """The files that differ between demo/ and a fresh build (empty when demo/ is current)."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = build(Path(tmp))
    out = []
    for rel, content in fresh.items():
        current = DEMO / rel
        if not current.is_file() or current.read_text() != content:
            out.append(rel)
    for path in DEMO.rglob("*"):
        if path.is_file() and str(path.relative_to(DEMO)) not in fresh:
            out.append(str(path.relative_to(DEMO)) + " (not built any more)")
    return out


if __name__ == "__main__":
    settings.presenter_key = ""
    if "--check" in sys.argv:
        diff = stale()
        if diff:
            print("demo/ is stale; run scripts/build_static_demo.py. Differs: " + ", ".join(diff))
            sys.exit(1)
        print("demo/ is current")
        sys.exit(0)
    written = build(DEMO)
    for rel in written:
        print(f"demo/{rel}  {len(written[rel]):,} bytes")
