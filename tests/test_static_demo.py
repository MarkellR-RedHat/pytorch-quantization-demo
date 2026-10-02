"""demo/, the replay-only static copy of the presenter: built from the app's own data and never allowed
to drift from it."""

import importlib.util
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app import main, tracks

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_static_demo", ROOT / "scripts" / "build_static_demo.py")
build_static_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_static_demo)


def data_of(js: str) -> dict:
    """The JSON a demo/data/<track>.js file assigns."""
    m = re.search(r"window\.DEMO_DATA\[\"\w+\"\] = (\{.*\});\n$", js, re.S)
    return json.loads(m.group(1))


def test_each_track_config_is_the_apps_own(tmp_path):
    files = build_static_demo.build(tmp_path)
    assert tracks.active().key == "llama"
    with TestClient(main.app) as client:
        for key in ("llama", "qwen"):
            tracks.select(key)
            assert data_of(files[f"data/{key}.js"])["config"] == client.get("/api/config").json()
    tracks.select("llama")


def test_every_preset_on_screen_has_a_script_for_every_setup(tmp_path):
    files = build_static_demo.build(tmp_path)
    for key, presets in (("llama", 7), ("qwen", 8)):
        data = data_of(files[f"data/{key}.js"])
        config, recordings = data["config"], data["recordings"]
        assert len(config["presets"]) == presets and config["mode"] == "simulated"
        for v in config["variants"]:
            for p in config["presets"]:
                script = recordings[v["key"]][p["key"]]
                assert script[0]["t"] == "start" and script[0]["text_source"] == "captured"
                assert script[-1]["t"] == "done" and script[-1]["source"] == "replay"
                assert script[-1]["tokens_per_second"] > 0
                assert script[-1]["tps_basis"] in ("recorded", "benchmark")
                deltas = [e for e in script if e["t"] == "delta"]
                assert deltas and all(e.get("delay_ms", 0) >= 0 for e in deltas)
                assert sum(e.get("delay_ms", 0) for e in script) >= 100  # paced, not dumped


def test_the_page_is_the_presenter_template_with_relative_paths(tmp_path):
    files = build_static_demo.build(tmp_path)
    html = files["index.html"]
    template = (ROOT / "templates" / "presenter.html").read_text()
    assert '"/static/' not in html and "http://" not in html.replace("http://localhost", "")
    assert 'src="../static/js/presenter.js"' in html and 'src="data/llama.js"' in html
    assert 'src="data/qwen.js"' in html and "window.DEMO_STATIC" in html
    # fonts travel inline, since file:// can't load a font file and nothing may come from the network
    assert 'rel="preload"' not in html and 'href="fonts.css"' in html
    css = files["fonts.css"]
    assert css.count("data:font/woff2;base64,") == 3 and "url('../" not in css
    # the same markup as the live page, asset paths aside
    assert html.count('<section class="scene') == template.count('<section class="scene')
    assert 'id="footLicense"' in html


def test_fonts_resolve_from_the_stylesheet():
    css = (ROOT / "static" / "css" / "base.css").read_text()
    assert "url('/static/" not in css and css.count("url('../fonts/") == 3


def test_demo_folder_is_current():
    """The gate: a rebuild must match what's committed, so demo/ never lags the app's data."""
    assert build_static_demo.stale() == []


def test_build_is_deterministic(tmp_path):
    a = build_static_demo.build(tmp_path / "a")
    b = build_static_demo.build(tmp_path / "b")
    assert a == b
